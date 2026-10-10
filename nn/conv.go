package nn

import "fmt"

// Conv2D is a 2-D convolution layer with TensorFlow padding="same"
// semantics, set up for one input shape.
//
// For every spatial axis with input length in, kernel size k and stride s,
// the output length is ceil(in/s) and TensorFlow pads
//
//	total  = max((out-1)*s + k - in, 0)
//	before = total / 2 (floor)
//	after  = total - before
//
// so an odd total puts the extra zero after the data. For the stride-3
// frequency convolutions of basic-pitch (docs/port-spec.md §4), in=264 gives
// out=88 and total = k-3: 4 (2 before, 2 after) for k=7 and 2 (1, 1) for
// k=5, both symmetric. An even kernel would be asymmetric: k=4 pads (0, 1)
// and k=6 pads (1, 2).
//
// The convolution is computed as im2col plus a blocked float32 GEMM: for
// each block of output positions of one row, the receptive fields are
// unrolled into contiguous rows of length KH*KW*CIn (in HWIO order), and a
// register-blocked kernel takes their dot products with the repacked
// weights. Scratch space is allocated once in [NewConv2D], so
// [Conv2D.ForwardRows] does not allocate. A Conv2D is not safe for
// concurrent use because of that scratch space.
type Conv2D struct {
	in, out          Shape
	kh, kw           int
	strideH, strideW int
	padTop, padLeft  int
	padBottom        int
	padRight         int

	// k is the GEMM depth, kh*kw*in.C.
	k int
	// wt holds the weights transposed to [out.C][k], one contiguous
	// k-vector per output channel.
	wt   []float32
	bias []float32
	// col is the im2col scratch for one block of output positions.
	col []float32
}

// ConvOption configures a [Conv2D].
type ConvOption func(*convConfig)

type convConfig struct {
	strideH, strideW int
}

// WithStride sets the stride along H (time) and W (frequency). The default
// is (1, 1).
func WithStride(h, w int) ConvOption {
	return func(c *convConfig) {
		c.strideH = h
		c.strideW = w
	}
}

// NewConv2D sets up a convolution of an input of shape in with a kh×kw
// kernel producing cout channels.
//
// weights is in HWIO layout, [kh][kw][in.C][cout], so element
// (ky, kx, ci, co) is at ((ky*kw+kx)*in.C+ci)*cout+co. bias has cout
// elements or is nil for no bias. Both are copied, so the caller may reuse
// them.
func NewConv2D(in Shape, kh, kw, cout int, weights, bias []float32, opts ...ConvOption) (*Conv2D, error) {
	cfg := convConfig{strideH: 1, strideW: 1}
	for _, opt := range opts {
		opt(&cfg)
	}

	if !in.valid() {
		return nil, fmt.Errorf("nn: conv2d: invalid input shape %v", in)
	}

	if kh <= 0 || kw <= 0 || cout <= 0 {
		return nil, fmt.Errorf("nn: conv2d: invalid kernel %dx%d with %d output channels", kh, kw, cout)
	}

	if cfg.strideH <= 0 || cfg.strideW <= 0 {
		return nil, fmt.Errorf("nn: conv2d: invalid stride (%d,%d)", cfg.strideH, cfg.strideW)
	}

	k := kh * kw * in.C
	if len(weights) != k*cout {
		return nil, fmt.Errorf("nn: conv2d: %d weights, want %d for HWIO (%d,%d,%d,%d)",
			len(weights), k*cout, kh, kw, in.C, cout)
	}

	if bias != nil && len(bias) != cout {
		return nil, fmt.Errorf("nn: conv2d: %d biases, want %d", len(bias), cout)
	}

	outH, padTop, padBottom := samePadding(in.H, kh, cfg.strideH)
	outW, padLeft, padRight := samePadding(in.W, kw, cfg.strideW)

	c := &Conv2D{
		in:        in,
		out:       Shape{H: outH, W: outW, C: cout},
		kh:        kh,
		kw:        kw,
		strideH:   cfg.strideH,
		strideW:   cfg.strideW,
		padTop:    padTop,
		padBottom: padBottom,
		padLeft:   padLeft,
		padRight:  padRight,
		k:         k,
		wt:        make([]float32, k*cout),
		bias:      make([]float32, cout),
		col:       make([]float32, blockM*k),
	}

	for i := range k {
		for co := range cout {
			c.wt[co*k+i] = weights[i*cout+co]
		}
	}

	copy(c.bias, bias)

	return c, nil
}

// samePadding returns the output length and the padding before and after
// for TensorFlow padding="same".
func samePadding(in, k, s int) (out, before, after int) {
	out = (in + s - 1) / s
	total := max((out-1)*s+k-in, 0)
	before = total / 2

	return out, before, total - before
}

// InShape returns the input shape the layer was set up for.
func (c *Conv2D) InShape() Shape { return c.in }

// OutShape returns the output shape, (ceil(H/strideH), ceil(W/strideW), cout).
func (c *Conv2D) OutShape() Shape { return c.out }

// Padding returns the implicit zero padding on each side of the input.
func (c *Conv2D) Padding() (top, bottom, left, right int) {
	return c.padTop, c.padBottom, c.padLeft, c.padRight
}

// MACsPerRow returns the number of multiply-accumulates for one output row.
func (c *Conv2D) MACsPerRow() int { return c.out.W * c.k * c.out.C }

// Forward computes all output rows. It is ForwardRows(dst, src, 0, OutShape().H).
func (c *Conv2D) Forward(dst, src *Tensor) error {
	return c.ForwardRows(dst, src, 0, c.out.H)
}

// ForwardRows computes output rows [r0, r1) of dst from src. The rows are
// independent, so a caller can split a full forward pass into steps of any
// size; every row costs [Conv2D.MACsPerRow] multiply-accumulates. src must
// have [Conv2D.InShape] and dst [Conv2D.OutShape]. Other rows of dst are
// left untouched.
func (c *Conv2D) ForwardRows(dst, src *Tensor, r0, r1 int) error {
	if err := src.check("conv2d: src"); err != nil {
		return err
	}

	if err := dst.check("conv2d: dst"); err != nil {
		return err
	}

	if src.Shape != c.in {
		return fmt.Errorf("nn: conv2d: src shape %v, want %v", src.Shape, c.in)
	}

	if dst.Shape != c.out {
		return fmt.Errorf("nn: conv2d: dst shape %v, want %v", dst.Shape, c.out)
	}

	if r0 < 0 || r1 > c.out.H || r0 > r1 {
		return fmt.Errorf("nn: conv2d: rows [%d,%d) outside [0,%d)", r0, r1, c.out.H)
	}

	for y := r0; y < r1; y++ {
		c.row(dst.Rows(y, y+1), src, y)
	}

	return nil
}

// row computes output row y into out (out.W*out.C values).
func (c *Conv2D) row(out []float32, src *Tensor, y int) {
	for x0 := 0; x0 < c.out.W; x0 += blockM {
		m := min(blockM, c.out.W-x0)
		c.im2col(src, y, x0, m)
		gemm(out[x0*c.out.C:(x0+m)*c.out.C], c.col[:m*c.k], c.wt, c.bias, m, c.k)
	}
}

// im2col unrolls the receptive fields of the m output positions
// (y, x0..x0+m-1) into the first m rows of c.col. Row i holds the input
// patch in (ky, kx, ci) order, with zeros where the patch reaches into the
// padding.
func (c *Conv2D) im2col(src *Tensor, y, x0, m int) {
	cin := c.in.C
	seg := c.kw * cin // one kernel row of a patch

	for i := range m {
		dst := c.col[i*c.k : (i+1)*c.k]
		ix0 := (x0+i)*c.strideW - c.padLeft
		// Valid kernel columns [kx0, kx1) stay inside [0, in.W).
		kx0 := max(0, -ix0)
		kx1 := min(c.kw, c.in.W-ix0)

		for ky := range c.kh {
			d := dst[ky*seg : (ky+1)*seg]

			iy := y*c.strideH - c.padTop + ky
			if iy < 0 || iy >= c.in.H || kx0 >= kx1 {
				clear(d)

				continue
			}

			clear(d[:kx0*cin])
			srcRow := src.Data[iy*c.in.W*cin:]
			copy(d[kx0*cin:kx1*cin], srcRow[(ix0+kx0)*cin:(ix0+kx1)*cin])
			clear(d[kx1*cin:])
		}
	}
}
