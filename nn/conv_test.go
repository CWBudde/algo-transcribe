package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"strings"
	"testing"
)

// convReducedWork reports whether expensive tests should check a subset of
// the work: under -short and under js/wasm, where the float64 reference of a
// full basic-pitch layer is too slow.
func convReducedWork() bool {
	return testing.Short() || runtime.GOARCH == "wasm"
}

// convRNG returns a deterministic generator for a test case.
func convRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
}

// convRandFloats returns n values uniformly distributed in [-1, 1).
func convRandFloats(rng *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}

	return v
}

// refSamePad computes TensorFlow padding="same" from first principles,
// independently of the production samePadding.
func refSamePad(in, k, s int) (out, before, after int) {
	out = in / s
	if in%s != 0 {
		out++
	}

	total := (out-1)*s + k - in
	if total < 0 {
		total = 0
	}

	before = total / 2
	after = total - before

	return out, before, after
}

// refConv2D is a direct float64 convolution with TF "same" padding. src is
// NHWC, w is HWIO [kh][kw][cin][cout], b has cout elements or is nil. Only
// the output rows in rows are computed (all rows if rows is nil); the other
// entries of want and sumAbs stay zero.
//
// sumAbs[i] = |bias| + Σ|w·x| over the terms of output i bounds the
// magnitude of every partial sum, so float32 rounding in any summation order
// stays within a small multiple of eps·sumAbs.
func refConv2D(src *Tensor, kh, kw, cout int, w, b []float32, sh, sw int, rows []int) (out Shape, want, sumAbs []float64) {
	inH, inW, cin := src.H, src.W, src.C
	outH, padTop, _ := refSamePad(inH, kh, sh)
	outW, padLeft, _ := refSamePad(inW, kw, sw)
	out = Shape{H: outH, W: outW, C: cout}
	want = make([]float64, out.Size())
	sumAbs = make([]float64, out.Size())

	if rows == nil {
		rows = make([]int, outH)
		for i := range rows {
			rows[i] = i
		}
	}

	acc := make([]float64, cout)
	abs := make([]float64, cout)

	for _, y := range rows {
		for x := range outW {
			for co := range cout {
				acc[co], abs[co] = 0, 0
				if b != nil {
					acc[co] = float64(b[co])
					abs[co] = math.Abs(float64(b[co]))
				}
			}

			for ky := range kh {
				iy := y*sh - padTop + ky
				if iy < 0 || iy >= inH {
					continue
				}

				for kx := range kw {
					ix := x*sw - padLeft + kx
					if ix < 0 || ix >= inW {
						continue
					}

					for ci := range cin {
						xv := float64(src.Data[(iy*inW+ix)*cin+ci])
						wrow := w[((ky*kw+kx)*cin+ci)*cout:][:cout]

						for co, wv := range wrow {
							p := xv * float64(wv)
							acc[co] += p
							abs[co] += math.Abs(p)
						}
					}
				}
			}

			base := (y*outW + x) * cout
			copy(want[base:base+cout], acc)
			copy(sumAbs[base:base+cout], abs)
		}
	}

	return out, want, sumAbs
}

// assertConvClose compares got against the reference on the given rows (all
// rows if rows is nil) with |got-want| <= 1e-5*sumAbs + 1e-6.
func assertConvClose(t *testing.T, got *Tensor, want, sumAbs []float64, rows []int) {
	t.Helper()

	if rows == nil {
		rows = make([]int, got.H)
		for i := range rows {
			rows[i] = i
		}
	}

	n := got.W * got.C
	bad := 0

	var msgs []string

	for _, y := range rows {
		for i := y * n; i < (y+1)*n; i++ {
			g := float64(got.Data[i])
			tol := 1e-5*sumAbs[i] + 1e-6

			if math.Abs(g-want[i]) <= tol {
				continue
			}

			bad++

			if len(msgs) < 5 {
				x, c := (i-y*n)/got.C, (i-y*n)%got.C
				msgs = append(msgs, fmt.Sprintf("(%d,%d,%d): got %v, want %v (tol %.3g)", y, x, c, g, want[i], tol))
			}
		}
	}

	if bad > 0 {
		t.Fatalf("%d mismatches, first: %s", bad, strings.Join(msgs, "; "))
	}
}

type convCase struct {
	name   string
	in     Shape
	kh, kw int
	cout   int
	sh, sw int
	noBias bool
	seed   uint64
}

// runConvCase builds the layer with random data, checks the output shape
// against the reference and compares Forward with refConv2D.
func runConvCase(t *testing.T, tc convCase) {
	t.Helper()

	rng := convRNG(tc.seed + 1)
	w := convRandFloats(rng, tc.kh*tc.kw*tc.in.C*tc.cout)

	var b []float32
	if !tc.noBias {
		b = convRandFloats(rng, tc.cout)
	}

	src, err := TensorFrom(convRandFloats(rng, tc.in.Size()), tc.in.H, tc.in.W, tc.in.C)
	if err != nil {
		t.Fatal(err)
	}

	conv, err := NewConv2D(tc.in, tc.kh, tc.kw, tc.cout, w, b, WithStride(tc.sh, tc.sw))
	if err != nil {
		t.Fatal(err)
	}

	outShape, want, sumAbs := refConv2D(src, tc.kh, tc.kw, tc.cout, w, b, tc.sh, tc.sw, nil)
	if conv.OutShape() != outShape {
		t.Fatalf("OutShape() = %v, want %v", conv.OutShape(), outShape)
	}

	if conv.InShape() != tc.in {
		t.Fatalf("InShape() = %v, want %v", conv.InShape(), tc.in)
	}

	dst := NewTensor(outShape.H, outShape.W, outShape.C)
	if err := conv.Forward(dst, src); err != nil {
		t.Fatal(err)
	}

	assertConvClose(t, dst, want, sumAbs, nil)
}

func TestConv2DMatchesReference(t *testing.T) {
	cases := []convCase{
		{name: "1x1/s1", in: Shape{5, 7, 3}, kh: 1, kw: 1, cout: 4, sh: 1, sw: 1},
		{name: "1x1/s1x3", in: Shape{4, 10, 8}, kh: 1, kw: 1, cout: 5, sh: 1, sw: 3},
		{name: "1x1/s3x3/stride>kernel", in: Shape{7, 8, 3}, kh: 1, kw: 1, cout: 9, sh: 3, sw: 3},
		{name: "1x1/1x1x1", in: Shape{1, 1, 1}, kh: 1, kw: 1, cout: 1, sh: 1, sw: 1},
		{name: "1x3/s1", in: Shape{3, 9, 3}, kh: 1, kw: 3, cout: 2, sh: 1, sw: 1},
		{name: "1x3/s1x2/nilbias", in: Shape{2, 9, 1}, kh: 1, kw: 3, cout: 33, sh: 1, sw: 2, noBias: true},
		{name: "3x1/s1", in: Shape{6, 5, 1}, kh: 3, kw: 1, cout: 3, sh: 1, sw: 1},
		{name: "3x1/s2x1/nilbias", in: Shape{7, 5, 3}, kh: 3, kw: 1, cout: 8, sh: 2, sw: 1, noBias: true},
		{name: "2x2/s1/asym", in: Shape{5, 6, 3}, kh: 2, kw: 2, cout: 7, sh: 1, sw: 1},
		{name: "2x2/s2x2/asym", in: Shape{7, 9, 3}, kh: 2, kw: 2, cout: 8, sh: 2, sw: 2},
		{name: "3x3/s1", in: Shape{6, 11, 8}, kh: 3, kw: 3, cout: 9, sh: 1, sw: 1},
		{name: "3x3/s2x2", in: Shape{9, 10, 3}, kh: 3, kw: 3, cout: 33, sh: 2, sw: 2},
		{name: "3x3/s1x3/w-not-divisible", in: Shape{4, 17, 8}, kh: 3, kw: 3, cout: 32, sh: 1, sw: 3},
		{name: "4x4/s1/asym", in: Shape{6, 7, 1}, kh: 4, kw: 4, cout: 3, sh: 1, sw: 1},
		{name: "4x4/s1x3/asym", in: Shape{6, 13, 3}, kh: 4, kw: 4, cout: 5, sh: 1, sw: 3},
		{name: "4x4/s3x3", in: Shape{10, 11, 1}, kh: 4, kw: 4, cout: 4, sh: 3, sw: 3},
		{name: "5x5/s1/H=1", in: Shape{1, 7, 3}, kh: 5, kw: 5, cout: 2, sh: 1, sw: 1},
		{name: "5x5/s1/W=2", in: Shape{4, 2, 3}, kh: 5, kw: 5, cout: 3, sh: 1, sw: 1},
		{name: "5x5/s1x3/w-not-divisible", in: Shape{7, 23, 8}, kh: 5, kw: 5, cout: 32, sh: 1, sw: 3},
		{name: "5x5/s2x1", in: Shape{9, 6, 3}, kh: 5, kw: 5, cout: 1, sh: 2, sw: 1},
		{name: "5x5/s2x2/h-not-divisible", in: Shape{11, 9, 1}, kh: 5, kw: 5, cout: 9, sh: 2, sw: 2},
		{name: "7x3/s1", in: Shape{10, 9, 8}, kh: 7, kw: 3, cout: 1, sh: 1, sw: 1},
		{name: "7x3/s3x3/kernel>H", in: Shape{5, 4, 3}, kh: 7, kw: 3, cout: 7, sh: 3, sw: 3},
		{name: "7x7/s1x3", in: Shape{9, 26, 1}, kh: 7, kw: 7, cout: 32, sh: 1, sw: 3},
		{name: "3x39/s1", in: Shape{4, 45, 2}, kh: 3, kw: 39, cout: 8, sh: 1, sw: 1},
		{name: "3x39/s1/kernel>W", in: Shape{3, 20, 1}, kh: 3, kw: 39, cout: 3, sh: 1, sw: 1},
		{name: "3x39/s1x3", in: Shape{3, 50, 3}, kh: 3, kw: 39, cout: 5, sh: 1, sw: 3},
		{name: "9x9/kernel>input", in: Shape{1, 1, 3}, kh: 9, kw: 9, cout: 4, sh: 1, sw: 1},
		{name: "6x6/s2x3/kernel>input", in: Shape{2, 3, 8}, kh: 6, kw: 6, cout: 33, sh: 2, sw: 3},
	}

	for i, tc := range cases {
		tc.seed = uint64(i)
		t.Run(tc.name, func(t *testing.T) {
			runConvCase(t, tc)
		})
	}
}

// TestConv2DBlockRemainders sweeps output widths and channel counts so that
// every remainder modulo 4 and 8 of both hits the GEMM edge handling, and the
// GEMM depth k = kh*kw*cin takes many residues as well.
func TestConv2DBlockRemainders(t *testing.T) {
	couts := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 32, 33}
	cins := []int{1, 3, 8}
	seed := uint64(1000)

	for w := 1; w <= 17; w++ {
		for j, cout := range couts {
			cin := cins[(w+j)%len(cins)]
			sw := 1 + (w+j)%3

			tc := convCase{in: Shape{3, w * sw, cin}, kh: 3, kw: 3, cout: cout, sh: 1, sw: sw, seed: seed}
			tc.name = fmt.Sprintf("outW=%d/cout=%d/cin=%d/sw=%d", w, cout, cin, sw)
			seed++

			t.Run(tc.name, func(t *testing.T) {
				runConvCase(t, tc)
			})
		}
	}

	// GEMM depth sweep: k = cin for 1x1, 3*cin for 1x3.
	for cin := 1; cin <= 17; cin++ {
		for _, kw := range []int{1, 3} {
			tc := convCase{in: Shape{2, 6, cin}, kh: 1, kw: kw, cout: 5, sh: 1, sw: 1, seed: seed}
			tc.name = fmt.Sprintf("k=%d/cin=%d/kw=%d", kw*cin, cin, kw)
			seed++

			t.Run(tc.name, func(t *testing.T) {
				runConvCase(t, tc)
			})
		}
	}
}

// TestConv2DGolden pins the HWIO weight layout and the TF placement of the
// padding with hand-computed outputs that are exact in float32.
func TestConv2DGolden(t *testing.T) {
	cases := []struct {
		name    string
		in      Shape
		src     []float32
		kh, kw  int
		cout    int
		sh, sw  int
		weights []float32
		bias    []float32
		want    []float32
	}{
		{
			// pad 1/1: [0 1 2 3 0] * [1 10 100] + 0.5
			name: "1x3/same", in: Shape{1, 3, 1}, src: []float32{1, 2, 3},
			kh: 1, kw: 3, cout: 1, sh: 1, sw: 1,
			weights: []float32{1, 10, 100}, bias: []float32{0.5},
			want: []float32{210.5, 321.5, 32.5},
		},
		{
			// out 3, total 2: pad 1/1, taps at -1..1, 1..3, 3..5.
			name: "1x3/stride2", in: Shape{1, 5, 1}, src: []float32{1, 2, 3, 4, 5},
			kh: 1, kw: 3, cout: 1, sh: 1, sw: 2,
			weights: []float32{1, 10, 100},
			want:    []float32{210, 432, 54},
		},
		{
			// HWIO [ci][co]: out[co] = 1*w[0][co] + 2*w[1][co].
			name: "1x1/hwio", in: Shape{1, 1, 2}, src: []float32{1, 2},
			kh: 1, kw: 1, cout: 3, sh: 1, sw: 1,
			weights: []float32{1, 2, 3, 10, 20, 30}, bias: []float32{0, 0.25, -1},
			want: []float32{21, 42.25, 62},
		},
		{
			// Even kernel: total 1 per axis, 0 before and 1 after.
			// [1 2; 3 4] with taps [1 10; 100 1000].
			name: "2x2/asym", in: Shape{2, 2, 1}, src: []float32{1, 2, 3, 4},
			kh: 2, kw: 2, cout: 1, sh: 1, sw: 1,
			weights: []float32{1, 10, 100, 1000},
			want:    []float32{4321, 402, 43, 4},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conv, err := NewConv2D(tc.in, tc.kh, tc.kw, tc.cout, tc.weights, tc.bias, WithStride(tc.sh, tc.sw))
			if err != nil {
				t.Fatal(err)
			}

			src, err := TensorFrom(tc.src, tc.in.H, tc.in.W, tc.in.C)
			if err != nil {
				t.Fatal(err)
			}

			o := conv.OutShape()
			dst := NewTensor(o.H, o.W, o.C)

			if err := conv.Forward(dst, src); err != nil {
				t.Fatal(err)
			}

			if len(dst.Data) != len(tc.want) {
				t.Fatalf("output has %d values, want %d", len(dst.Data), len(tc.want))
			}

			for i, v := range tc.want {
				if dst.Data[i] != v {
					t.Errorf("out[%d] = %v, want %v", i, dst.Data[i], v)
				}
			}
		})
	}
}

func TestSamePadding(t *testing.T) {
	cases := []struct {
		name                 string
		in, k, s             int
		out, before, after   int
		wantNegativeClamping bool
	}{
		{name: "bp/W264/k7/s3", in: 264, k: 7, s: 3, out: 88, before: 2, after: 2},
		{name: "bp/W264/k5/s3", in: 264, k: 5, s: 3, out: 88, before: 1, after: 1},
		// (88-1)*3 + 4 - 264 = 1: all of the padding goes after. (The Conv2D
		// doc comment claims 3 = 1+2 here; that is a doc error.)
		{name: "W264/k4/s3/asym", in: 264, k: 4, s: 3, out: 88, before: 0, after: 1},
		{name: "W264/k6/s3/asym", in: 264, k: 6, s: 3, out: 88, before: 1, after: 2},
		{name: "bp/W264/k39/s1", in: 264, k: 39, s: 1, out: 264, before: 19, after: 19},
		{name: "bp/W264/k5/s1", in: 264, k: 5, s: 1, out: 264, before: 2, after: 2},
		{name: "bp/W88/k3/s1", in: 88, k: 3, s: 1, out: 88, before: 1, after: 1},
		{name: "bp/H172/k3/s1", in: 172, k: 3, s: 1, out: 172, before: 1, after: 1},
		{name: "bp/H172/k5/s1", in: 172, k: 5, s: 1, out: 172, before: 2, after: 2},
		{name: "bp/H172/k7/s1", in: 172, k: 7, s: 1, out: 172, before: 3, after: 3},
		{name: "W264/k1/s3/stride>kernel", in: 264, k: 1, s: 3, out: 88, before: 0, after: 0, wantNegativeClamping: true},
		{name: "in10/k1/s4/negative", in: 10, k: 1, s: 4, out: 3, before: 0, after: 0, wantNegativeClamping: true},
		{name: "in10/k2/s4/zero", in: 10, k: 2, s: 4, out: 3, before: 0, after: 0},
		{name: "in7/k2/s3/odd", in: 7, k: 2, s: 3, out: 3, before: 0, after: 1},
		{name: "in3/k2/s2/odd", in: 3, k: 2, s: 2, out: 2, before: 0, after: 1},
		{name: "in6/k3/s2/odd", in: 6, k: 3, s: 2, out: 3, before: 0, after: 1},
		{name: "in5/k5/s2", in: 5, k: 5, s: 2, out: 3, before: 2, after: 2},
		{name: "in1/k5/s1/kernel>in", in: 1, k: 5, s: 1, out: 1, before: 2, after: 2},
		{name: "in2/k5/s1/kernel>in", in: 2, k: 5, s: 1, out: 2, before: 2, after: 2},
		{name: "in2/k4/s1/kernel>in", in: 2, k: 4, s: 1, out: 2, before: 1, after: 2},
		{name: "in1/k3/s3", in: 1, k: 3, s: 3, out: 1, before: 1, after: 1},
		{name: "in1/k1/s1", in: 1, k: 1, s: 1, out: 1, before: 0, after: 0},
		{name: "in4/k7/s1", in: 4, k: 7, s: 1, out: 4, before: 3, after: 3},
		{name: "in2/k1/s5/negative", in: 2, k: 1, s: 5, out: 1, before: 0, after: 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if tc.wantNegativeClamping && (tc.out-1)*tc.s+tc.k-tc.in >= 0 {
				t.Fatalf("case does not exercise a negative total")
			}

			// The independent reference must agree with the table.
			ro, rb, ra := refSamePad(tc.in, tc.k, tc.s)
			if ro != tc.out || rb != tc.before || ra != tc.after {
				t.Fatalf("refSamePad = (%d,%d,%d), want (%d,%d,%d)", ro, rb, ra, tc.out, tc.before, tc.after)
			}

			out, before, after := samePadding(tc.in, tc.k, tc.s)
			if out != tc.out || before != tc.before || after != tc.after {
				t.Errorf("samePadding(%d,%d,%d) = (%d,%d,%d), want (%d,%d,%d)",
					tc.in, tc.k, tc.s, out, before, after, tc.out, tc.before, tc.after)
			}

			// The same geometry along each axis of a layer.
			conv, err := NewConv2D(Shape{H: tc.in, W: tc.in, C: 1}, tc.k, tc.k, 2,
				make([]float32, tc.k*tc.k*2), nil, WithStride(tc.s, tc.s))
			if err != nil {
				t.Fatal(err)
			}

			top, bottom, left, right := conv.Padding()
			if top != tc.before || bottom != tc.after || left != tc.before || right != tc.after {
				t.Errorf("Padding() = (%d,%d,%d,%d), want (%d,%d,%d,%d)",
					top, bottom, left, right, tc.before, tc.after, tc.before, tc.after)
			}

			if want := (Shape{H: tc.out, W: tc.out, C: 2}); conv.OutShape() != want {
				t.Errorf("OutShape() = %v, want %v", conv.OutShape(), want)
			}
		})
	}
}

// TestConv2DBasicPitchLayers checks the exact layer shapes of
// docs/port-spec.md §4 against the float64 reference.
func TestConv2DBasicPitchLayers(t *testing.T) {
	cases := []struct {
		name       string
		in         Shape
		kh, kw     int
		cout       int
		sh, sw     int
		out        Shape
		pads       [4]int // top, bottom, left, right
		macsPerRow int
	}{
		{
			name: "conv2d_1", in: Shape{172, 264, 8}, kh: 3, kw: 39, cout: 8, sh: 1, sw: 1,
			out: Shape{172, 264, 8}, pads: [4]int{1, 1, 19, 19}, macsPerRow: 264 * 936 * 8,
		},
		{
			name: "contours-reduced", in: Shape{172, 264, 8}, kh: 5, kw: 5, cout: 1, sh: 1, sw: 1,
			out: Shape{172, 264, 1}, pads: [4]int{2, 2, 2, 2}, macsPerRow: 264 * 200 * 1,
		},
		{
			name: "conv2d_2", in: Shape{172, 264, 1}, kh: 7, kw: 7, cout: 32, sh: 1, sw: 3,
			out: Shape{172, 88, 32}, pads: [4]int{3, 3, 2, 2}, macsPerRow: 88 * 49 * 32,
		},
		{
			name: "conv2d_3", in: Shape{172, 88, 32}, kh: 7, kw: 3, cout: 1, sh: 1, sw: 1,
			out: Shape{172, 88, 1}, pads: [4]int{3, 3, 1, 1}, macsPerRow: 88 * 672 * 1,
		},
		{
			name: "conv2d_4", in: Shape{172, 264, 8}, kh: 5, kw: 5, cout: 32, sh: 1, sw: 3,
			out: Shape{172, 88, 32}, pads: [4]int{2, 2, 1, 1}, macsPerRow: 88 * 200 * 32,
		},
		{
			name: "conv2d_5", in: Shape{172, 88, 33}, kh: 3, kw: 3, cout: 1, sh: 1, sw: 1,
			out: Shape{172, 88, 1}, pads: [4]int{1, 1, 1, 1}, macsPerRow: 88 * 297 * 1,
		},
	}

	// Under -short and wasm only a subset of rows is checked: both edges
	// (which see the top/bottom padding) and a few in the middle.
	var rows []int
	if convReducedWork() {
		rows = []int{0, 1, 2, 3, 85, 86, 168, 169, 170, 171}
	}

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := convRNG(uint64(500 + i))
			w := convRandFloats(rng, tc.kh*tc.kw*tc.in.C*tc.cout)
			b := convRandFloats(rng, tc.cout)

			src, err := TensorFrom(convRandFloats(rng, tc.in.Size()), tc.in.H, tc.in.W, tc.in.C)
			if err != nil {
				t.Fatal(err)
			}

			conv, err := NewConv2D(tc.in, tc.kh, tc.kw, tc.cout, w, b, WithStride(tc.sh, tc.sw))
			if err != nil {
				t.Fatal(err)
			}

			if conv.OutShape() != tc.out {
				t.Fatalf("OutShape() = %v, want %v", conv.OutShape(), tc.out)
			}

			if got := conv.MACsPerRow(); got != tc.macsPerRow {
				t.Errorf("MACsPerRow() = %d, want %d", got, tc.macsPerRow)
			}

			top, bottom, left, right := conv.Padding()
			if got := [4]int{top, bottom, left, right}; got != tc.pads {
				t.Errorf("Padding() = %v, want %v", got, tc.pads)
			}

			dst := NewTensor(tc.out.H, tc.out.W, tc.out.C)
			if rows == nil {
				err = conv.Forward(dst, src)
			} else {
				for _, y := range rows {
					if err = conv.ForwardRows(dst, src, y, y+1); err != nil {
						break
					}
				}
			}

			if err != nil {
				t.Fatal(err)
			}

			outShape, want, sumAbs := refConv2D(src, tc.kh, tc.kw, tc.cout, w, b, tc.sh, tc.sw, rows)
			if outShape != tc.out {
				t.Fatalf("reference output shape %v, want %v", outShape, tc.out)
			}

			assertConvClose(t, dst, want, sumAbs, rows)
		})
	}
}

// newConvFixture builds a layer with random weights and a random input.
func newConvFixture(t *testing.T, in Shape, kh, kw, cout, sh, sw int, seed uint64) (*Conv2D, *Tensor) {
	t.Helper()

	rng := convRNG(seed)
	w := convRandFloats(rng, kh*kw*in.C*cout)
	b := convRandFloats(rng, cout)

	conv, err := NewConv2D(in, kh, kw, cout, w, b, WithStride(sh, sw))
	if err != nil {
		t.Fatal(err)
	}

	src, err := TensorFrom(convRandFloats(rng, in.Size()), in.H, in.W, in.C)
	if err != nil {
		t.Fatal(err)
	}

	return conv, src
}

func newFilledTensor(s Shape, v float32) *Tensor {
	t := NewTensor(s.H, s.W, s.C)
	for i := range t.Data {
		t.Data[i] = v
	}

	return t
}

func TestConv2DForwardRowsChunking(t *testing.T) {
	layers := []struct {
		name   string
		in     Shape
		kh, kw int
		cout   int
		sh, sw int
	}{
		{name: "5x5/s1x3", in: Shape{13, 23, 3}, kh: 5, kw: 5, cout: 9, sh: 1, sw: 3},
		{name: "3x3/s2x2", in: Shape{12, 10, 8}, kh: 3, kw: 3, cout: 7, sh: 2, sw: 2},
		{name: "2x2/s1", in: Shape{9, 7, 1}, kh: 2, kw: 2, cout: 33, sh: 1, sw: 1},
	}

	for i, l := range layers {
		t.Run(l.name, func(t *testing.T) {
			conv, src := newConvFixture(t, l.in, l.kh, l.kw, l.cout, l.sh, l.sw, uint64(700+i))
			o := conv.OutShape()

			full := NewTensor(o.H, o.W, o.C)
			if err := conv.Forward(full, src); err != nil {
				t.Fatal(err)
			}

			ones := make([]int, o.H)
			for j := range ones {
				ones[j] = 1
			}

			// Each chunking lists step sizes; zero steps are empty ranges.
			chunkings := map[string][]int{
				"one-row":     ones,
				"all":         {o.H},
				"uneven":      {2, 0, 3, 1, o.H},
				"empty-first": {0, 0, o.H - 1, 1, 0},
				"halves":      {o.H / 2, o.H - o.H/2},
			}

			for name, steps := range chunkings {
				t.Run(name, func(t *testing.T) {
					dst := newFilledTensor(o, float32(math.NaN()))

					r0 := 0
					for _, n := range steps {
						r1 := min(r0+n, o.H)
						if err := conv.ForwardRows(dst, src, r0, r1); err != nil {
							t.Fatal(err)
						}

						r0 = r1
					}

					if r0 != o.H {
						t.Fatalf("chunking covers %d of %d rows", r0, o.H)
					}

					assertBitIdentical(t, dst.Data, full.Data)
				})
			}

			// A partial range writes exactly rows [r0, r1).
			for _, rr := range [][2]int{{0, 0}, {o.H, o.H}, {2, 2}, {1, 4}, {0, 1}, {o.H - 1, o.H}} {
				t.Run(fmt.Sprintf("only[%d,%d)", rr[0], rr[1]), func(t *testing.T) {
					const sentinel = 12345

					dst := newFilledTensor(o, sentinel)
					if err := conv.ForwardRows(dst, src, rr[0], rr[1]); err != nil {
						t.Fatal(err)
					}

					n := o.W * o.C
					for y := range o.H {
						got := dst.Data[y*n : (y+1)*n]
						if y >= rr[0] && y < rr[1] {
							assertBitIdentical(t, got, full.Data[y*n:(y+1)*n])

							continue
						}

						for x, v := range got {
							if v != sentinel {
								t.Fatalf("row %d outside range written at %d: %v", y, x, v)
							}
						}
					}
				})
			}
		})
	}
}

func assertBitIdentical(t *testing.T, got, want []float32) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}

	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("value %d = %v (%#08x), want %v (%#08x)",
				i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestConv2DZeroAllocs(t *testing.T) {
	// conv2d_4 geometry with fewer time rows to keep the runtime small.
	conv, src := newConvFixture(t, Shape{8, 264, 8}, 5, 5, 32, 1, 3, 900)
	o := conv.OutShape()
	dst := NewTensor(o.H, o.W, o.C)

	var err error

	allocs := testing.AllocsPerRun(5, func() {
		err = conv.ForwardRows(dst, src, 2, 5)
	})

	if err != nil {
		t.Fatal(err)
	}

	if allocs != 0 {
		t.Errorf("ForwardRows: %v allocations per run, want 0", allocs)
	}

	allocs = testing.AllocsPerRun(3, func() {
		err = conv.Forward(dst, src)
	})

	if err != nil {
		t.Fatal(err)
	}

	if allocs != 0 {
		t.Errorf("Forward: %v allocations per run, want 0", allocs)
	}
}

func TestNewConv2DErrors(t *testing.T) {
	in := Shape{4, 5, 2}

	cases := []struct {
		name    string
		in      Shape
		kh, kw  int
		cout    int
		weights []float32
		bias    []float32
		opts    []ConvOption
	}{
		{name: "H=0", in: Shape{0, 5, 2}, kh: 1, kw: 1, cout: 1, weights: make([]float32, 2)},
		{name: "W<0", in: Shape{4, -1, 2}, kh: 1, kw: 1, cout: 1, weights: make([]float32, 2)},
		{name: "C=0", in: Shape{4, 5, 0}, kh: 1, kw: 1, cout: 1},
		{name: "kh=0", in: in, kh: 0, kw: 3, cout: 1},
		{name: "kw<0", in: in, kh: 3, kw: -3, cout: 1},
		{name: "cout=0", in: in, kh: 3, kw: 3, cout: 0},
		{name: "strideH=0", in: in, kh: 1, kw: 1, cout: 1, weights: make([]float32, 2), opts: []ConvOption{WithStride(0, 1)}},
		{name: "strideW<0", in: in, kh: 1, kw: 1, cout: 1, weights: make([]float32, 2), opts: []ConvOption{WithStride(1, -2)}},
		{name: "weights-short", in: in, kh: 3, kw: 3, cout: 2, weights: make([]float32, 3*3*2*2-1)},
		{name: "weights-long", in: in, kh: 3, kw: 3, cout: 2, weights: make([]float32, 3*3*2*2+1)},
		{name: "weights-nil", in: in, kh: 3, kw: 3, cout: 2},
		{name: "bias-short", in: in, kh: 1, kw: 1, cout: 3, weights: make([]float32, 6), bias: make([]float32, 2)},
		{name: "bias-long", in: in, kh: 1, kw: 1, cout: 3, weights: make([]float32, 6), bias: make([]float32, 4)},
		{name: "bias-empty-non-nil", in: in, kh: 1, kw: 1, cout: 3, weights: make([]float32, 6), bias: []float32{}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conv, err := NewConv2D(tc.in, tc.kh, tc.kw, tc.cout, tc.weights, tc.bias, tc.opts...)
			if err == nil {
				t.Fatalf("NewConv2D succeeded with %+v", conv.OutShape())
			}

			if conv != nil {
				t.Errorf("NewConv2D returned a layer with error %v", err)
			}

			if !strings.HasPrefix(err.Error(), "nn: conv2d: ") {
				t.Errorf("error %q lacks the package prefix", err)
			}
		})
	}

	// Sanity: the valid counterpart of the cases above succeeds.
	if _, err := NewConv2D(in, 3, 3, 2, make([]float32, 3*3*2*2), make([]float32, 2)); err != nil {
		t.Fatalf("valid layer: %v", err)
	}
}

func TestConv2DForwardRowsErrors(t *testing.T) {
	conv, src := newConvFixture(t, Shape{6, 7, 3}, 3, 3, 4, 2, 1, 950)
	o := conv.OutShape() // (3, 7, 4)
	dst := NewTensor(o.H, o.W, o.C)

	cases := []struct {
		name     string
		dst, src *Tensor
		r0, r1   int
	}{
		{name: "nil-src", dst: dst, src: nil, r0: 0, r1: 1},
		{name: "nil-dst", dst: nil, src: src, r0: 0, r1: 1},
		{name: "src-shape", dst: dst, src: NewTensor(6, 7, 2), r0: 0, r1: 1},
		{name: "src-transposed", dst: dst, src: NewTensor(7, 6, 3), r0: 0, r1: 1},
		{name: "dst-shape", dst: NewTensor(3, 7, 5), src: src, r0: 0, r1: 1},
		{name: "dst-unstrided", dst: NewTensor(6, 7, 4), src: src, r0: 0, r1: 1},
		{name: "src-data-short", dst: dst, src: &Tensor{Shape: src.Shape, Data: src.Data[:src.Size()-1]}, r0: 0, r1: 1},
		{name: "src-data-long", dst: dst, src: &Tensor{Shape: src.Shape, Data: make([]float32, src.Size()+1)}, r0: 0, r1: 1},
		{name: "dst-data-short", dst: &Tensor{Shape: o, Data: make([]float32, o.Size()-1)}, src: src, r0: 0, r1: 1},
		{name: "dst-data-nil", dst: &Tensor{Shape: o}, src: src, r0: 0, r1: 1},
		{name: "src-zero-shape", dst: dst, src: &Tensor{}, r0: 0, r1: 1},
		{name: "r0<0", dst: dst, src: src, r0: -1, r1: 1},
		{name: "r1>H", dst: dst, src: src, r0: 0, r1: o.H + 1},
		{name: "r0>r1", dst: dst, src: src, r0: 2, r1: 1},
		{name: "r0>H", dst: dst, src: src, r0: o.H + 1, r1: o.H + 1},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Errors must be reported before anything is written.
			if tc.dst != nil {
				for i := range tc.dst.Data {
					tc.dst.Data[i] = 7
				}
			}

			err := conv.ForwardRows(tc.dst, tc.src, tc.r0, tc.r1)
			if err == nil {
				t.Fatal("ForwardRows succeeded")
			}

			if !strings.HasPrefix(err.Error(), "nn: conv2d: ") {
				t.Errorf("error %q lacks the package prefix", err)
			}

			if tc.dst != nil {
				for i, v := range tc.dst.Data {
					if v != 7 {
						t.Fatalf("dst[%d] written on error: %v", i, v)
					}
				}
			}
		})
	}

	if err := conv.Forward(nil, src); err == nil {
		t.Error("Forward with nil dst succeeded")
	}
}

// TestConv2DCopiesParameters checks that NewConv2D copies weights and bias:
// changing the caller's slices afterwards must not change the output.
func TestConv2DCopiesParameters(t *testing.T) {
	in := Shape{5, 9, 3}
	rng := convRNG(42)
	w := convRandFloats(rng, 3*3*3*6)
	b := convRandFloats(rng, 6)

	src, err := TensorFrom(convRandFloats(rng, in.Size()), in.H, in.W, in.C)
	if err != nil {
		t.Fatal(err)
	}

	conv, err := NewConv2D(in, 3, 3, 6, w, b, WithStride(1, 2))
	if err != nil {
		t.Fatal(err)
	}

	o := conv.OutShape()
	before := NewTensor(o.H, o.W, o.C)

	if err := conv.Forward(before, src); err != nil {
		t.Fatal(err)
	}

	for i := range w {
		w[i] = 100
	}

	for i := range b {
		b[i] = -100
	}

	after := NewTensor(o.H, o.W, o.C)
	if err := conv.Forward(after, src); err != nil {
		t.Fatal(err)
	}

	assertBitIdentical(t, after.Data, before.Data)
}

// TestConv2DDoesNotModifySource checks that the input tensor is read-only.
func TestConv2DDoesNotModifySource(t *testing.T) {
	conv, src := newConvFixture(t, Shape{7, 11, 3}, 5, 5, 8, 1, 3, 77)
	orig := append([]float32(nil), src.Data...)
	o := conv.OutShape()

	if err := conv.Forward(NewTensor(o.H, o.W, o.C), src); err != nil {
		t.Fatal(err)
	}

	assertBitIdentical(t, src.Data, orig)
}
