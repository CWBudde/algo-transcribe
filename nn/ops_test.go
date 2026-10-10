package nn

import (
	"math"
	"math/rand/v2"
	"testing"
)

func float32Equal(a, b float32) bool {
	if math.IsNaN(float64(a)) || math.IsNaN(float64(b)) {
		return math.IsNaN(float64(a)) && math.IsNaN(float64(b))
	}

	return a == b
}

func inf32(sign int) float32 { return float32(math.Inf(sign)) }

func TestReLU(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   []float32
		want []float32
	}{
		{"empty", []float32{}, []float32{}},
		{"nil", nil, nil},
		{"zero", []float32{0}, []float32{0}},
		{"positive", []float32{0.5, 3, 1e30}, []float32{0.5, 3, 1e30}},
		{"negative", []float32{-0.5, -3, -1e30}, []float32{0, 0, 0}},
		{"infinities", []float32{inf32(1), inf32(-1)}, []float32{inf32(1), 0}},
		{"mixed", []float32{-1, 2, -3, 4}, []float32{0, 2, 0, 4}},
		{"nan passes through", []float32{float32(math.NaN())}, []float32{float32(math.NaN())}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			x := append([]float32(nil), tt.in...)
			ReLU(x)

			for i := range tt.want {
				if !float32Equal(x[i], tt.want[i]) {
					t.Errorf("ReLU(%v)[%d] = %v, want %v", tt.in, i, x[i], tt.want[i])
				}
			}
		})
	}
}

func TestSigmoid(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   float32
		want float32
	}{
		{"zero", 0, 0.5},
		{"one", 1, float32(1 / (1 + math.Exp(-1)))},
		{"minus one", -1, float32(1 / (1 + math.Exp(1)))},
		{"two", 2, 0.880797077977882},
		{"minus two", -2, 0.11920292202211755},
		{"plus 20", 20, float32(1 / (1 + math.Exp(-20)))},
		{"minus 20", -20, float32(1 / (1 + math.Exp(20)))},
		{"plus 100", 100, 1},
		{"minus 100", -100, float32(1 / (1 + math.Exp(100)))},
		{"minus 1000", -1000, 0},
		{"plus 1000", 1000, 1},
		{"max float32", math.MaxFloat32, 1},
		{"min float32", -math.MaxFloat32, 0},
		{"plus inf", inf32(1), 1},
		{"minus inf", inf32(-1), 0},
		{"nan", float32(math.NaN()), float32(math.NaN())},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			x := []float32{tt.in}
			Sigmoid(x)

			if !float32Equal(x[0], tt.want) {
				t.Errorf("Sigmoid(%v) = %v, want %v", tt.in, x[0], tt.want)
			}
		})
	}
}

func TestSigmoidSymmetry(t *testing.T) {
	t.Parallel()

	// sigmoid(v) + sigmoid(-v) = 1 holds up to float32 rounding.
	for v := float32(-30); v <= 30; v += 0.25 {
		x := []float32{v, -v}
		Sigmoid(x)

		if d := math.Abs(float64(x[0]+x[1]) - 1); d > 1e-7 {
			t.Errorf("sigmoid(%v)+sigmoid(%v) = %v", v, -v, x[0]+x[1])
		}

		if x[0] < 0 || x[0] > 1 {
			t.Errorf("sigmoid(%v) = %v outside [0, 1]", v, x[0])
		}
	}

	Sigmoid(nil)
	Sigmoid([]float32{})
}

// seqTensor returns a tensor whose element at offset i is base+i.
func seqTensor(h, w, c int, base float32) *Tensor {
	t := NewTensor(h, w, c)
	for i := range t.Data {
		t.Data[i] = base + float32(i)
	}

	return t
}

func TestConcatChannelsHandComputed(t *testing.T) {
	t.Parallel()

	// a: (1,2,1), b: (1,2,2) -> (1,2,3), per pixel [a | b].
	a, err := TensorFrom([]float32{1, 2}, 1, 2, 1)
	if err != nil {
		t.Fatal(err)
	}

	b, err := TensorFrom([]float32{10, 11, 20, 21}, 1, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	dst := NewTensor(1, 2, 3)
	if err := ConcatChannels(dst, a, b); err != nil {
		t.Fatal(err)
	}

	want := []float32{1, 10, 11, 2, 20, 21}
	for i, v := range want {
		if dst.Data[i] != v {
			t.Fatalf("dst = %v, want %v", dst.Data, want)
		}
	}

	// Reversed order puts b's channels first.
	if err := ConcatChannels(dst, b, a); err != nil {
		t.Fatal(err)
	}

	want = []float32{10, 11, 1, 20, 21, 2}
	for i, v := range want {
		if dst.Data[i] != v {
			t.Fatalf("dst = %v, want %v", dst.Data, want)
		}
	}
}

func TestConcatChannelsBasicPitch(t *testing.T) {
	t.Parallel()

	// docs/port-spec.md §4 layer 6: [note (172,88,1), conv2d_4 (172,88,32)].
	note := seqTensor(172, 88, 1, 0)
	feat := seqTensor(172, 88, 32, 1e6)
	dst := NewTensor(172, 88, 33)

	if err := ConcatChannels(dst, note, feat); err != nil {
		t.Fatal(err)
	}

	if want := (Shape{H: 172, W: 88, C: 33}); dst.Shape != want {
		t.Fatalf("shape %v, want %v", dst.Shape, want)
	}

	for y := range 172 {
		for x := range 88 {
			if got, want := dst.At(y, x, 0), note.At(y, x, 0); got != want {
				t.Fatalf("(%d,%d,0) = %v, want %v", y, x, got, want)
			}

			for c := range 32 {
				if got, want := dst.At(y, x, c+1), feat.At(y, x, c); got != want {
					t.Fatalf("(%d,%d,%d) = %v, want %v", y, x, c+1, got, want)
				}
			}
		}
	}
}

func TestConcatChannelsRows(t *testing.T) {
	t.Parallel()

	a := seqTensor(5, 3, 2, 0)
	b := seqTensor(5, 3, 1, 100)
	c := seqTensor(5, 3, 3, 200)

	full := NewTensor(5, 3, 6)
	if err := ConcatChannels(full, a, b, c); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name   string
		r0, r1 int
	}{
		{"empty", 2, 2},
		{"first", 0, 1},
		{"middle", 1, 4},
		{"last", 4, 5},
		{"all", 0, 5},
	}

	const sentinel = -7

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			dst := NewTensor(5, 3, 6)
			for i := range dst.Data {
				dst.Data[i] = sentinel
			}

			if err := ConcatChannelsRows(dst, tt.r0, tt.r1, a, b, c); err != nil {
				t.Fatal(err)
			}

			for y := range 5 {
				inside := y >= tt.r0 && y < tt.r1

				for i, v := range dst.Rows(y, y+1) {
					want := float32(sentinel)
					if inside {
						want = full.Rows(y, y+1)[i]
					}

					if v != want {
						t.Fatalf("row %d[%d] = %v, want %v", y, i, v, want)
					}
				}
			}
		})
	}

	// Stepping in chunks reproduces the full call.
	dst := NewTensor(5, 3, 6)
	for r := 0; r < 5; r += 2 {
		if err := ConcatChannelsRows(dst, r, min(r+2, 5), a, b, c); err != nil {
			t.Fatal(err)
		}
	}

	for i := range dst.Data {
		if dst.Data[i] != full.Data[i] {
			t.Fatalf("chunked[%d] = %v, want %v", i, dst.Data[i], full.Data[i])
		}
	}
}

func TestConcatChannelsErrors(t *testing.T) {
	t.Parallel()

	a := NewTensor(4, 3, 2)
	b := NewTensor(4, 3, 1)
	broken := &Tensor{Shape: Shape{H: 4, W: 3, C: 1}, Data: make([]float32, 5)}

	tests := []struct {
		name   string
		dst    *Tensor
		r0, r1 int
		srcs   []*Tensor
	}{
		{"nil dst", nil, 0, 4, []*Tensor{a, b}},
		{"broken dst", &Tensor{Shape: Shape{H: 4, W: 3, C: 3}}, 0, 4, []*Tensor{a, b}},
		{"nil src", NewTensor(4, 3, 3), 0, 4, []*Tensor{a, nil}},
		{"broken src", NewTensor(4, 3, 3), 0, 4, []*Tensor{a, broken}},
		{"H mismatch", NewTensor(4, 3, 3), 0, 4, []*Tensor{a, NewTensor(5, 3, 1)}},
		{"W mismatch", NewTensor(4, 3, 3), 0, 4, []*Tensor{a, NewTensor(4, 2, 1)}},
		{"too few channels", NewTensor(4, 3, 4), 0, 4, []*Tensor{a, b}},
		{"too many channels", NewTensor(4, 3, 2), 0, 4, []*Tensor{a, b}},
		{"no sources", NewTensor(4, 3, 3), 0, 4, nil},
		{"negative r0", NewTensor(4, 3, 3), -1, 2, []*Tensor{a, b}},
		{"r1 past end", NewTensor(4, 3, 3), 0, 5, []*Tensor{a, b}},
		{"r0 after r1", NewTensor(4, 3, 3), 3, 2, []*Tensor{a, b}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if err := ConcatChannelsRows(tt.dst, tt.r0, tt.r1, tt.srcs...); err == nil {
				t.Errorf("ConcatChannelsRows: no error")
			}

			// The full variant shares every check except the row range.
			if tt.r0 == 0 && tt.r1 == 4 {
				if err := ConcatChannels(tt.dst, tt.srcs...); err == nil {
					t.Errorf("ConcatChannels: no error")
				}
			}
		})
	}
}

// batchNorm applies bn to every channel of x (NHWC) in float64.
func batchNorm(x *Tensor, bn BatchNorm) {
	for i, v := range x.Data {
		c := i % x.C
		y := float64(bn.Gamma[c])*(float64(v)-float64(bn.Mean[c]))/
			math.Sqrt(float64(bn.Variance[c])+float64(bn.Epsilon)) + float64(bn.Beta[c])
		x.Data[i] = float32(y)
	}
}

func randSlice(rng *rand.Rand, n int, lo, hi float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = lo + (hi-lo)*rng.Float32()
	}

	return s
}

func TestFoldBatchNormMatchesConvThenBN(t *testing.T) {
	t.Parallel()

	const (
		kh, kw, cin, cout = 3, 3, 3, 4
	)

	in := Shape{H: 6, W: 9, C: cin}

	tests := []struct {
		name    string
		hasBias bool
		eps     float32
	}{
		{"with bias", true, 1e-3},
		{"without bias", false, 1e-3},
		{"tiny epsilon", true, 1e-5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewPCG(1, 2))

			src, err := TensorFrom(randSlice(rng, in.Size(), -1, 1), in.H, in.W, in.C)
			if err != nil {
				t.Fatal(err)
			}

			weights := randSlice(rng, kh*kw*cin*cout, -0.5, 0.5)
			bias := make([]float32, cout)

			if tt.hasBias {
				bias = randSlice(rng, cout, -0.2, 0.2)
			}

			bn := BatchNorm{
				Gamma:    randSlice(rng, cout, 0.5, 2),
				Beta:     randSlice(rng, cout, -1, 1),
				Mean:     randSlice(rng, cout, -0.3, 0.3),
				Variance: randSlice(rng, cout, 0.01, 2),
				Epsilon:  tt.eps,
			}

			ref, err := NewConv2D(in, kh, kw, cout, weights, bias, WithStride(1, 2))
			if err != nil {
				t.Fatal(err)
			}

			want := NewTensor(ref.OutShape().H, ref.OutShape().W, ref.OutShape().C)
			if err := ref.Forward(want, src); err != nil {
				t.Fatal(err)
			}

			batchNorm(want, bn)

			fw := append([]float32(nil), weights...)
			fb := append([]float32(nil), bias...)

			if err := FoldBatchNorm(fw, fb, bn); err != nil {
				t.Fatal(err)
			}

			folded, err := NewConv2D(in, kh, kw, cout, fw, fb, WithStride(1, 2))
			if err != nil {
				t.Fatal(err)
			}

			got := NewTensor(want.H, want.W, want.C)
			if err := folded.Forward(got, src); err != nil {
				t.Fatal(err)
			}

			if got.Shape != (Shape{H: 6, W: 5, C: cout}) {
				t.Fatalf("out shape %v", got.Shape)
			}

			for i := range want.Data {
				if d := math.Abs(float64(got.Data[i] - want.Data[i])); d > 1e-5*(1+math.Abs(float64(want.Data[i]))) {
					t.Fatalf("[%d] folded %v, conv+BN %v (diff %g)", i, got.Data[i], want.Data[i], d)
				}
			}
		})
	}
}

func TestFoldBatchNormValues(t *testing.T) {
	t.Parallel()

	// cout = 2, two weights per channel: HWIO puts channel 0 at even indices.
	weights := []float32{1, 10, 2, 20}
	bias := []float32{0.5, -1}
	bn := BatchNorm{
		Gamma:    []float32{2, 3},
		Beta:     []float32{1, -1},
		Mean:     []float32{0.5, 1},
		Variance: []float32{3, 8},
		Epsilon:  1,
	}

	if err := FoldBatchNorm(weights, bias, bn); err != nil {
		t.Fatal(err)
	}

	// scale = [2/sqrt(4), 3/sqrt(9)] = [1, 1], so only the bias moves:
	// (0.5-0.5)*1 + 1 = 1 and (-1-1)*1 - 1 = -3.
	wantW := []float32{1, 10, 2, 20}
	wantB := []float32{1, -3}

	for i := range wantW {
		if weights[i] != wantW[i] {
			t.Errorf("weights = %v, want %v", weights, wantW)
		}
	}

	for i := range wantB {
		if bias[i] != wantB[i] {
			t.Errorf("bias = %v, want %v", bias, wantB)
		}
	}

	weights = []float32{1, 10, 2, 20}
	bias = []float32{0, 0}
	bn.Gamma = []float32{4, 0.5}

	if err := FoldBatchNorm(weights, bias, bn); err != nil {
		t.Fatal(err)
	}

	// scale = [4/2, 0.5/3]; bias = -mean*scale + beta.
	wantW = []float32{2, float32(10 * 0.5 / 3.0), 4, float32(20 * 0.5 / 3.0)}
	wantB = []float32{0, float32(-1*0.5/3.0 - 1)}

	for i := range wantW {
		if weights[i] != wantW[i] {
			t.Errorf("weights = %v, want %v", weights, wantW)
		}
	}

	for i := range wantB {
		if bias[i] != wantB[i] {
			t.Errorf("bias = %v, want %v", bias, wantB)
		}
	}
}

func TestFoldBatchNormErrors(t *testing.T) {
	t.Parallel()

	one, zero := []float32{1, 1}, []float32{0, 0}
	nan := float32(math.NaN())

	tests := []struct {
		name    string
		weights []float32
		bias    []float32
		bn      BatchNorm
	}{
		{"no channels", []float32{1, 2}, []float32{}, BatchNorm{}},
		{"short beta", []float32{1, 2}, zero, BatchNorm{one, []float32{0}, zero, one, 1e-3}},
		{"nil mean", []float32{1, 2}, zero, BatchNorm{one, zero, nil, one, 1e-3}},
		{"long variance", []float32{1, 2}, zero, BatchNorm{one, zero, zero, []float32{1, 1, 1}, 1e-3}},
		{"nil bias", []float32{1, 2}, nil, BatchNorm{one, zero, zero, one, 1e-3}},
		{"short bias", []float32{1, 2}, []float32{0}, BatchNorm{one, zero, zero, one, 1e-3}},
		{"no weights", nil, zero, BatchNorm{one, zero, zero, one, 1e-3}},
		{"weights not a multiple", []float32{1, 2, 3}, zero, BatchNorm{one, zero, zero, one, 1e-3}},
		{"zero variance and epsilon", []float32{1, 2}, zero, BatchNorm{one, zero, zero, []float32{1, 0}, 0}},
		{"negative variance", []float32{1, 2}, zero, BatchNorm{one, zero, zero, []float32{-1, 1}, 1e-3}},
		{"nan variance", []float32{1, 2}, zero, BatchNorm{one, zero, zero, []float32{1, nan}, 1e-3}},
		{"nan epsilon", []float32{1, 2}, zero, BatchNorm{one, zero, zero, one, nan}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			w := append([]float32(nil), tt.weights...)
			b := append([]float32(nil), tt.bias...)

			if err := FoldBatchNorm(w, b, tt.bn); err == nil {
				t.Fatal("no error")
			}

			for i := range w {
				if w[i] != tt.weights[i] {
					t.Errorf("weights modified on error: %v", w)
				}
			}

			for i := range b {
				if b[i] != tt.bias[i] {
					t.Errorf("bias modified on error: %v", b)
				}
			}
		})
	}
}

func TestOpsZeroAlloc(t *testing.T) {
	x := make([]float32, 1024)
	for i := range x {
		x[i] = float32(i) - 512
	}

	if n := testing.AllocsPerRun(100, func() { ReLU(x) }); n != 0 {
		t.Errorf("ReLU: %v allocs", n)
	}

	if n := testing.AllocsPerRun(100, func() { Sigmoid(x) }); n != 0 {
		t.Errorf("Sigmoid: %v allocs", n)
	}

	note := NewTensor(172, 88, 1)
	feat := NewTensor(172, 88, 32)
	dst := NewTensor(172, 88, 33)

	if n := testing.AllocsPerRun(100, func() {
		if err := ConcatChannelsRows(dst, 10, 20, note, feat); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("ConcatChannelsRows: %v allocs", n)
	}

	if n := testing.AllocsPerRun(100, func() {
		if err := ConcatChannels(dst, note, feat); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("ConcatChannels: %v allocs", n)
	}

	weights := make([]float32, 5*5*8*32)
	bias := make([]float32, 32)
	bn := BatchNorm{
		Gamma:    make([]float32, 32),
		Beta:     make([]float32, 32),
		Mean:     make([]float32, 32),
		Variance: make([]float32, 32),
		Epsilon:  1e-3,
	}

	if n := testing.AllocsPerRun(100, func() {
		if err := FoldBatchNorm(weights, bias, bn); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("FoldBatchNorm: %v allocs", n)
	}
}
