package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// simdCovers reports whether gemmSIMD is expected to take a shape.
func simdCovers(n, k int) bool {
	return haveSIMD && k >= 4 && (n == 1 || n%4 == 0)
}

// TestGemm checks the dispatching gemm, the SIMD kernels where they apply
// and the Go tiles elsewhere, against the float64 reference over the full
// sweep, and that gemmSIMD takes exactly the shapes it should.
func TestGemm(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(5, 6))

	for _, k := range gemmSweepK {
		for m := 0; m <= 17; m++ {
			for _, n := range gemmSweepN {
				name := fmt.Sprintf("k%d/m%d/n%d", k, m, n)
				checkGemm(t, rng, name, gemm, m, n, k)

				out := make([]float32, m*n)
				for i := range out {
					out[i] = 42
				}

				taken := gemmSIMD(out, make([]float32, m*k), make([]float32, n*k), make([]float32, n), m, k)
				if want := simdCovers(n, k); taken != want {
					t.Fatalf("%s: gemmSIMD = %v, want %v", name, taken, want)
				}

				if !taken {
					for i, v := range out {
						if v != 42 {
							t.Fatalf("%s: gemmSIMD returned false but wrote out[%d]", name, i)
						}
					}
				}
			}
		}
	}
}

// TestGemmSIMDTails checks the SIMD kernels against the float64 reference
// at the depths TestGemmTiled's sweep lacks: together they give every k%8
// remainder, so every k%4 tail mask and the single 4-element step of the 8×1
// kernel run.
func TestGemmSIMDTails(t *testing.T) {
	t.Parallel()

	if !haveSIMD {
		t.Skip("no SIMD gemm on this target")
	}

	rng := rand.New(rand.NewPCG(7, 8))

	simd := func(out, a, wt, bias []float32, m, k int) {
		if !gemmSIMD(out, a, wt, bias, m, k) {
			t.Fatalf("gemmSIMD declined m=%d n=%d k=%d", m, len(bias), k)
		}
	}

	for _, k := range []int{4, 6, 7, 10, 11, 14, 15} {
		for m := 0; m <= 17; m++ {
			for _, n := range gemmSweepN {
				if !simdCovers(n, k) {
					continue
				}

				checkGemm(t, rng, fmt.Sprintf("k%d/m%d/n%d", k, m, n), simd, m, n, k)
			}
		}
	}
}

// TestGemmSIMDMatchesGo compares the SIMD kernels directly with the Go
// tiles of this target on random data.
func TestGemmSIMDMatchesGo(t *testing.T) {
	t.Parallel()

	if !haveSIMD {
		t.Skip("no SIMD gemm on this target")
	}

	rng := rand.New(rand.NewPCG(9, 10))

	shapes := append(bpGemmShapes(), [][3]int{
		{17, 4, 4}, {5, 12, 7}, {9, 1, 13}, {3, 32, 297}, {16, 8, 936}, {1, 1, 5},
	}...)

	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]
		a := gemmRandSlice(rng, m*k)
		wt := gemmRandSlice(rng, n*k)
		bias := gemmRandSlice(rng, n)

		got := make([]float32, m*n)
		want := make([]float32, m*n)

		if !gemmSIMD(got, a, wt, bias, m, k) {
			t.Fatalf("m%d/n%d/k%d: gemmSIMD declined", m, n, k)
		}

		gemmTiled(want, a, wt, bias, m, k, tileN)

		for i := range m {
			for j := range n {
				sumAbs := math.Abs(float64(bias[j]))
				for p := range k {
					sumAbs += math.Abs(float64(a[i*k+p]) * float64(wt[j*k+p]))
				}

				g, w := float64(got[i*n+j]), float64(want[i*n+j])
				if math.Abs(g-w) > 1e-5*sumAbs+1e-6 {
					t.Fatalf("m%d/n%d/k%d: out[%d,%d] = %v (SIMD), %v (Go)", m, n, k, i, j, g, w)
				}
			}
		}
	}
}

// bpGemmShapes returns (m, n, k) of the gemm calls of the six basic-pitch
// convolutions (bpLayers): m = blockM, n = cout, k = kh*kw*cin.
func bpGemmShapes() [][3]int {
	shapes := make([][3]int, 0, len(bpLayers))
	for _, l := range bpLayers {
		shapes = append(shapes, [3]int{blockM, l.cout, l.kh * l.kw * l.in.C})
	}

	return shapes
}

// TestGemmSIMDBasicPitchShapes checks that the SIMD path is taken for every
// basic-pitch layer on targets that have it, and never elsewhere.
func TestGemmSIMDBasicPitchShapes(t *testing.T) {
	t.Parallel()

	want := map[string][3]int{
		"conv2d_1":         {8, 8, 936},
		"contours-reduced": {8, 1, 200},
		"conv2d_2":         {8, 32, 49},
		"conv2d_3":         {8, 1, 672},
		"conv2d_4":         {8, 32, 200},
		"conv2d_5":         {8, 1, 297},
	}

	for i, s := range bpGemmShapes() {
		name := bpLayers[i].name
		if s != want[name] {
			t.Fatalf("%s: shape %v, want %v", name, s, want[name])
		}

		m, n, k := s[0], s[1], s[2]
		out := make([]float32, m*n)

		if got := gemmSIMD(out, make([]float32, m*k), make([]float32, n*k), make([]float32, n), m, k); got != haveSIMD {
			t.Errorf("%s: gemmSIMD = %v, want %v", name, got, haveSIMD)
		}
	}
}

// TestGemmDeterministic checks that two gemm calls on the same inputs give
// bit-identical results, for every basic-pitch shape and some remainders.
func TestGemmDeterministic(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(11, 12))

	shapes := append(bpGemmShapes(), [3]int{7, 4, 5}, [3]int{13, 1, 299}, [3]int{3, 12, 50})
	for _, s := range shapes {
		m, n, k := s[0], s[1], s[2]
		a := gemmRandSlice(rng, m*k)
		wt := gemmRandSlice(rng, n*k)
		bias := gemmRandSlice(rng, n)

		first := make([]float32, m*n)
		second := make([]float32, m*n)

		gemm(first, a, wt, bias, m, k)
		gemm(second, a, wt, bias, m, k)

		for i := range first {
			if math.Float32bits(first[i]) != math.Float32bits(second[i]) {
				t.Fatalf("m%d/n%d/k%d: out[%d] = %v, then %v", m, n, k, i, first[i], second[i])
			}
		}
	}
}

// TestGemmTailInf checks that the masked k%4 tail of the SIMD kernels does
// not turn an infinity in the overlapping reload into NaN (0*Inf): the
// result must be +Inf as with the Go tiles.
func TestGemmTailInf(t *testing.T) {
	t.Parallel()

	for _, n := range []int{1, 4} {
		for _, k := range []int{5, 6, 7, 13} {
			for _, m := range []int{1, 4, 8, 9} {
				a := make([]float32, m*k)
				wt := make([]float32, n*k)
				bias := make([]float32, n)

				for i := range a {
					a[i] = 1
				}

				for i := range wt {
					wt[i] = 1
				}

				// Element k-4 is summed in the last full step and reloaded,
				// masked, by the tail.
				for i := range m {
					a[i*k+k-4] = float32(math.Inf(1))
				}

				for _, f := range []gemmFunc{gemm, func(out, a, wt, bias []float32, m, k int) {
					gemmTiled(out, a, wt, bias, m, k, tileN)
				}} {
					out := make([]float32, m*n)
					f(out, a, wt, bias, m, k)

					for i, v := range out {
						if !math.IsInf(float64(v), 1) {
							t.Fatalf("n%d/k%d/m%d: out[%d] = %v, want +Inf", n, k, m, i, v)
						}
					}
				}
			}
		}
	}
}

// TestGemmSIMDShortSlices checks that gemmSIMD panics, rather than letting
// the assembly read or write past the end, when a slice is shorter than the
// shape needs, even if its capacity would suffice.
func TestGemmSIMDShortSlices(t *testing.T) {
	t.Parallel()

	if !haveSIMD {
		t.Skip("no SIMD gemm on this target")
	}

	const m, n, k = 8, 4, 9

	cases := map[string]func() (out, a, wt []float32){
		"out": func() (out, a, wt []float32) {
			return make([]float32, m*n-1, m*n+16), make([]float32, m*k), make([]float32, n*k)
		},
		"a": func() (out, a, wt []float32) {
			return make([]float32, m*n), make([]float32, m*k-1, m*k+16), make([]float32, n*k)
		},
		"wt": func() (out, a, wt []float32) {
			return make([]float32, m*n), make([]float32, m*k), make([]float32, n*k-1, n*k+16)
		},
	}

	for name, mk := range cases {
		out, a, wt := mk()

		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: short slice did not panic", name)
				}
			}()

			gemmSIMD(out, a, wt, make([]float32, n), m, k)
		}()
	}
}

// BenchmarkGemm measures one gemm call, the unit of work of a Conv2D block,
// for every basic-pitch layer shape, without the im2col around it.
func BenchmarkGemm(b *testing.B) {
	rng := rand.New(rand.NewPCG(13, 14))

	for i, s := range bpGemmShapes() {
		m, n, k := s[0], s[1], s[2]
		a := gemmRandSlice(rng, m*k)
		wt := gemmRandSlice(rng, n*k)
		bias := gemmRandSlice(rng, n)
		out := make([]float32, m*n)

		b.Run(fmt.Sprintf("%s/m%d_n%d_k%d", bpLayers[i].name, m, n, k), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				gemm(out, a, wt, bias, m, k)
			}

			b.ReportMetric(float64(m*n*k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
		})
	}
}
