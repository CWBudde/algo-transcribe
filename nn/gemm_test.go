package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// gemmSizes are the depths and channel counts of the GEMM sweeps: every
// k%4 and k%8 remainder, every basic-pitch depth (49, 200, 297, 672, 936),
// n = 1..9 for every channel remainder and n = 32 as in basic-pitch.
var (
	gemmSweepK = []int{1, 2, 3, 5, 8, 49, 200, 297, 672, 936}
	gemmSweepN = []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 32}
)

// gemmFunc is the signature of gemm and its variants.
type gemmFunc func(out, a, wt, bias []float32, m, k int)

// TestGemmTiled checks both tile widths on every architecture against a
// float64 reference, over sizes that hit every row and channel remainder of
// the 4×tileN and 8×1 tiles.
func TestGemmTiled(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(3, 4))

	for _, nr := range []int{4, 2} {
		tiled := func(out, a, wt, bias []float32, m, k int) {
			gemmTiled(out, a, wt, bias, m, k, nr)
		}

		for _, k := range gemmSweepK {
			for m := 0; m <= 17; m++ {
				for _, n := range gemmSweepN {
					name := fmt.Sprintf("nr%d/k%d/m%d/n%d", nr, k, m, n)
					checkGemm(t, rng, name, tiled, m, n, k)
				}
			}
		}
	}
}

// checkGemm runs f on random data and compares every output with a float64
// reference, with a guard element after the output.
func checkGemm(t *testing.T, rng *rand.Rand, name string, f gemmFunc, m, n, k int) {
	t.Helper()

	a := gemmRandSlice(rng, m*k)
	wt := gemmRandSlice(rng, n*k)
	bias := gemmRandSlice(rng, n)

	out := make([]float32, m*n+1)
	out[m*n] = 42 // guard element after the matrix

	f(out[:m*n], a, wt, bias, m, k)

	if out[m*n] != 42 {
		t.Fatalf("%s: wrote past the output", name)
	}

	for i := range m {
		for j := range n {
			want, sumAbs := float64(bias[j]), math.Abs(float64(bias[j]))

			for p := range k {
				v := float64(a[i*k+p]) * float64(wt[j*k+p])
				want += v
				sumAbs += math.Abs(v)
			}

			if got := float64(out[i*n+j]); math.Abs(got-want) > 1e-5*sumAbs+1e-6 {
				t.Fatalf("%s: out[%d,%d] = %v, want %v", name, i, j, got, want)
			}
		}
	}
}

func gemmRandSlice(rng *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = rng.Float32()*2 - 1
	}

	return s
}

func TestTileN(t *testing.T) {
	t.Parallel()

	if tileN != 4 && tileN != 2 {
		t.Fatalf("tileN = %d, want 4 or 2", tileN)
	}
}
