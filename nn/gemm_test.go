package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// TestGemmTiled checks both tile widths on every architecture against a
// float64 reference, over sizes that hit every row and channel remainder of
// the 4×tileN and 8×1 tiles.
func TestGemmTiled(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(3, 4))

	for _, nr := range []int{4, 2} {
		for _, k := range []int{1, 2, 3, 5, 8, 49, 200, 936} {
			for m := 0; m <= 17; m++ {
				for n := 1; n <= 9; n++ {
					name := fmt.Sprintf("nr%d/k%d/m%d/n%d", nr, k, m, n)
					checkGemm(t, rng, name, nr, m, n, k)
				}
			}
		}
	}
}

func checkGemm(t *testing.T, rng *rand.Rand, name string, nr, m, n, k int) {
	t.Helper()

	a := gemmRandSlice(rng, m*k)
	wt := gemmRandSlice(rng, n*k)
	bias := gemmRandSlice(rng, n)

	out := make([]float32, m*n+1)
	out[m*n] = 42 // guard element after the matrix

	gemmTiled(out[:m*n], a, wt, bias, m, k, nr)

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
