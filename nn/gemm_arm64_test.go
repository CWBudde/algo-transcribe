//go:build arm64 && !purego

package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
)

// neonSweepK covers every k%4 and k%8 tail the kernels handle (k >= 4) and
// every basic-pitch depth.
var neonSweepK = []int{4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 49, 200, 297, 672, 936}

// TestGemmNEON4x4Direct calls the multi-channel assembly entry point
// directly, without gemmSIMD, over every row remainder of its 4×4 and 1×4
// tiles, and compares it with the float64 reference and the Go tiles.
func TestGemmNEON4x4Direct(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(7, 8))

	f := func(out, a, wt, bias []float32, m, k int) {
		gemmNEON4x4(&out[0], &a[0], &wt[0], &bias[0], m, len(bias), k)
	}

	for _, k := range neonSweepK {
		for m := 1; m <= 17; m++ {
			for _, n := range []int{4, 8, 12, 32} {
				name := fmt.Sprintf("k%d/m%d/n%d", k, m, n)
				checkGemm(t, rng, name, f, m, n, k)
				checkNEONMatchesGo(t, rng, name, f, m, n, k)
			}
		}
	}
}

// TestGemmNEON8x1Direct calls the single-channel assembly entry point
// directly over every row remainder of its 8×1 tiles and 1×1 loop.
func TestGemmNEON8x1Direct(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(9, 10))

	f := func(out, a, wt, bias []float32, m, k int) {
		gemmNEON8x1(&out[0], &a[0], &wt[0], &bias[0], m, k)
	}

	for _, k := range neonSweepK {
		for m := 1; m <= 25; m++ {
			name := fmt.Sprintf("k%d/m%d", k, m)
			checkGemm(t, rng, name, f, m, 1, k)
			checkNEONMatchesGo(t, rng, name, f, m, 1, k)
		}
	}
}

// checkNEONMatchesGo runs a raw kernel and the 4-wide Go tiles on the same
// data. They sum in different orders, so they agree to within twice the
// reference tolerance of checkGemm, not bit for bit.
func checkNEONMatchesGo(t *testing.T, rng *rand.Rand, name string, f gemmFunc, m, n, k int) {
	t.Helper()

	a := gemmRandSlice(rng, m*k)
	wt := gemmRandSlice(rng, n*k)
	bias := gemmRandSlice(rng, n)

	got := make([]float32, m*n)
	want := make([]float32, m*n)

	f(got, a, wt, bias, m, k)
	gemmTiled(want, a, wt, bias, m, k, 4)

	for i := range m {
		for j := range n {
			sumAbs := math.Abs(float64(bias[j]))
			for p := range k {
				sumAbs += math.Abs(float64(a[i*k+p]) * float64(wt[j*k+p]))
			}

			if d := math.Abs(float64(got[i*n+j]) - float64(want[i*n+j])); d > 2*(1e-5*sumAbs+1e-6) {
				t.Fatalf("%s: NEON out[%d,%d] = %v, Go tiles %v", name, i, j, got[i*n+j], want[i*n+j])
			}
		}
	}
}
