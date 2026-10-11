//go:build arm64 && !purego

package nn

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// TestGemmNEON4x4Direct calls the multi-channel assembly entry point
// directly, without gemmSIMD, over every row remainder of its 4×4 and 1×4
// tiles, and compares it with the float64 reference and the Go tiles.
func TestGemmNEON4x4Direct(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewPCG(7, 8))

	f := func(out, a, wt, bias []float32, m, k int) {
		gemmNEON4x4(&out[0], &a[0], &wt[0], &bias[0], m, len(bias), k)
	}

	for _, k := range simdSweepK {
		for m := 1; m <= 17; m++ {
			for _, n := range []int{4, 8, 12, 32} {
				name := fmt.Sprintf("k%d/m%d/n%d", k, m, n)
				checkGemm(t, rng, name, f, m, n, k)
				checkKernelMatchesGo(t, rng, name, f, m, n, k)
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

	for _, k := range simdSweepK {
		for m := 1; m <= 25; m++ {
			name := fmt.Sprintf("k%d/m%d", k, m)
			checkGemm(t, rng, name, f, m, 1, k)
			checkKernelMatchesGo(t, rng, name, f, m, 1, k)
		}
	}
}
