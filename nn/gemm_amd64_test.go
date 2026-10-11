//go:build amd64 && !purego

package nn

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"testing"

	"golang.org/x/sys/cpu"
)

// avx2SweepK adds the depths below 4, which gemmSIMD declines but the
// kernels handle (the whole row is one masked step), to the common sweep.
var avx2SweepK = append([]int{1, 2, 3}, simdSweepK...)

// skipWithoutAVX2 skips a test that calls the AVX2 kernels directly on a CPU
// that would fault on them.
func skipWithoutAVX2(t *testing.T) {
	t.Helper()

	if !haveSIMD {
		t.Skip("the CPU lacks AVX2 or FMA")
	}
}

// TestGemmAVX2N4Direct calls the multi-channel assembly entry point
// directly, without gemmSIMD, over every row remainder of its 2×4 and 1×4
// tiles, and compares it with the float64 reference and the Go tiles.
func TestGemmAVX2N4Direct(t *testing.T) {
	t.Parallel()
	skipWithoutAVX2(t)

	rng := rand.New(rand.NewPCG(15, 16))

	f := func(out, a, wt, bias []float32, m, k int) {
		gemmAVX2N4(&out[0], &a[0], &wt[0], &bias[0], m, len(bias), k)
	}

	for _, k := range avx2SweepK {
		for m := 1; m <= 17; m++ {
			for _, n := range []int{4, 8, 12, 32} {
				name := fmt.Sprintf("k%d/m%d/n%d", k, m, n)
				checkGemm(t, rng, name, f, m, n, k)
				checkKernelMatchesGo(t, rng, name, f, m, n, k)
			}
		}
	}
}

// TestGemmAVX2N1Direct calls the single-channel assembly entry point
// directly over every row remainder of its 8×1 tiles and 1×1 loop.
func TestGemmAVX2N1Direct(t *testing.T) {
	t.Parallel()
	skipWithoutAVX2(t)

	rng := rand.New(rand.NewPCG(17, 18))

	f := func(out, a, wt, bias []float32, m, k int) {
		gemmAVX2N1(&out[0], &a[0], &wt[0], &bias[0], m, k)
	}

	for _, k := range avx2SweepK {
		for m := 1; m <= 25; m++ {
			name := fmt.Sprintf("k%d/m%d", k, m)
			checkGemm(t, rng, name, f, m, 1, k)
			checkKernelMatchesGo(t, rng, name, f, m, 1, k)
		}
	}
}

// TestGemmAVX2MaskedLanes checks that the masked k%8 tail ignores the lanes
// past the end of a row. The tail of every row but the last overlaps the
// next row, and the tail of the last row the memory after the slice, which
// is NaN here; every output must still be finite and equal the Go tiles'.
func TestGemmAVX2MaskedLanes(t *testing.T) {
	t.Parallel()
	skipWithoutAVX2(t)

	rng := rand.New(rand.NewPCG(19, 20))
	nan := float32(math.NaN())

	for _, n := range []int{1, 4} {
		for _, k := range []int{1, 5, 7, 9, 13, 49} {
			for _, m := range []int{1, 2, 3, 8, 9} {
				aBuf := make([]float32, m*k+8)
				wBuf := make([]float32, n*k+8)

				for i := range aBuf {
					aBuf[i] = nan
				}

				for i := range wBuf {
					wBuf[i] = nan
				}

				a := aBuf[:m*k]
				wt := wBuf[:n*k]
				copy(a, gemmRandSlice(rng, m*k))
				copy(wt, gemmRandSlice(rng, n*k))

				bias := gemmRandSlice(rng, n)

				got := make([]float32, m*n)
				want := make([]float32, m*n)

				if n == 1 {
					gemmAVX2N1(&got[0], &a[0], &wt[0], &bias[0], m, k)
				} else {
					gemmAVX2N4(&got[0], &a[0], &wt[0], &bias[0], m, n, k)
				}

				gemmTiled(want, a, wt, bias, m, k, 4)

				for i, v := range got {
					if math.IsNaN(float64(v)) || math.Abs(float64(v-want[i])) > 1e-4 {
						t.Fatalf("n%d/k%d/m%d: out[%d] = %v, want %v", n, k, m, i, v, want[i])
					}
				}
			}
		}
	}
}

// TestGemmAVX2Selected checks that gemm runs the AVX2 kernels for every
// basic-pitch layer shape whenever the CPU reports AVX2 and FMA. It fails,
// rather than skips, in that case, so the ubuntu CI job (GitHub Actions sets
// CI=true, and its x86 runners have both features) provably exercises the
// kernels instead of silently testing the Go tiles twice. Only a CPU without
// the features skips it.
func TestGemmAVX2Selected(t *testing.T) {
	t.Parallel()

	ci := os.Getenv("CI")

	if !cpu.X86.HasAVX2 || !cpu.X86.HasFMA {
		t.Skipf("CI=%q: the CPU lacks AVX2 (%v) or FMA (%v), so gemm runs the Go tiles",
			ci, cpu.X86.HasAVX2, cpu.X86.HasFMA)
	}

	if !haveSIMD {
		t.Fatalf("CI=%q: the CPU reports AVX2 and FMA, but haveSIMD is false", ci)
	}

	for i, s := range bpGemmShapes() {
		m, n, k := s[0], s[1], s[2]
		out := make([]float32, m*n)

		if !gemmSIMD(out, make([]float32, m*k), make([]float32, n*k), make([]float32, n), m, k) {
			t.Errorf("CI=%q: %s: the AVX2 kernel was not selected for m=%d n=%d k=%d",
				ci, bpLayers[i].name, m, n, k)
		}
	}
}
