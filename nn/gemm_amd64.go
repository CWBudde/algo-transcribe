//go:build amd64 && !purego

package nn

import "golang.org/x/sys/cpu"

// haveSIMD reports whether gemm has native SIMD kernels on this machine. The
// AVX2 kernels also need FMA (VFMADD231PS), so both features are checked,
// once, at package initialisation; without them gemm runs the Go tiles.
var haveSIMD = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// gemmAVX2N4 is the multi-channel AVX2 kernel (gemm_amd64.s): 2×4 tiles
// plus 1×4 tiles for the last m%2 rows. It needs m >= 1, k >= 1 and n a
// positive multiple of 4, and reads and writes exactly the elements gemm
// does.
//
//go:noescape
func gemmAVX2N4(out, a, wt, bias *float32, m, n, k int)

// gemmAVX2N1 is the single-channel (n = 1) AVX2 kernel: 8×1 tiles plus a
// 1×1 loop for the last m%8 rows. It needs m >= 1 and k >= 1.
//
//go:noescape
func gemmAVX2N1(out, a, wt, bias *float32, m, k int)

// gemmSIMD runs gemm with the AVX2 kernels and reports whether it did. It
// returns false, without touching out, on CPUs without AVX2 and FMA and for
// shapes no kernel covers: n = len(bias) neither 1 nor a multiple of 4. It
// also declines k < 4, which the kernels would handle, so that both
// architectures take exactly the same shapes. The caller then runs the Go
// tiles.
//
// The kernels vectorise over k and reduce the partial sums at the end, so
// they round differently from the Go tiles, but each call is deterministic.
func gemmSIMD(out, a, wt, bias []float32, m, k int) bool {
	n := len(bias)
	if !haveSIMD || k < 4 || (n != 1 && n%4 != 0) {
		return false
	}

	// Bound the slices by their lengths before handing raw pointers to the
	// assembly, which reads and writes exactly the first m*k, n*k and m*n
	// elements: s[0:x:len(s)] panics when x > len(s), whereas s[:x] would
	// only check the capacity.
	out = out[0 : m*n : len(out)]
	a = a[0 : m*k : len(a)]
	wt = wt[0 : n*k : len(wt)]

	if m == 0 || n == 0 {
		return true
	}

	if n == 1 {
		gemmAVX2N1(&out[0], &a[0], &wt[0], &bias[0], m, k)
	} else {
		gemmAVX2N4(&out[0], &a[0], &wt[0], &bias[0], m, n, k)
	}

	return true
}
