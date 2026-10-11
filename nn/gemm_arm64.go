//go:build arm64 && !purego

package nn

// haveSIMD reports whether gemm has native SIMD kernels on this target.
// NEON is mandatory on arm64, so there is no runtime detection.
const haveSIMD = true

// gemmNEON4x4 is the multi-channel NEON kernel (gemm_arm64.s): 4×4 tiles
// plus 1×4 tiles for the last m%4 rows. It needs m >= 1, k >= 4 and n a
// positive multiple of 4, and reads and writes exactly the elements gemm
// does.
//
//go:noescape
func gemmNEON4x4(out, a, wt, bias *float32, m, n, k int)

// gemmNEON8x1 is the single-channel (n = 1) NEON kernel: 8×1 tiles plus a
// 1×1 loop for the last m%8 rows. It needs m >= 1 and k >= 4.
//
//go:noescape
func gemmNEON8x1(out, a, wt, bias *float32, m, k int)

// gemmSIMD runs gemm with the NEON kernels and reports whether it did. It
// returns false, without touching out, for shapes no kernel covers: k < 4
// (the kernels reload the last four elements of a row for the k%4 tail),
// and n = len(bias) neither 1 nor a multiple of 4. The caller then runs the
// Go tiles.
//
// The kernels vectorise over k and reduce the partial sums at the end, so
// they round differently from the Go tiles, but each call is deterministic.
func gemmSIMD(out, a, wt, bias []float32, m, k int) bool {
	n := len(bias)
	if k < 4 || (n != 1 && n%4 != 0) {
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
		gemmNEON8x1(&out[0], &a[0], &wt[0], &bias[0], m, k)
	} else {
		gemmNEON4x4(&out[0], &a[0], &wt[0], &bias[0], m, n, k)
	}

	return true
}
