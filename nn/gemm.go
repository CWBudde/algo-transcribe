package nn

// blockM is the number of output positions unrolled per im2col block. It is
// the row count of the single-channel GEMM kernel and twice that of the
// multi-channel kernels.
const blockM = 8

// gemm computes out[i*n+j] = bias[j] + dot(a[i*k:(i+1)*k], wt[j*k:(j+1)*k])
// for i < m and j < n, where n = len(bias). a holds the im2col rows and wt
// the weights transposed to one contiguous k-vector per output channel.
//
// The work is register-blocked into tiles of independent accumulators, which
// hides the floating-point latency and reuses every loaded operand across a
// tile. The tile width is tileN (see gemm_tile*.go): output channels are
// covered by 4×tileN tiles, and the remaining n%tileN channels by 8×1 tiles.
// The kernels are plain Go written so that the compiler removes the bounds
// checks from the inner loops, and they run on every target, js/wasm
// included.
//
// On arm64 (without the purego tag) gemm first tries the NEON kernels of
// gemmSIMD, which cover n = 1 and n a multiple of 4 for k >= 4. On amd64 it
// tries AVX2 kernels with the same coverage, provided the CPU has AVX2 and
// FMA, which is detected at run time. Other shapes and CPUs fall back to the
// Go tiles.
func gemm(out, a, wt, bias []float32, m, k int) {
	if gemmSIMD(out, a, wt, bias, m, k) {
		return
	}

	gemmTiled(out, a, wt, bias, m, k, tileN)
}

// gemmTiled is gemm with the tile width nr (4 or 2) as a parameter, so that
// tests can exercise every kernel on every architecture.
func gemmTiled(out, a, wt, bias []float32, m, k, nr int) {
	n := len(bias)
	nWide := n - n%nr

	if nWide > 0 {
		i := 0
		for ; i+4 <= m; i += 4 {
			a0 := a[i*k : (i+1)*k]
			a1 := a[(i+1)*k : (i+2)*k]
			a2 := a[(i+2)*k : (i+3)*k]
			a3 := a[(i+3)*k : (i+4)*k]

			for j := 0; j < nWide; j += nr {
				c := out[i*n+j:]
				if nr == 4 {
					kernel4x4(c, n, a0, a1, a2, a3,
						wt[j*k:(j+1)*k], wt[(j+1)*k:(j+2)*k], wt[(j+2)*k:(j+3)*k], wt[(j+3)*k:(j+4)*k],
						bias[j:j+4])
				} else {
					kernel4x2(c, n, a0, a1, a2, a3, wt[j*k:(j+1)*k], wt[(j+1)*k:(j+2)*k], bias[j:j+2])
				}
			}
		}

		for ; i < m; i++ {
			ar := a[i*k : (i+1)*k]
			for j := range nWide {
				out[i*n+j] = bias[j] + dot(ar, wt[j*k:(j+1)*k])
			}
		}
	}

	for j := nWide; j < n; j++ {
		w := wt[j*k : (j+1)*k]

		i := 0
		for ; i+8 <= m; i += 8 {
			kernel8x1(out[i*n+j:], n, a[i*k:(i+8)*k], k, w, bias[j])
		}

		for ; i < m; i++ {
			out[i*n+j] = bias[j] + dot(a[i*k:(i+1)*k], w)
		}
	}
}

// kernel4x4 writes the tile c[r*ldc+j] = bias[j] + dot(a_r, b_j) for
// r, j < 4. Its 16 accumulators fit the 32 floating-point registers of
// arm64.
func kernel4x4(c []float32, ldc int, a0, a1, a2, a3, b0, b1, b2, b3, bias []float32) {
	a1 = a1[:len(a0)]
	a2 = a2[:len(a0)]
	a3 = a3[:len(a0)]
	b0 = b0[:len(a0)]
	b1 = b1[:len(a0)]
	b2 = b2[:len(a0)]
	b3 = b3[:len(a0)]

	var (
		c00, c01, c02, c03 float32
		c10, c11, c12, c13 float32
		c20, c21, c22, c23 float32
		c30, c31, c32, c33 float32
	)

	for p, x0 := range a0 {
		x1, x2, x3 := a1[p], a2[p], a3[p]
		y0, y1, y2, y3 := b0[p], b1[p], b2[p], b3[p]

		c00 += x0 * y0
		c01 += x0 * y1
		c02 += x0 * y2
		c03 += x0 * y3
		c10 += x1 * y0
		c11 += x1 * y1
		c12 += x1 * y2
		c13 += x1 * y3
		c20 += x2 * y0
		c21 += x2 * y1
		c22 += x2 * y2
		c23 += x2 * y3
		c30 += x3 * y0
		c31 += x3 * y1
		c32 += x3 * y2
		c33 += x3 * y3
	}

	bias = bias[:4]
	r := c[:4]
	r[0], r[1], r[2], r[3] = bias[0]+c00, bias[1]+c01, bias[2]+c02, bias[3]+c03
	r = c[ldc : ldc+4]
	r[0], r[1], r[2], r[3] = bias[0]+c10, bias[1]+c11, bias[2]+c12, bias[3]+c13
	r = c[2*ldc : 2*ldc+4]
	r[0], r[1], r[2], r[3] = bias[0]+c20, bias[1]+c21, bias[2]+c22, bias[3]+c23
	r = c[3*ldc : 3*ldc+4]
	r[0], r[1], r[2], r[3] = bias[0]+c30, bias[1]+c31, bias[2]+c32, bias[3]+c33
}

// kernel4x2 writes the tile c[r*ldc+j] = bias[j] + dot(a_r, b_j) for r < 4
// and j < 2. Its 8 accumulators plus operands fit the 16 float32 registers
// of Go's wasm backend, where the 4×4 tile spills.
func kernel4x2(c []float32, ldc int, a0, a1, a2, a3, b0, b1, bias []float32) {
	a1 = a1[:len(a0)]
	a2 = a2[:len(a0)]
	a3 = a3[:len(a0)]
	b0 = b0[:len(a0)]
	b1 = b1[:len(a0)]

	var c00, c01, c10, c11, c20, c21, c30, c31 float32

	for p, x0 := range a0 {
		x1, x2, x3 := a1[p], a2[p], a3[p]
		y0, y1 := b0[p], b1[p]

		c00 += x0 * y0
		c01 += x0 * y1
		c10 += x1 * y0
		c11 += x1 * y1
		c20 += x2 * y0
		c21 += x2 * y1
		c30 += x3 * y0
		c31 += x3 * y1
	}

	bias = bias[:2]
	r := c[:2]
	r[0], r[1] = bias[0]+c00, bias[1]+c01
	r = c[ldc : ldc+2]
	r[0], r[1] = bias[0]+c10, bias[1]+c11
	r = c[2*ldc : 2*ldc+2]
	r[0], r[1] = bias[0]+c20, bias[1]+c21
	r = c[3*ldc : 3*ldc+2]
	r[0], r[1] = bias[0]+c30, bias[1]+c31
}

// kernel8x1 writes c[r*ldc] = bias + dot(a[r*k:(r+1)*k], b) for r < 8, the
// tile for a single output channel: eight im2col rows against one weight
// vector.
func kernel8x1(c []float32, ldc int, a []float32, k int, b []float32, bias float32) {
	a0 := a[:k]
	a1 := a[k : 2*k]
	a2 := a[2*k : 3*k]
	a3 := a[3*k : 4*k]
	a4 := a[4*k : 5*k]
	a5 := a[5*k : 6*k]
	a6 := a[6*k : 7*k]
	a7 := a[7*k : 8*k]
	b = b[:len(a0)]

	var s0, s1, s2, s3, s4, s5, s6, s7 float32

	for p, y := range b {
		s0 += a0[p] * y
		s1 += a1[p] * y
		s2 += a2[p] * y
		s3 += a3[p] * y
		s4 += a4[p] * y
		s5 += a5[p] * y
		s6 += a6[p] * y
		s7 += a7[p] * y
	}

	c[0] = bias + s0
	c[ldc] = bias + s1
	c[2*ldc] = bias + s2
	c[3*ldc] = bias + s3
	c[4*ldc] = bias + s4
	c[5*ldc] = bias + s5
	c[6*ldc] = bias + s6
	c[7*ldc] = bias + s7
}

// dot returns the dot product of a and b[:len(a)] with four accumulators.
// It handles the tail rows of a block.
func dot(a, b []float32) float32 {
	b = b[:len(a)]

	var s0, s1, s2, s3 float32

	p := 3
	for ; p < len(a); p += 4 {
		s0 += a[p-3] * b[p-3]
		s1 += a[p-2] * b[p-2]
		s2 += a[p-1] * b[p-1]
		s3 += a[p] * b[p]
	}

	for p -= 3; p < len(a); p++ {
		s0 += a[p] * b[p]
	}

	return (s0 + s1) + (s2 + s3)
}
