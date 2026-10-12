//go:build amd64 && !purego

// AVX2/FMA GEMM kernels for nn, dot form over the transposed weights.
//
// Both kernels compute out[i*n+j] = bias[j] + dot(a[i*k:(i+1)*k],
// wt[j*k:(j+1)*k]) with a, wt and out in the layout of gemm (gemm.go), the
// same operands as the NEON kernels of gemm_arm64.s. They vectorise over k:
// every accumulator is a ymm vector of partial sums over eight consecutive k
// positions, and a VHADDPS tree reduces the accumulators to the scalar dot
// products after the k loop.
//
// The k%8 tail is a single masked step: VMASKMOVPS loads the last k%8
// elements of each row with the remaining lanes zeroed. Both operands are
// masked, so the zeroed lanes contribute 0*0 = +0 even when the data holds
// infinities, and masked lanes never touch memory, so the step reads nothing
// past the end of a slice. The kernels therefore work for any k >= 1.
//
// Every vector instruction is VEX-encoded (no legacy SSE, which would cost a
// state transition per mix), and VZEROUPPER runs before every RET. The
// caller (gemmSIMD) runs these kernels only when the CPU has AVX2 and FMA.
//
// Plan 9 operand order: sources first, destination last.
//   VFMADD231PS Y8, Y12, Y0  Y0 += Y12 * Y8 (the first operand may be memory)
//   VMASKMOVPS  (R), Y15, Y8 Y8 = memory at R in the lanes whose Y15 sign bit
//                            is set, 0 elsewhere
//   VHADDPS     Y1, Y0, Y8   Y8 = [Y0[0]+Y0[1], Y0[2]+Y0[3], Y1[0]+Y1[1],
//                            Y1[2]+Y1[3]] in each 128-bit half
//
// BP (frame pointer) and R15 (clobbered by global accesses when dynamic
// linking) are left alone; R14 (g under ABIInternal) is restored by the ABI
// wrapper after an ABI0 function returns.

#include "textflag.h"

// gemmTailMask holds eight all-ones lanes followed by eight zero lanes. The
// 32 bytes at offset 32-4*tail have exactly the first tail lanes set, the
// mask of a k%8 tail of tail elements.
DATA gemmTailMask<>+0x00(SB)/8, $0xffffffffffffffff
DATA gemmTailMask<>+0x08(SB)/8, $0xffffffffffffffff
DATA gemmTailMask<>+0x10(SB)/8, $0xffffffffffffffff
DATA gemmTailMask<>+0x18(SB)/8, $0xffffffffffffffff
DATA gemmTailMask<>+0x20(SB)/8, $0x0000000000000000
DATA gemmTailMask<>+0x28(SB)/8, $0x0000000000000000
DATA gemmTailMask<>+0x30(SB)/8, $0x0000000000000000
DATA gemmTailMask<>+0x38(SB)/8, $0x0000000000000000
GLOBL gemmTailMask<>(SB), RODATA|NOPTR, $64

// FMA2X4 accumulates the a vectors Y12, Y13 (rows) against the weight
// vectors Y8..Y11 (channels) into Y0..Y7, accumulator Y(4*row+channel).
// Each lane holds the partial sum of one k position.
#define FMA2X4 \
	VFMADD231PS Y8, Y12, Y0;  \
	VFMADD231PS Y9, Y12, Y1;  \
	VFMADD231PS Y10, Y12, Y2; \
	VFMADD231PS Y11, Y12, Y3; \
	VFMADD231PS Y8, Y13, Y4;  \
	VFMADD231PS Y9, Y13, Y5;  \
	VFMADD231PS Y10, Y13, Y6; \
	VFMADD231PS Y11, Y13, Y7

// REDUCE4 reduces the four accumulators c0..c3 of one row (one per output
// channel) to [sum(c0), sum(c1), sum(c2), sum(c3)], adds the four biases at
// (CX)(R9*1) and stores the four outputs at dst. It uses Y8 and Y9:
//
//	Y8 = [c0 0+1, c0 2+3, c1 0+1, c1 2+3 | the same for lanes 4..7]
//	Y9 = the same for c2, c3
//	Y8 = [c0 0..3, c1 0..3, c2 0..3, c3 0..3 | c0 4..7, c1 4..7, c2 4..7, c3 4..7]
//	X8 = low half + high half + bias
#define REDUCE4(c0, c1, c2, c3, dst) \
	VHADDPS      c1, c0, Y8;         \
	VHADDPS      c3, c2, Y9;         \
	VHADDPS      Y9, Y8, Y8;         \
	VEXTRACTF128 $1, Y8, X9;         \
	VADDPS       X9, X8, X8;         \
	VADDPS       (CX)(R9*1), X8, X8; \
	VMOVUPS      X8, dst

// func gemmAVX2N4(out, a, wt, bias *float32, m, n, k int)
//
// n must be a positive multiple of 4, m >= 1 and k >= 1. Rows are covered
// by 2×4 tiles (2 rows × 4 channels: 8 accumulators, 4 weight and 2 a
// vectors, 14 of the 16 ymm registers plus the tail mask in Y15), the last
// m%2 rows by 1×4 tiles.
TEXT ·gemmAVX2N4(SB), NOSPLIT, $0-56
	MOVQ out+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ wt+16(FP), DX
	MOVQ bias+24(FP), CX
	MOVQ m+32(FP), R8
	MOVQ n+40(FP), R11
	MOVQ k+48(FP), R10

	SHLQ $2, R10           // R10 = k*4, byte stride of the rows of a and wt
	SHLQ $2, R11           // R11 = n*4, byte stride of the rows of out
	LEAQ (R10)(R10*2), R14 // R14 = 3*k*4, offset of the fourth wt row of a block

	MOVQ    R10, AX
	ANDQ    $31, AX                   // AX = 4*(k%8), bytes in the tail
	JZ      n4_main
	LEAQ    gemmTailMask<>+32(SB), BX
	SUBQ    AX, BX
	VMOVUPS (BX), Y15                 // Y15 = first k%8 lanes set, constant for the call

n4_main:
	SHRQ $1, R8 // R8 = m/2 row pairs
	JZ   n4_rows1

n4_rowpair:
	XORQ R9, R9  // R9 = byte offset of the channel block in an out row and in bias
	MOVQ DX, R13 // R13 = wt row of the first channel of the block

n4_tile:
	MOVQ   SI, BX     // BX = a row 0, row 1 at +R10
	MOVQ   R13, R12   // R12 = wt row 0, rows 1, 2, 3 at +R10, +2*R10, +R14
	VXORPS Y0, Y0, Y0 // zero the 8 accumulators
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	MOVQ R10, AX
	SHRQ $5, AX  // AX = k/8 full steps
	JZ   n4_tail

n4_kloop:
	VMOVUPS (R12), Y8         // eight k positions of each wt row
	VMOVUPS (R12)(R10*1), Y9
	VMOVUPS (R12)(R10*2), Y10
	VMOVUPS (R12)(R14*1), Y11
	VMOVUPS (BX), Y12         // and of each a row
	VMOVUPS (BX)(R10*1), Y13
	FMA2X4
	ADDQ    $32, BX
	ADDQ    $32, R12
	DECQ    AX
	JNZ     n4_kloop

n4_tail:
	TESTQ $31, R10 // k%8 == 0: no tail
	JZ    n4_reduce

	VMASKMOVPS (R12), Y15, Y8 // the last k%8 elements, the other lanes zero
	VMASKMOVPS (R12)(R10*1), Y15, Y9
	VMASKMOVPS (R12)(R10*2), Y15, Y10
	VMASKMOVPS (R12)(R14*1), Y15, Y11
	VMASKMOVPS (BX), Y15, Y12
	VMASKMOVPS (BX)(R10*1), Y15, Y13
	FMA2X4

n4_reduce:
	LEAQ (DI)(R9*1), BX // BX = out row 0 of the tile, row 1 at +R11
	REDUCE4(Y0, Y1, Y2, Y3, (BX))
	REDUCE4(Y4, Y5, Y6, Y7, (BX)(R11*1))

	ADDQ $16, R9           // next 4 channels
	LEAQ (R13)(R10*4), R13 // next 4 wt rows
	CMPQ R9, R11
	JB   n4_tile

	LEAQ (SI)(R10*2), SI // next 2 a rows
	LEAQ (DI)(R11*2), DI // next 2 out rows
	DECQ R8
	JNZ  n4_rowpair

n4_rows1:
	MOVQ  m+32(FP), R8
	TESTQ $1, R8       // m even: no row left
	JZ    n4_done

	XORQ R9, R9
	MOVQ DX, R13

n4_tile1:
	// 1×4 tile: one a row against four wt rows, accumulators Y0..Y3.
	MOVQ   SI, BX
	MOVQ   R13, R12
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3

	MOVQ R10, AX
	SHRQ $5, AX
	JZ   n4_tail1

n4_kloop1:
	VMOVUPS     (BX), Y12
	VFMADD231PS (R12), Y12, Y0        // Y0 += a * wt row 0, the weights from memory
	VFMADD231PS (R12)(R10*1), Y12, Y1
	VFMADD231PS (R12)(R10*2), Y12, Y2
	VFMADD231PS (R12)(R14*1), Y12, Y3
	ADDQ        $32, BX
	ADDQ        $32, R12
	DECQ        AX
	JNZ         n4_kloop1

n4_tail1:
	TESTQ $31, R10
	JZ    n4_reduce1

	VMASKMOVPS  (BX), Y15, Y12
	VMASKMOVPS  (R12), Y15, Y8
	VMASKMOVPS  (R12)(R10*1), Y15, Y9
	VMASKMOVPS  (R12)(R10*2), Y15, Y10
	VMASKMOVPS  (R12)(R14*1), Y15, Y11
	VFMADD231PS Y8, Y12, Y0
	VFMADD231PS Y9, Y12, Y1
	VFMADD231PS Y10, Y12, Y2
	VFMADD231PS Y11, Y12, Y3

n4_reduce1:
	LEAQ (DI)(R9*1), BX
	REDUCE4(Y0, Y1, Y2, Y3, (BX))

	ADDQ $16, R9
	LEAQ (R13)(R10*4), R13
	CMPQ R9, R11
	JB   n4_tile1

n4_done:
	VZEROUPPER
	RET

// func gemmAVX2N1(out, a, wt, bias *float32, m, k int)
//
// The single-channel case n = 1, so out rows are contiguous. m >= 1 and
// k >= 1. Rows are covered by 8×1 tiles: 8 accumulators (one per row, enough
// independent FMA chains for two FMA ports of latency 4) against one weight
// vector, the a rows taken from memory by the FMAs. The remaining m%8 rows
// run a 1×1 loop.
TEXT ·gemmAVX2N1(SB), NOSPLIT, $0-48
	MOVQ out+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ wt+16(FP), DX
	MOVQ bias+24(FP), CX
	MOVQ m+32(FP), R8
	MOVQ k+40(FP), R10

	VBROADCASTSS (CX), Y14 // Y14 = bias in every lane

	SHLQ $2, R10           // R10 = k*4, byte stride of the rows of a
	LEAQ (R10)(R10*2), R11 // R11 = 3*k*4

	MOVQ    R10, AX
	ANDQ    $31, AX
	JZ      n1_main
	LEAQ    gemmTailMask<>+32(SB), BX
	SUBQ    AX, BX
	VMOVUPS (BX), Y15                 // Y15 = tail mask, see gemmAVX2N4

n1_main:
	SHRQ $3, R8 // R8 = m/8 row blocks
	JZ   n1_rows1

n1_block:
	MOVQ   SI, BX           // BX = a row 0, rows 1, 2, 3 at +R10, +2*R10, +R11
	LEAQ   (SI)(R10*4), R13 // R13 = a row 4, rows 5, 6, 7 likewise
	MOVQ   DX, R12          // R12 = wt
	VXORPS Y0, Y0, Y0       // zero the 8 accumulators
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7

	MOVQ R10, AX
	SHRQ $5, AX  // AX = k/8 full steps
	JZ   n1_tail

n1_kloop:
	VMOVUPS     (R12), Y8            // eight k positions of wt
	VFMADD231PS (BX), Y8, Y0         // Y0 += wt * a row 0, the a row from memory
	VFMADD231PS (BX)(R10*1), Y8, Y1
	VFMADD231PS (BX)(R10*2), Y8, Y2
	VFMADD231PS (BX)(R11*1), Y8, Y3
	VFMADD231PS (R13), Y8, Y4
	VFMADD231PS (R13)(R10*1), Y8, Y5
	VFMADD231PS (R13)(R10*2), Y8, Y6
	VFMADD231PS (R13)(R11*1), Y8, Y7
	ADDQ        $32, BX
	ADDQ        $32, R13
	ADDQ        $32, R12
	DECQ        AX
	JNZ         n1_kloop

n1_tail:
	TESTQ $31, R10
	JZ    n1_reduce

	VMASKMOVPS  (R12), Y15, Y8         // the last k%8 elements, the other lanes zero
	VMASKMOVPS  (BX), Y15, Y9
	VMASKMOVPS  (BX)(R10*1), Y15, Y10
	VMASKMOVPS  (BX)(R10*2), Y15, Y11
	VMASKMOVPS  (BX)(R11*1), Y15, Y12
	VFMADD231PS Y9, Y8, Y0
	VFMADD231PS Y10, Y8, Y1
	VFMADD231PS Y11, Y8, Y2
	VFMADD231PS Y12, Y8, Y3
	VMASKMOVPS  (R13), Y15, Y9
	VMASKMOVPS  (R13)(R10*1), Y15, Y10
	VMASKMOVPS  (R13)(R10*2), Y15, Y11
	VMASKMOVPS  (R13)(R11*1), Y15, Y12
	VFMADD231PS Y9, Y8, Y4
	VFMADD231PS Y10, Y8, Y5
	VFMADD231PS Y11, Y8, Y6
	VFMADD231PS Y12, Y8, Y7

n1_reduce:
	VHADDPS    Y1, Y0, Y0        // Y0 = [r0 0+1, r0 2+3, r1 0+1, r1 2+3 | lanes 4..7]
	VHADDPS    Y3, Y2, Y2        // likewise rows 2, 3
	VHADDPS    Y5, Y4, Y4        // rows 4, 5
	VHADDPS    Y7, Y6, Y6        // rows 6, 7
	VHADDPS    Y2, Y0, Y0        // Y0 = [rows 0..3 over lanes 0..3 | over lanes 4..7]
	VHADDPS    Y6, Y4, Y4        // Y4 = the same for rows 4..7
	VPERM2F128 $0x20, Y4, Y0, Y1 // Y1 = [Y0 low half, Y4 low half]
	VPERM2F128 $0x31, Y4, Y0, Y2 // Y2 = [Y0 high half, Y4 high half]
	VADDPS     Y2, Y1, Y1        // Y1 = [sum row 0, ..., sum row 7]
	VADDPS     Y14, Y1, Y1       // += bias
	VMOVUPS    Y1, (DI)          // eight contiguous outputs (n = 1)

	ADDQ $32, DI
	LEAQ (SI)(R10*8), SI // next 8 a rows
	DECQ R8
	JNZ  n1_block

n1_rows1:
	MOVQ m+32(FP), R8
	ANDQ $7, R8       // R8 = m%8 remaining rows
	JZ   n1_done

n1_row:
	// 1×1: one a row against wt, one accumulator Y0.
	MOVQ   SI, BX
	MOVQ   DX, R12
	VXORPS Y0, Y0, Y0

	MOVQ R10, AX
	SHRQ $5, AX
	JZ   n1_tail1

n1_kloop1:
	VMOVUPS     (R12), Y8
	VFMADD231PS (BX), Y8, Y0
	ADDQ        $32, BX
	ADDQ        $32, R12
	DECQ        AX
	JNZ         n1_kloop1

n1_tail1:
	TESTQ $31, R10
	JZ    n1_reduce1

	VMASKMOVPS  (R12), Y15, Y8
	VMASKMOVPS  (BX), Y15, Y9
	VFMADD231PS Y9, Y8, Y0

n1_reduce1:
	VEXTRACTF128 $1, Y0, X1
	VADDPS       X1, X0, X0  // X0 = [l0+l4, l1+l5, l2+l6, l3+l7]
	VHADDPS      X0, X0, X0  // X0 = [l0+l4+l1+l5, l2+l6+l3+l7, the same]
	VHADDPS      X0, X0, X0  // the sum in every lane
	VADDSS       X14, X0, X0 // += bias
	VMOVSS       X0, (DI)

	ADDQ $4, DI
	ADDQ R10, SI // next a row
	DECQ R8
	JNZ  n1_row

n1_done:
	VZEROUPPER
	RET
