//go:build arm64 && !purego

// NEON GEMM kernels for nn, dot form over the transposed weights.
//
// Both kernels compute out[i*n+j] = bias[j] + dot(a[i*k:(i+1)*k],
// wt[j*k:(j+1)*k]) with a, wt and out in the layout of gemm (gemm.go). They
// vectorise over k: every accumulator is a .S4 vector of partial sums over
// four consecutive k positions, and a pairwise-add (FADDP) tree reduces the
// accumulators to the scalar dot products after the k loop.
//
// The k%4 tail is handled without scalar code: the last four elements of
// each row are loaded again, overlapping the last full step, and the lanes
// that were already summed are zeroed with a mask before the FMA. Both
// operands are masked, so the zeroed lanes contribute 0*0 = +0 even when the
// data holds infinities. This needs k >= 4, which the Go wrapper checks.
//
// Plan 9 operand order: sources first, destination last.
//   VFMLA Vm.S4, Vn.S4, Vd.S4   Vd += Vn * Vm
//   VAND  Vm.B16, Vn.B16, Vd.B16  Vd = Vn & Vm
// The numeric macros VADDF_S4 and VFADDP_S4 come from neon_fp.h.
//
// Registers R18 (platform), R26 (closure context), R27 (assembler temp),
// R28 (g), R29 (frame pointer) and R30 (link register) are left alone.

#include "textflag.h"
#include "neon_fp.h"

// gemmTailMask holds, for tail = k%4 in 1..3, the 16-byte lane mask at
// offset (tail-1)*16 that keeps the last tail lanes of a vector loaded from
// element k-4 of a row, i.e. lanes [4-tail, 4). Lane 0 is at the lowest
// address, so lane 3 is the upper half of the second doubleword.
DATA gemmTailMask<>+0x00(SB)/8, $0x0000000000000000 // tail 1: lane 3
DATA gemmTailMask<>+0x08(SB)/8, $0xffffffff00000000
DATA gemmTailMask<>+0x10(SB)/8, $0x0000000000000000 // tail 2: lanes 2, 3
DATA gemmTailMask<>+0x18(SB)/8, $0xffffffffffffffff
DATA gemmTailMask<>+0x20(SB)/8, $0xffffffff00000000 // tail 3: lanes 1, 2, 3
DATA gemmTailMask<>+0x28(SB)/8, $0xffffffffffffffff
GLOBL gemmTailMask<>(SB), RODATA|NOPTR, $48

// FMA4X4 accumulates the outer product of the a vectors V16..V19 (rows) and
// the weight vectors V20..V23 (channels) into V0..V15, accumulator
// V(4*row+channel). Each lane holds the partial sum of one k position.
#define FMA4X4 \
	VFMLA V20.S4, V16.S4, V0.S4;  \
	VFMLA V21.S4, V16.S4, V1.S4;  \
	VFMLA V22.S4, V16.S4, V2.S4;  \
	VFMLA V23.S4, V16.S4, V3.S4;  \
	VFMLA V20.S4, V17.S4, V4.S4;  \
	VFMLA V21.S4, V17.S4, V5.S4;  \
	VFMLA V22.S4, V17.S4, V6.S4;  \
	VFMLA V23.S4, V17.S4, V7.S4;  \
	VFMLA V20.S4, V18.S4, V8.S4;  \
	VFMLA V21.S4, V18.S4, V9.S4;  \
	VFMLA V22.S4, V18.S4, V10.S4; \
	VFMLA V23.S4, V18.S4, V11.S4; \
	VFMLA V20.S4, V19.S4, V12.S4; \
	VFMLA V21.S4, V19.S4, V13.S4; \
	VFMLA V22.S4, V19.S4, V14.S4; \
	VFMLA V23.S4, V19.S4, V15.S4

// REDUCE4 reduces the four accumulators c0..c3 of one row (one per output
// channel) to the vector [sum(c0), sum(c1), sum(c2), sum(c3)] in V26, adds
// the bias vector V28 and stores the four outputs at (dst):
//
//	V24 = [c0[0]+c0[1], c0[2]+c0[3], c1[0]+c1[1], c1[2]+c1[3]]
//	V25 = the same for c2, c3
//	V26 = [sum c0, sum c1, sum c2, sum c3] + bias
#define REDUCE4(c0, c1, c2, c3, dst) \
	VFADDP_S4(c0, c1, 24);  \
	VFADDP_S4(c2, c3, 25);  \
	VFADDP_S4(24, 25, 26);  \
	VADDF_S4(26, 28, 26);   \
	VST1 [V26.S4], (dst)

// func gemmNEON4x4(out, a, wt, bias *float32, m, n, k int)
//
// n must be a positive multiple of 4, m >= 1 and k >= 4. Rows are covered
// by 4×4 tiles (4 rows × 4 channels, 16 accumulators), the remaining m%4
// rows by 1×4 tiles.
TEXT ·gemmNEON4x4(SB), NOSPLIT, $0-56
	MOVD out+0(FP), R0
	MOVD a+8(FP), R1
	MOVD wt+16(FP), R2
	MOVD bias+24(FP), R3
	MOVD m+32(FP), R4
	MOVD n+40(FP), R5
	MOVD k+48(FP), R6

	LSL $2, R6, R9    // R9 = k*4, byte stride of the rows of a and wt
	LSL $2, R5, R10   // R10 = n*4, byte stride of the rows of out
	AND $3, R6, R8    // R8 = tail = k%4
	LSR $2, R6, R6    // R6 = k/4, full k steps (>= 1)
	SUB $4, R8, R11   // R11 = tail-4
	LSL $2, R11, R11  // R11 = (tail-4)*4: from the end of the full steps back to element k-4
	CBZ R8, mc_main

	MOVD $gemmTailMask<>(SB), R12
	SUB  $1, R8, R24
	ADD  R24<<4, R12, R12 // R12 = &gemmTailMask[(tail-1)*16]
	VLD1 (R12), [V31.S4]  // V31 = tail mask, constant for the whole call

mc_main:
	LSR $2, R4, R12 // R12 = m/4 row blocks
	CBZ R12, mc_rows1

mc_rowblock:
	MOVD R2, R24    // R24 = wt rows of the current channel block
	MOVD R3, R23    // R23 = bias of the current channel block
	MOVD R0, R22    // R22 = out tile
	LSR  $2, R5, R7 // R7 = n/4 channel blocks

mc_tile:
	MOVD R1, R13       // R13..R16 = a rows 0..3
	ADD  R9, R13, R14
	ADD  R9, R14, R15
	ADD  R9, R15, R16
	MOVD R24, R17      // R17, R19, R20, R21 = wt rows 0..3 (R18 is reserved)
	ADD  R9, R17, R19
	ADD  R9, R19, R20
	ADD  R9, R20, R21

	VEOR V0.B16, V0.B16, V0.B16 // zero the 16 accumulators
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	MOVD R6, R25 // R25 = k/4 steps left

mc_kloop:
	VLD1.P 16(R13), [V16.S4] // four k positions of each a row, post-increment
	VLD1.P 16(R14), [V17.S4]
	VLD1.P 16(R15), [V18.S4]
	VLD1.P 16(R16), [V19.S4]
	VLD1.P 16(R17), [V20.S4] // and of each wt row
	VLD1.P 16(R19), [V21.S4]
	VLD1.P 16(R20), [V22.S4]
	VLD1.P 16(R21), [V23.S4]
	FMA4X4
	SUB  $1, R25
	CBNZ R25, mc_kloop

	CBZ R8, mc_reduce

	// k%4 tail: reload elements k-4..k-1 and mask off the summed lanes.
	ADD  R11, R13, R13
	ADD  R11, R14, R14
	ADD  R11, R15, R15
	ADD  R11, R16, R16
	ADD  R11, R17, R17
	ADD  R11, R19, R19
	ADD  R11, R20, R20
	ADD  R11, R21, R21
	VLD1 (R13), [V16.S4]
	VLD1 (R14), [V17.S4]
	VLD1 (R15), [V18.S4]
	VLD1 (R16), [V19.S4]
	VLD1 (R17), [V20.S4]
	VLD1 (R19), [V21.S4]
	VLD1 (R20), [V22.S4]
	VLD1 (R21), [V23.S4]
	VAND V31.B16, V16.B16, V16.B16
	VAND V31.B16, V17.B16, V17.B16
	VAND V31.B16, V18.B16, V18.B16
	VAND V31.B16, V19.B16, V19.B16
	VAND V31.B16, V20.B16, V20.B16
	VAND V31.B16, V21.B16, V21.B16
	VAND V31.B16, V22.B16, V22.B16
	VAND V31.B16, V23.B16, V23.B16
	FMA4X4

mc_reduce:
	VLD1 (R23), [V28.S4] // V28 = bias of the four channels
	MOVD R22, R13        // R13 = out row 0 of the tile
	REDUCE4(0, 1, 2, 3, R13)
	ADD  R10, R13, R13
	REDUCE4(4, 5, 6, 7, R13)
	ADD  R10, R13, R13
	REDUCE4(8, 9, 10, 11, R13)
	ADD  R10, R13, R13
	REDUCE4(12, 13, 14, 15, R13)

	ADD  $16, R23, R23     // next 4 biases
	ADD  $16, R22, R22     // next 4 output columns
	ADD  R9<<2, R24, R24   // next 4 wt rows
	SUB  $1, R7
	CBNZ R7, mc_tile

	ADD  R9<<2, R1, R1     // next 4 a rows
	ADD  R10<<2, R0, R0    // next 4 out rows
	SUB  $1, R12
	CBNZ R12, mc_rowblock

mc_rows1:
	AND $3, R4, R12 // R12 = m%4 remaining rows
	CBZ R12, mc_done

mc_row:
	MOVD R2, R24
	MOVD R3, R23
	MOVD R0, R22
	LSR  $2, R5, R7

mc_tile1:
	// 1×4 tile: one a row against four wt rows, accumulators V0..V3.
	MOVD R1, R13
	MOVD R24, R17
	ADD  R9, R17, R19
	ADD  R9, R19, R20
	ADD  R9, R20, R21

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16

	MOVD R6, R25

mc_kloop1:
	VLD1.P 16(R13), [V16.S4]
	VLD1.P 16(R17), [V20.S4]
	VLD1.P 16(R19), [V21.S4]
	VLD1.P 16(R20), [V22.S4]
	VLD1.P 16(R21), [V23.S4]
	VFMLA  V20.S4, V16.S4, V0.S4
	VFMLA  V21.S4, V16.S4, V1.S4
	VFMLA  V22.S4, V16.S4, V2.S4
	VFMLA  V23.S4, V16.S4, V3.S4
	SUB    $1, R25
	CBNZ   R25, mc_kloop1

	CBZ R8, mc_reduce1

	ADD  R11, R13, R13
	ADD  R11, R17, R17
	ADD  R11, R19, R19
	ADD  R11, R20, R20
	ADD  R11, R21, R21
	VLD1 (R13), [V16.S4]
	VLD1 (R17), [V20.S4]
	VLD1 (R19), [V21.S4]
	VLD1 (R20), [V22.S4]
	VLD1 (R21), [V23.S4]
	VAND V31.B16, V16.B16, V16.B16
	VAND V31.B16, V20.B16, V20.B16
	VAND V31.B16, V21.B16, V21.B16
	VAND V31.B16, V22.B16, V22.B16
	VAND V31.B16, V23.B16, V23.B16
	VFMLA V20.S4, V16.S4, V0.S4
	VFMLA V21.S4, V16.S4, V1.S4
	VFMLA V22.S4, V16.S4, V2.S4
	VFMLA V23.S4, V16.S4, V3.S4

mc_reduce1:
	VLD1 (R23), [V28.S4]
	REDUCE4(0, 1, 2, 3, R22)

	ADD  $16, R23, R23
	ADD  $16, R22, R22
	ADD  R9<<2, R24, R24
	SUB  $1, R7
	CBNZ R7, mc_tile1

	ADD  R9, R1, R1   // next a row
	ADD  R10, R0, R0  // next out row
	SUB  $1, R12
	CBNZ R12, mc_row

mc_done:
	RET

// func gemmNEON8x1(out, a, wt, bias *float32, m, k int)
//
// The single-channel case n = 1, so out rows are contiguous. m >= 1 and
// k >= 4. Rows are covered by 8×1 tiles with 16 accumulators: V0..V7 take
// the even and V8..V15 the odd 4-element k blocks of rows 0..7, which keeps
// 16 independent FMA chains in flight. The remaining m%8 rows run a 1×1
// loop.
TEXT ·gemmNEON8x1(SB), NOSPLIT, $0-48
	MOVD out+0(FP), R0
	MOVD a+8(FP), R1
	MOVD wt+16(FP), R2
	MOVD bias+24(FP), R3
	MOVD m+32(FP), R4
	MOVD k+40(FP), R6

	VLD1R (R3), [V30.S4] // V30 = bias broadcast to all lanes

	LSL $2, R6, R9    // R9 = k*4, byte stride of the rows of a
	LSR $3, R6, R7    // R7 = k/8 double steps
	AND $3, R6, R8    // R8 = tail = k%4
	SUB $4, R8, R11
	LSL $2, R11, R11  // R11 = (tail-4)*4, see gemmNEON4x4
	CBZ R8, sc_main

	MOVD $gemmTailMask<>(SB), R12
	SUB  $1, R8, R24
	ADD  R24<<4, R12, R12
	VLD1 (R12), [V31.S4] // V31 = tail mask

sc_main:
	LSR $3, R4, R12 // R12 = m/8 row blocks
	CBZ R12, sc_rows1

sc_rowblock:
	MOVD R1, R13       // R13..R17, R19..R21 = a rows 0..7 (R18 is reserved)
	ADD  R9, R13, R14
	ADD  R9, R14, R15
	ADD  R9, R15, R16
	ADD  R9, R16, R17
	ADD  R9, R17, R19
	ADD  R9, R19, R20
	ADD  R9, R20, R21
	MOVD R2, R22       // R22 = wt

	VEOR V0.B16, V0.B16, V0.B16 // zero the 16 accumulators
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	CBZ  R7, sc_single
	MOVD R7, R25 // R25 = k/8 double steps left

sc_kloop:
	VLD1.P 32(R22), [V24.S4, V25.S4] // wt, eight k positions as two vectors
	VLD1.P 32(R13), [V16.S4, V17.S4] // rows 0..3, eight k positions each
	VLD1.P 32(R14), [V18.S4, V19.S4]
	VLD1.P 32(R15), [V20.S4, V21.S4]
	VLD1.P 32(R16), [V22.S4, V23.S4]
	VFMLA  V24.S4, V16.S4, V0.S4  // even block into V0..V7,
	VFMLA  V25.S4, V17.S4, V8.S4  // odd block into V8..V15
	VFMLA  V24.S4, V18.S4, V1.S4
	VFMLA  V25.S4, V19.S4, V9.S4
	VFMLA  V24.S4, V20.S4, V2.S4
	VFMLA  V25.S4, V21.S4, V10.S4
	VFMLA  V24.S4, V22.S4, V3.S4
	VFMLA  V25.S4, V23.S4, V11.S4
	VLD1.P 32(R17), [V16.S4, V17.S4] // rows 4..7, reusing V16..V23
	VLD1.P 32(R19), [V18.S4, V19.S4]
	VLD1.P 32(R20), [V20.S4, V21.S4]
	VLD1.P 32(R21), [V22.S4, V23.S4]
	VFMLA  V24.S4, V16.S4, V4.S4
	VFMLA  V25.S4, V17.S4, V12.S4
	VFMLA  V24.S4, V18.S4, V5.S4
	VFMLA  V25.S4, V19.S4, V13.S4
	VFMLA  V24.S4, V20.S4, V6.S4
	VFMLA  V25.S4, V21.S4, V14.S4
	VFMLA  V24.S4, V22.S4, V7.S4
	VFMLA  V25.S4, V23.S4, V15.S4
	SUB    $1, R25
	CBNZ   R25, sc_kloop

sc_single:
	TBZ $2, R6, sc_tail // bit 2 of k clear: no single 4-element step left

	VLD1.P 16(R22), [V24.S4]
	VLD1.P 16(R13), [V16.S4]
	VLD1.P 16(R14), [V17.S4]
	VLD1.P 16(R15), [V18.S4]
	VLD1.P 16(R16), [V19.S4]
	VLD1.P 16(R17), [V20.S4]
	VLD1.P 16(R19), [V21.S4]
	VLD1.P 16(R20), [V22.S4]
	VLD1.P 16(R21), [V23.S4]
	VFMLA  V24.S4, V16.S4, V0.S4
	VFMLA  V24.S4, V17.S4, V1.S4
	VFMLA  V24.S4, V18.S4, V2.S4
	VFMLA  V24.S4, V19.S4, V3.S4
	VFMLA  V24.S4, V20.S4, V4.S4
	VFMLA  V24.S4, V21.S4, V5.S4
	VFMLA  V24.S4, V22.S4, V6.S4
	VFMLA  V24.S4, V23.S4, V7.S4

sc_tail:
	CBZ R8, sc_reduce

	// k%4 tail: reload elements k-4..k-1, mask, accumulate into V8..V15.
	ADD  R11, R22, R22
	ADD  R11, R13, R13
	ADD  R11, R14, R14
	ADD  R11, R15, R15
	ADD  R11, R16, R16
	ADD  R11, R17, R17
	ADD  R11, R19, R19
	ADD  R11, R20, R20
	ADD  R11, R21, R21
	VLD1 (R22), [V24.S4]
	VLD1 (R13), [V16.S4]
	VLD1 (R14), [V17.S4]
	VLD1 (R15), [V18.S4]
	VLD1 (R16), [V19.S4]
	VLD1 (R17), [V20.S4]
	VLD1 (R19), [V21.S4]
	VLD1 (R20), [V22.S4]
	VLD1 (R21), [V23.S4]
	VAND V31.B16, V24.B16, V24.B16
	VAND V31.B16, V16.B16, V16.B16
	VAND V31.B16, V17.B16, V17.B16
	VAND V31.B16, V18.B16, V18.B16
	VAND V31.B16, V19.B16, V19.B16
	VAND V31.B16, V20.B16, V20.B16
	VAND V31.B16, V21.B16, V21.B16
	VAND V31.B16, V22.B16, V22.B16
	VAND V31.B16, V23.B16, V23.B16
	VFMLA V24.S4, V16.S4, V8.S4
	VFMLA V24.S4, V17.S4, V9.S4
	VFMLA V24.S4, V18.S4, V10.S4
	VFMLA V24.S4, V19.S4, V11.S4
	VFMLA V24.S4, V20.S4, V12.S4
	VFMLA V24.S4, V21.S4, V13.S4
	VFMLA V24.S4, V22.S4, V14.S4
	VFMLA V24.S4, V23.S4, V15.S4

sc_reduce:
	VADDF_S4(0, 8, 0)    // merge the even and odd accumulators of each row
	VADDF_S4(1, 9, 1)
	VADDF_S4(2, 10, 2)
	VADDF_S4(3, 11, 3)
	VADDF_S4(4, 12, 4)
	VADDF_S4(5, 13, 5)
	VADDF_S4(6, 14, 6)
	VADDF_S4(7, 15, 7)
	VFADDP_S4(0, 1, 16)  // V16 = [row 0 pairs, row 1 pairs]
	VFADDP_S4(2, 3, 17)  // V17 = [row 2 pairs, row 3 pairs]
	VFADDP_S4(16, 17, 18) // V18 = [sum row 0, 1, 2, 3]
	VFADDP_S4(4, 5, 20)
	VFADDP_S4(6, 7, 21)
	VFADDP_S4(20, 21, 19) // V19 = [sum row 4, 5, 6, 7]
	VADDF_S4(18, 30, 18) // += bias
	VADDF_S4(19, 30, 19)
	VST1.P [V18.S4, V19.S4], 32(R0) // eight contiguous outputs (n = 1)

	ADD  R9<<3, R1, R1 // next 8 a rows
	SUB  $1, R12
	CBNZ R12, sc_rowblock

sc_rows1:
	AND $7, R4, R12 // R12 = m%8 remaining rows
	CBZ R12, sc_done

sc_row:
	// 1×1: one a row against wt, one accumulator V0.
	MOVD R1, R13
	MOVD R2, R22
	VEOR V0.B16, V0.B16, V0.B16
	LSR  $2, R6, R25 // R25 = k/4 steps (>= 1)

sc_kloop1:
	VLD1.P 16(R13), [V16.S4]
	VLD1.P 16(R22), [V24.S4]
	VFMLA  V24.S4, V16.S4, V0.S4
	SUB    $1, R25
	CBNZ   R25, sc_kloop1

	CBZ R8, sc_reduce1

	ADD   R11, R13, R13
	ADD   R11, R22, R22
	VLD1  (R13), [V16.S4]
	VLD1  (R22), [V24.S4]
	VAND  V31.B16, V16.B16, V16.B16
	VAND  V31.B16, V24.B16, V24.B16
	VFMLA V24.S4, V16.S4, V0.S4

sc_reduce1:
	VFADDP_S4(0, 0, 1)   // V1 = [v0+v1, v2+v3, v0+v1, v2+v3]
	VFADDP_S4(1, 1, 2)   // V2 = the sum in every lane
	VADDF_S4(2, 30, 2)   // += bias
	FMOVS F2, (R0)       // store lane 0 (F2 is the low lane of V2)
	ADD   $4, R0, R0

	ADD  R9, R1, R1
	SUB  $1, R12
	CBNZ R12, sc_row

sc_done:
	RET
