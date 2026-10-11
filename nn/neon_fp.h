// Vector floating-point arithmetic for Go's arm64 assembler.
//
// Adapted from github.com/cwbudde/algo-fft internal/asm/arm64/neon_fp.h.
//
// Go's arm64 assembler up to at least Go 1.25 (the module's minimum) has no
// mnemonic for the NEON vector FADD or the pairwise FADDP; they are rejected
// as "unrecognized instruction". Go 1.27 knows them as VFADD and VFADDP, but
// the module must still assemble with 1.25, so the encodings are emitted
// directly with WORD.
//
// The macros take register NUMBERS, not names, because the encoding embeds the
// number and Go's assembler preprocessor does not support token pasting. Callers
// pass the bare index and the identity V<n> <-> n holds everywhere:
// VADDF_S4(0, 2, 8) means "V0 + V2 -> V8". Keep that correspondence literal,
// since nothing checks it; the reference tests in gemm_simd_test.go catch a
// wrong number.
//
// Encodings (ARM DDI 0487, Advanced SIMD three same, Q=1, sz=0):
//   FADD  Vd.4S, Vn.4S, Vm.4S   0x4E20D400 | Rm<<16 | Rn<<5 | Rd
//   FADDP Vd.4S, Vn.4S, Vm.4S   0x6E20D400 | Rm<<16 | Rn<<5 | Rd
//
// FADDP (vector) is "0 Q 1 01110 0 sz 1 Rm 11010 1 Rn Rd": it differs from
// FADD only in the U bit (bit 29). Both encodings were checked against the
// output of Go 1.27's native VFADD and VFADDP (identical object bytes) and
// against `go tool objdump`, which decodes the words as FADD and FADDP.

// float32 (.4S): d = a + b.
#define VADDF_S4(a, b, d) WORD $(0x4E20D400 | ((b)<<16) | ((a)<<5) | (d))

// float32 (.4S) pairwise add: d = [a0+a1, a2+a3, b0+b1, b2+b3].
#define VFADDP_S4(a, b, d) WORD $(0x6E20D400 | ((b)<<16) | ((a)<<5) | (d))
