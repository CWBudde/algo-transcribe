//go:build !wasm

package nn

// tileN is the output-channel width of the multi-channel GEMM tile. Native
// targets have at least 16 (amd64) and usually 32 (arm64) floating-point
// registers and run the 4×4 tile fastest.
const tileN = 4
