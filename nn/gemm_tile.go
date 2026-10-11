//go:build !wasm && !amd64

package nn

// tileN is the output-channel width of the multi-channel GEMM tile. arm64
// has 32 floating-point registers and runs the 4×4 tile fastest.
const tileN = 4
