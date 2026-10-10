//go:build wasm

package nn

// tileN is the output-channel width of the multi-channel GEMM tile. Go's
// wasm backend has 16 float32 registers, so the 4×4 tile spills and the
// 4×2 tile runs more than twice as fast.
const tileN = 2
