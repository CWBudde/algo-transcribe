//go:build wasm || amd64

package nn

// tileN is the output-channel width of the multi-channel GEMM tile. Go's
// wasm backend and amd64 (without AVX2, or under the purego tag) have 16
// float registers, so the 4×4 tile spills and the 4×2 tile runs faster:
// more than twice as fast under wasm, and 1.6 to 1.8 times on an Intel Xeon
// Gold 5520+ (docs/benchmarks/2026-10-11-simd-gemm.md).
const tileN = 2
