//go:build (!arm64 && !amd64) || purego

package nn

// haveSIMD reports whether gemm has native SIMD kernels on this target.
const haveSIMD = false

// gemmSIMD has no kernels on this target and always leaves the work to the
// Go tiles.
func gemmSIMD(_, _, _, _ []float32, _, _ int) bool { return false }
