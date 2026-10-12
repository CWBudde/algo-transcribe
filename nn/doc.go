// Package nn is a minimal float32 inference library for the convolutional
// network of basic-pitch.
//
// Tensors are NHWC with a batch size of one: a plain []float32 plus its
// height, width and channel count (see [Tensor]). There is no generic tensor
// framework; the package offers exactly the operations the model needs:
//
//   - [Conv2D]: 2-D convolution with TensorFlow padding="same" semantics,
//     arbitrary (h, w) strides, HWIO weights and a bias, computed as im2col
//     plus a blocked float32 GEMM. [Conv2D.ForwardRows] computes a range of
//     output rows so a caller can bound the work per step.
//   - [ReLU] and [Sigmoid], in place on a slice.
//   - [ConcatChannels]: concatenation along the channel axis.
//   - [FoldBatchNorm]: folds an inference BatchNorm into conv weights and bias.
//
// The GEMM behind [Conv2D] is plain Go, register-blocked so that it stays
// fast under GOOS=js GOARCH=wasm, where it always runs. On arm64 it is
// preceded by NEON assembly kernels (dot form over the transposed weights)
// that cover every basic-pitch layer shape, and on amd64 by AVX2 kernels with
// the same coverage when the CPU has AVX2 and FMA (detected at run time). The
// Go tiles remain the fallback for other shapes and CPUs and run everywhere
// under the purego build tag. There is no cgo, and nothing allocates after
// setup.
package nn
