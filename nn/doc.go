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
// Everything is plain Go without assembly or cgo so that it stays fast under
// GOOS=js GOARCH=wasm, and nothing allocates after setup.
package nn
