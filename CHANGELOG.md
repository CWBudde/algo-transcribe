# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

- Repository scaffold: module, CI, release guard and the basic-pitch port
  specification (`docs/port-spec.md`).
- Package `nn`: float32 NHWC inference ops for the basic-pitch CNN —
  `Conv2D` (TF `padding="same"`, strides, HWIO weights, im2col plus blocked
  GEMM, row-steppable `ForwardRows`), `ReLU`, `Sigmoid`, `ConcatChannels`
  and `FoldBatchNorm`.
- `internal/onnx`: dependency-free ONNX protobuf reader.
- `basicpitch`: embedded basic-pitch v0.4.0 weights (`weights.bin`, HWIO
  kernels, CQT constants), generated from `testdata/nmp.onnx` by
  `internal/cmd/extractweights` via `go generate`.
