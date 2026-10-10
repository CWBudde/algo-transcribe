package main

import (
	"fmt"
	"slices"

	"github.com/cwbudde/algo-transcribe/internal/onnx"
)

// frontEndTensors extracts the constants of the in-graph CQT front end
// (docs/port-spec.md §3): the real and imaginary top-octave kernels, the
// low-pass filter, the sqrt-length factors and the folded CQT BatchNorm.
func frontEndTensors(g *onnx.Graph, convs modelConvs) ([]namedTensor, error) {
	kReal, kImag, lowpass, err := cqtConvs(g, convs)
	if err != nil {
		return nil, err
	}

	sqrtLengths, err := sqrtLengthFactors(g)
	if err != nil {
		return nil, err
	}

	scale, offset, err := cqtBatchNorm(g, convs.harmonicStack)
	if err != nil {
		return nil, err
	}

	parts := []struct {
		name, ident string
		t           *onnx.Tensor
		shape       []int
	}{
		{"cqt/kernel_real", "CQTKernelReal", kReal, []int{cqtBins, cqtFFT}},
		{"cqt/kernel_imag", "CQTKernelImag", kImag, []int{cqtBins, cqtFFT}},
		{"cqt/lowpass", "CQTLowpass", lowpass, []int{cqtFFT}},
		{"cqt/sqrt_lengths", "CQTSqrtLengths", sqrtLengths, []int{cqtOutputBins}},
		{"batch_normalization/scale", "BatchNormScale", scale, []int{1}},
		{"batch_normalization/offset", "BatchNormOffset", offset, []int{1}},
	}

	out := make([]namedTensor, 0, len(parts))

	for _, p := range parts {
		data, err := p.t.Float32s()
		if err != nil {
			return nil, fmt.Errorf("extractweights: %s: %w", p.name, err)
		}

		out = append(out, namedTensor{name: p.name, ident: p.ident, shape: p.shape, data: data})
	}

	return out, nil
}

// cqtConvs classifies every Conv that is not a model layer. There must be 18
// CQT convolutions, real and imaginary for each of the 9 octaves with strides
// 256 down to 1, and 8 stride-2 low-pass convolutions. The imaginary kernel is
// the one whose Conv output is negated (port-spec §3.1); each kind must use a
// single kernel initializer throughout.
func cqtConvs(g *onnx.Graph, convs modelConvs) (kReal, kImag, lowpass *onnx.Tensor, err error) {
	model := []*onnx.Node{convs.conv2d1, convs.contours, convs.conv2d2, convs.conv2d3, convs.conv2d4, convs.conv2d5}

	var realStrides, imagStrides []int64

	nLowpass := 0

	for i := range g.Nodes {
		n := &g.Nodes[i]
		if n.OpType != opConv || slices.Contains(model, n) {
			continue
		}

		if len(n.Inputs) != 3 {
			return nil, nil, nil, fmt.Errorf("extractweights: front-end Conv %q has %d inputs, want 3", n.Name, len(n.Inputs))
		}

		k := g.Initializer(n.Inputs[1])
		if k == nil {
			return nil, nil, nil, fmt.Errorf("extractweights: front-end Conv %q: kernel is not an initializer", n.Name)
		}

		switch {
		case slices.Equal(k.Dims, []int64{cqtBins, 1, 1, cqtFFT}):
			stride, isImag, err := cqtConv(g, n)
			if err != nil {
				return nil, nil, nil, err
			}

			dst, strides := &kReal, &realStrides
			if isImag {
				dst, strides = &kImag, &imagStrides
			}

			if *dst != nil && *dst != k {
				return nil, nil, nil, fmt.Errorf("extractweights: CQT Conv %q uses a second kernel initializer", n.Name)
			}

			*dst = k

			*strides = append(*strides, stride)
		case slices.Equal(k.Dims, []int64{1, 1, 1, cqtFFT}):
			if _, _, err := frontConvWeights(g, n, k.Dims, []int64{1, 2}); err != nil {
				return nil, nil, nil, fmt.Errorf("extractweights: low-pass: %w", err)
			}

			if lowpass != nil && lowpass != k {
				return nil, nil, nil, fmt.Errorf("extractweights: low-pass Conv %q uses a second kernel initializer", n.Name)
			}

			lowpass = k
			nLowpass++
		default:
			return nil, nil, nil, fmt.Errorf("extractweights: unexpected Conv %q with kernel dims %v", n.Name, k.Dims)
		}
	}

	if err := checkCQTStrides("real", realStrides); err != nil {
		return nil, nil, nil, err
	}

	if err := checkCQTStrides("imaginary", imagStrides); err != nil {
		return nil, nil, nil, err
	}

	if kReal == kImag {
		return nil, nil, nil, fmt.Errorf("extractweights: real and imaginary CQT convolutions share a kernel")
	}

	if nLowpass != lowpassConvs {
		return nil, nil, nil, fmt.Errorf("extractweights: %d low-pass convolutions, want %d", nLowpass, lowpassConvs)
	}

	return kReal, kImag, lowpass, nil
}

// cqtConv checks one CQT Conv and reports its frequency stride and whether
// its output is negated (the imaginary part).
func cqtConv(g *onnx.Graph, n *onnx.Node) (stride int64, isImag bool, err error) {
	a := n.Attr("strides")
	if a == nil || len(a.Ints) != 2 {
		return 0, false, fmt.Errorf("extractweights: CQT Conv %q: want two strides", n.Name)
	}

	if _, _, err := frontConvWeights(g, n, []int64{cqtBins, 1, 1, cqtFFT}, a.Ints); err != nil {
		return 0, false, fmt.Errorf("extractweights: CQT: %w", err)
	}

	if a.Ints[0] != 1 {
		return 0, false, fmt.Errorf("extractweights: CQT Conv %q: strides %v, want [1 s]", n.Name, a.Ints)
	}

	for _, out := range n.Outputs {
		for _, c := range g.Consumers(out) {
			if c.OpType == "Neg" {
				isImag = true
			}
		}
	}

	return a.Ints[1], isImag, nil
}

// frontConvWeights checks a front-end Conv (no padding) and requires its
// bias to be all zero.
func frontConvWeights(g *onnx.Graph, n *onnx.Node, kernel, strides []int64) (*onnx.Tensor, *onnx.Tensor, error) {
	k, b, err := convWeights(g, n, kernel, strides, nil)
	if err != nil {
		return nil, nil, err
	}

	bias, err := b.Float32s()
	if err != nil {
		return nil, nil, fmt.Errorf("conv %q bias: %w", n.Name, err)
	}

	if slices.ContainsFunc(bias, func(v float32) bool { return v != 0 }) {
		return nil, nil, fmt.Errorf("conv %q has a nonzero bias", n.Name)
	}

	return k, b, nil
}

// checkCQTStrides requires one convolution per octave with hops 256, 128, …,
// 1 (port-spec §3.1), in any order.
func checkCQTStrides(kind string, strides []int64) error {
	want := make([]int64, 0, cqtOctaves)
	for o := range cqtOctaves {
		want = append(want, cqtFFT>>o)
	}

	got := slices.Clone(strides)
	slices.Sort(got)
	slices.Sort(want)

	if !slices.Equal(got, want) {
		return fmt.Errorf("extractweights: %s CQT convolutions have hops %v, want %v", kind, strides, want)
	}

	return nil
}

// sqrtLengthFactors returns the [309,1,1] float initializer that a Mul node
// applies to the CQT before the magnitude. Exactly one such Mul must exist.
func sqrtLengthFactors(g *onnx.Graph) (*onnx.Tensor, error) {
	var found *onnx.Tensor

	for i := range g.Nodes {
		n := &g.Nodes[i]
		if n.OpType != "Mul" {
			continue
		}

		for _, in := range n.Inputs {
			t := g.Initializer(in)
			if t == nil || t.DataType != onnx.DataTypeFloat || !slices.Equal(t.Dims, []int64{cqtOutputBins, 1, 1}) {
				continue
			}

			if found != nil {
				return nil, fmt.Errorf("extractweights: sqrt lengths: more than one Mul by a [%d,1,1] constant", cqtOutputBins)
			}

			found = t
		}
	}

	if found == nil {
		return nil, fmt.Errorf("extractweights: sqrt lengths: no Mul by a [%d,1,1] float constant", cqtOutputBins)
	}

	return found, nil
}

// cqtBatchNorm resolves the folded inference BatchNorm on the CQT
// (port-spec §3.3). The harmonic stack's input 1 is the unshifted h=1 channel,
// which is directly the BatchNorm output Add(Mul(x, scale), offset); the other
// seven inputs must be shifted (Slice/Pad) copies of the same Add.
func cqtBatchNorm(g *onnx.Graph, stack *onnx.Node) (scale, offset *onnx.Tensor, err error) {
	if len(stack.Inputs) != 8 {
		return nil, nil, fmt.Errorf("extractweights: harmonic stack has %d inputs, want 8", len(stack.Inputs))
	}

	if a := stack.Attr("axis"); a == nil || (a.I != -1 && a.I != 3) {
		return nil, nil, fmt.Errorf("extractweights: harmonic stack does not concatenate on the channel axis (-1)")
	}

	add := g.Producer(stack.Inputs[1])
	if add == nil || add.OpType != "Add" {
		return nil, nil, fmt.Errorf("extractweights: harmonic stack input 1 is not produced by an Add")
	}

	for i, in := range stack.Inputs {
		if i == 1 {
			continue
		}

		n, err := behind(g, in, "Add", "")
		if err != nil {
			return nil, nil, fmt.Errorf("extractweights: harmonic stack input %d: %w", i, err)
		}

		if n != add {
			return nil, nil, fmt.Errorf("extractweights: harmonic stack input %d is a copy of %q, want %q", i, n.Name, add.Name)
		}
	}

	x, offset, err := affineOperands(g, add)
	if err != nil {
		return nil, nil, fmt.Errorf("extractweights: CQT BatchNorm offset: %w", err)
	}

	mul := g.Producer(x)
	if mul == nil || mul.OpType != "Mul" {
		return nil, nil, fmt.Errorf("extractweights: CQT BatchNorm: Add %q does not read a Mul", add.Name)
	}

	if _, scale, err = affineOperands(g, mul); err != nil {
		return nil, nil, fmt.Errorf("extractweights: CQT BatchNorm scale: %w", err)
	}

	return scale, offset, nil
}

// affineOperands splits a binary Mul or Add into its data input and its
// one-element float constant.
func affineOperands(g *onnx.Graph, n *onnx.Node) (string, *onnx.Tensor, error) {
	if len(n.Inputs) != 2 {
		return "", nil, fmt.Errorf("%s %q has %d inputs, want 2", n.OpType, n.Name, len(n.Inputs))
	}

	data := dataInputs(g, n)
	if len(data) != 1 {
		return "", nil, fmt.Errorf("%s %q has %d non-constant inputs, want 1", n.OpType, n.Name, len(data))
	}

	for _, in := range n.Inputs {
		if in != data[0] {
			t, err := floatInitializer(g, in, []int64{1})
			if err != nil {
				return "", nil, err
			}

			return data[0], t, nil
		}
	}

	return "", nil, fmt.Errorf("%s %q has no constant input", n.OpType, n.Name)
}
