package main

import (
	"fmt"
	"slices"

	"github.com/cwbudde/algo-transcribe/internal/onnx"
)

// Graph outputs as pinned by docs/port-spec.md §4.
const (
	outContour = "StatefulPartitionedCall:0"
	outNote    = "StatefulPartitionedCall:1"
	outOnset   = "StatefulPartitionedCall:2"
)

// Activation op types that may sit between a Conv and its consumer.
const (
	opSigmoid = "Sigmoid"
	opRelu    = "Relu"
	opConv    = "Conv"
	opConcat  = "Concat"
)

// convSpec is the expected shape of one model Conv in ONNX terms.
type convSpec struct {
	name    string
	ident   string  // Go identifier stem for weights.go
	kernel  []int64 // OIHW
	strides []int64 // [t, f]
	pads    []int64 // [t_begin, f_begin, t_end, f_end]
}

// The six model layers of docs/port-spec.md §4, in manifest order.
var (
	specConv2d1 = convSpec{
		name: "conv2d_1", ident: "Conv2d1",
		kernel: []int64{8, 8, 3, 39}, strides: []int64{1, 1}, pads: []int64{1, 19, 1, 19},
	}
	specContours = convSpec{
		name: "contours-reduced", ident: "ContoursReduced",
		kernel: []int64{1, 8, 5, 5}, strides: []int64{1, 1}, pads: []int64{2, 2, 2, 2},
	}
	specConv2d2 = convSpec{
		name: "conv2d_2", ident: "Conv2d2",
		kernel: []int64{32, 1, 7, 7}, strides: []int64{1, 3}, pads: []int64{3, 2, 3, 2},
	}
	specConv2d3 = convSpec{
		name: "conv2d_3", ident: "Conv2d3",
		kernel: []int64{1, 32, 7, 3}, strides: []int64{1, 1}, pads: []int64{3, 1, 3, 1},
	}
	specConv2d4 = convSpec{
		name: "conv2d_4", ident: "Conv2d4",
		kernel: []int64{32, 8, 5, 5}, strides: []int64{1, 3}, pads: []int64{2, 1, 2, 1},
	}
	specConv2d5 = convSpec{
		name: "conv2d_5", ident: "Conv2d5",
		kernel: []int64{1, 33, 3, 3}, strides: []int64{1, 1}, pads: []int64{1, 1, 1, 1},
	}
)

// Front-end constant shapes (docs/port-spec.md §3.1).
const (
	cqtBins       = 36  // kernels per octave
	cqtFFT        = 256 // kernel length
	cqtOctaves    = 9
	cqtOutputBins = 309
	lowpassConvs  = cqtOctaves - 1
)

// passThrough lists the op types walkBack steps over. Each of them has exactly
// one non-initializer input in the basic-pitch graph.
var passThrough = []string{
	"Reshape", "Transpose", "Unsqueeze", "Squeeze", "Slice", "Pad", "Identity",
	opSigmoid, opRelu,
}

// modelConvs holds the Conv nodes of the six model layers and the harmonic
// stack Concat they read.
type modelConvs struct {
	conv2d1, contours, conv2d2, conv2d3, conv2d4, conv2d5 *onnx.Node
	harmonicStack                                         *onnx.Node
}

// mapModel identifies every exported tensor by following node inputs and
// returns them in manifest order, kernels converted to HWIO.
func mapModel(m *onnx.Model) ([]namedTensor, error) {
	g := &m.Graph

	convs, err := findModelConvs(g)
	if err != nil {
		return nil, err
	}

	layers := []struct {
		node *onnx.Node
		spec convSpec
	}{
		{convs.conv2d1, specConv2d1},
		{convs.contours, specContours},
		{convs.conv2d2, specConv2d2},
		{convs.conv2d3, specConv2d3},
		{convs.conv2d4, specConv2d4},
		{convs.conv2d5, specConv2d5},
	}

	var out []namedTensor

	for _, l := range layers {
		kernel, bias, err := layerTensors(g, l.node, l.spec)
		if err != nil {
			return nil, err
		}

		out = append(out, kernel, bias)
	}

	front, err := frontEndTensors(g, convs)
	if err != nil {
		return nil, err
	}

	return append(out, front...), nil
}

// findModelConvs walks back from the three graph outputs to the six model
// Conv nodes and checks the wiring of docs/port-spec.md §4.
func findModelConvs(g *onnx.Graph) (modelConvs, error) {
	var c modelConvs

	for _, name := range []string{outContour, outNote, outOnset} {
		if !slices.ContainsFunc(g.Outputs, func(v onnx.ValueInfo) bool { return v.Name == name }) {
			return c, fmt.Errorf("extractweights: graph has no output %q", name)
		}
	}

	var err error

	steps := []struct {
		what  string
		dst   **onnx.Node
		value func() string
		op    string
		act   string
	}{
		{"contours-reduced (contour output)", &c.contours, func() string { return outContour }, opConv, opSigmoid},
		{"conv2d_3 (note output)", &c.conv2d3, func() string { return outNote }, opConv, opSigmoid},
		{"conv2d_5 (onset output)", &c.conv2d5, func() string { return outOnset }, opConv, opSigmoid},
		{"conv2d_1 (contours-reduced input)", &c.conv2d1, func() string { return firstInput(c.contours) }, opConv, opRelu},
		{"conv2d_2 (conv2d_3 input)", &c.conv2d2, func() string { return firstInput(c.conv2d3) }, opConv, opRelu},
		{"harmonic stack (conv2d_1 input)", &c.harmonicStack, func() string { return firstInput(c.conv2d1) }, opConcat, ""},
	}

	for _, s := range steps {
		if *s.dst, err = behind(g, s.value(), s.op, s.act); err != nil {
			return c, fmt.Errorf("extractweights: %s: %w", s.what, err)
		}
	}

	// conv2d_2 reads the contour output, so walking back from its input must
	// cross a Sigmoid and reach contours-reduced.
	contours, err := behind(g, firstInput(c.conv2d2), opConv, opSigmoid)
	if err != nil {
		return c, fmt.Errorf("extractweights: conv2d_2 input: %w", err)
	}

	if contours != c.contours {
		return c, fmt.Errorf("extractweights: conv2d_2 input: reaches Conv %q, want contours-reduced %q",
			contours.Name, c.contours.Name)
	}

	if err := findOnsetInputs(g, &c); err != nil {
		return c, err
	}

	// conv2d_4 reads the harmonic stack, exactly like conv2d_1.
	if firstInput(c.conv2d4) != firstInput(c.conv2d1) {
		return c, fmt.Errorf("extractweights: conv2d_4 reads %q, conv2d_1 reads %q; want the same harmonic stack",
			firstInput(c.conv2d4), firstInput(c.conv2d1))
	}

	nodes := []*onnx.Node{c.conv2d1, c.contours, c.conv2d2, c.conv2d3, c.conv2d4, c.conv2d5}
	for i, n := range nodes {
		if slices.Contains(nodes[:i], n) {
			return c, fmt.Errorf("extractweights: Conv %q maps to two model layers", n.Name)
		}
	}

	return c, nil
}

// findOnsetInputs resolves the Concat feeding conv2d_5: channel 0 is the note
// (post-sigmoid conv2d_3) and channels 1..32 are conv2d_4 after ReLU.
func findOnsetInputs(g *onnx.Graph, c *modelConvs) error {
	concat, err := behind(g, firstInput(c.conv2d5), opConcat, "")
	if err != nil {
		return fmt.Errorf("extractweights: conv2d_5 input: %w", err)
	}

	if len(concat.Inputs) != 2 {
		return fmt.Errorf("extractweights: onset concat has %d inputs, want 2", len(concat.Inputs))
	}

	if a := concat.Attr("axis"); a == nil || a.I != 1 {
		return fmt.Errorf("extractweights: onset concat: want axis 1 (NCHW channels)")
	}

	note, err := behind(g, concat.Inputs[0], opConv, opSigmoid)
	if err != nil {
		return fmt.Errorf("extractweights: onset concat input 0: %w", err)
	}

	if note != c.conv2d3 {
		return fmt.Errorf("extractweights: onset concat input 0 reaches Conv %q, want conv2d_3 %q",
			note.Name, c.conv2d3.Name)
	}

	if c.conv2d4, err = behind(g, concat.Inputs[1], opConv, opRelu); err != nil {
		return fmt.Errorf("extractweights: conv2d_4 (onset concat input 1): %w", err)
	}

	return nil
}

// behind walks back from value through pass-through ops to the first node
// that is not one, and requires it to have op type want. If act is not empty,
// the walk must cross exactly one activation of that type, directly after the
// node; if act is empty it must cross none.
func behind(g *onnx.Graph, value, want, act string) (*onnx.Node, error) {
	n, crossed, err := walkBack(g, value)
	if err != nil {
		return nil, err
	}

	if n.OpType != want {
		return nil, fmt.Errorf("walk from %q stops at %s %q, want %s", value, n.OpType, n.Name, want)
	}

	acts := 0

	for _, op := range crossed {
		if op == opSigmoid || op == opRelu {
			acts++
		}
	}

	switch {
	case act == "" && acts != 0:
		return nil, fmt.Errorf("walk from %q crosses activations %v, want none", value, crossed)
	case act != "" && (acts != 1 || crossed[len(crossed)-1] != act):
		return nil, fmt.Errorf("walk from %q crosses %v, want exactly one %s directly after the %s",
			value, crossed, act, want)
	}

	return n, nil
}

// walkBack steps from value to its producer while that producer is a
// pass-through op, and returns the first other producer together with the op
// types crossed (nearest to value first).
func walkBack(g *onnx.Graph, value string) (*onnx.Node, []string, error) {
	var crossed []string

	for range len(g.Nodes) + 1 {
		n := g.Producer(value)
		if n == nil {
			return nil, nil, fmt.Errorf("no node produces %q", value)
		}

		if !slices.Contains(passThrough, n.OpType) {
			return n, crossed, nil
		}

		data := dataInputs(g, n)
		if len(data) != 1 {
			return nil, nil, fmt.Errorf("%s %q has %d non-constant inputs, want 1", n.OpType, n.Name, len(data))
		}

		crossed = append(crossed, n.OpType)
		value = data[0]
	}

	return nil, nil, fmt.Errorf("walk from %q does not terminate", value)
}

// firstInput returns the data input of a node, or "" if it has no inputs.
func firstInput(n *onnx.Node) string {
	if len(n.Inputs) == 0 {
		return ""
	}

	return n.Inputs[0]
}

// dataInputs returns the inputs of n that are neither initializers nor
// omitted optional inputs.
func dataInputs(g *onnx.Graph, n *onnx.Node) []string {
	var data []string

	for _, in := range n.Inputs {
		if in != "" && g.Initializer(in) == nil {
			data = append(data, in)
		}
	}

	return data
}

// layerTensors checks a model Conv against its spec and returns its kernel
// (HWIO) and bias.
func layerTensors(g *onnx.Graph, n *onnx.Node, s convSpec) (namedTensor, namedTensor, error) {
	kernel, bias, err := convWeights(g, n, s.kernel, s.strides, s.pads)
	if err != nil {
		return namedTensor{}, namedTensor{}, fmt.Errorf("extractweights: %s: %w", s.name, err)
	}

	kd, err := kernel.Float32s()
	if err != nil {
		return namedTensor{}, namedTensor{}, fmt.Errorf("extractweights: %s kernel: %w", s.name, err)
	}

	o, i, h, w := int(s.kernel[0]), int(s.kernel[1]), int(s.kernel[2]), int(s.kernel[3])

	hwio, err := oihwToHWIO(kd, o, i, h, w)
	if err != nil {
		return namedTensor{}, namedTensor{}, fmt.Errorf("extractweights: %s: %w", s.name, err)
	}

	bd, err := bias.Float32s()
	if err != nil {
		return namedTensor{}, namedTensor{}, fmt.Errorf("extractweights: %s bias: %w", s.name, err)
	}

	return namedTensor{name: s.name + "/kernel", ident: s.ident + "Kernel", shape: []int{h, w, i, o}, data: hwio},
		namedTensor{name: s.name + "/bias", ident: s.ident + "Bias", shape: []int{o}, data: bd},
		nil
}

// convWeights checks the inputs and attributes of a Conv node and returns its
// kernel and bias initializers. kernel is the expected OIHW shape, strides the
// expected [h, w] strides, and pads the expected pads (nil: absent or zero).
func convWeights(g *onnx.Graph, n *onnx.Node, kernel, strides, pads []int64) (*onnx.Tensor, *onnx.Tensor, error) {
	if n.OpType != opConv || n.Domain != "" {
		return nil, nil, fmt.Errorf("node %q is %s in domain %q, want Conv", n.Name, n.OpType, n.Domain)
	}

	if len(n.Inputs) != 3 {
		return nil, nil, fmt.Errorf("conv %q has %d inputs, want 3", n.Name, len(n.Inputs))
	}

	if g.Initializer(n.Inputs[0]) != nil {
		return nil, nil, fmt.Errorf("conv %q reads a constant as data", n.Name)
	}

	k, err := floatInitializer(g, n.Inputs[1], kernel)
	if err != nil {
		return nil, nil, fmt.Errorf("conv %q kernel: %w", n.Name, err)
	}

	b, err := floatInitializer(g, n.Inputs[2], kernel[:1])
	if err != nil {
		return nil, nil, fmt.Errorf("conv %q bias: %w", n.Name, err)
	}

	if err := checkConvAttrs(n, kernel[2:], strides, pads); err != nil {
		return nil, nil, fmt.Errorf("conv %q: %w", n.Name, err)
	}

	return k, b, nil
}

// checkConvAttrs validates every attribute of a 2-D Conv. Unknown attributes
// are rejected so that nothing the port would ignore slips through.
func checkConvAttrs(n *onnx.Node, kernelHW, strides, pads []int64) error {
	seen := map[string]bool{}

	for _, a := range n.Attributes {
		seen[a.Name] = true

		switch a.Name {
		case "strides":
			if !slices.Equal(a.Ints, strides) {
				return fmt.Errorf("strides %v, want %v", a.Ints, strides)
			}
		case "pads":
			if pads == nil && slices.ContainsFunc(a.Ints, func(p int64) bool { return p != 0 }) ||
				pads != nil && !slices.Equal(a.Ints, pads) {
				return fmt.Errorf("pads %v, want %v", a.Ints, pads)
			}
		case "dilations":
			if !slices.Equal(a.Ints, []int64{1, 1}) {
				return fmt.Errorf("dilations %v, want [1 1]", a.Ints)
			}
		case "group":
			if a.I != 1 {
				return fmt.Errorf("group %d, want 1", a.I)
			}
		case "auto_pad":
			if s := string(a.S); s != "" && s != "NOTSET" {
				return fmt.Errorf("auto_pad %q, want NOTSET", s)
			}
		case "kernel_shape":
			if !slices.Equal(a.Ints, kernelHW) {
				return fmt.Errorf("kernel_shape %v, want %v", a.Ints, kernelHW)
			}
		default:
			return fmt.Errorf("unexpected attribute %q", a.Name)
		}
	}

	if !seen["strides"] {
		return fmt.Errorf("no strides attribute, want %v", strides)
	}

	if pads != nil && !seen["pads"] {
		return fmt.Errorf("no pads attribute, want %v", pads)
	}

	return nil
}

// floatInitializer returns the float32 initializer named name and checks its
// dims.
func floatInitializer(g *onnx.Graph, name string, dims []int64) (*onnx.Tensor, error) {
	t := g.Initializer(name)
	if t == nil {
		return nil, fmt.Errorf("%q is not an initializer", name)
	}

	if t.DataType != onnx.DataTypeFloat {
		return nil, fmt.Errorf("initializer %q has data type %d, want float", name, t.DataType)
	}

	if !slices.Equal(t.Dims, dims) {
		return nil, fmt.Errorf("initializer %q has dims %v, want %v", name, t.Dims, dims)
	}

	return t, nil
}
