package onnx_test

import (
	"fmt"
	"os"
	"strings"

	"github.com/cwbudde/algo-transcribe/internal/onnx"
)

// loadModel parses the committed basic-pitch model for the examples.
func loadModel() *onnx.Model {
	data, err := os.ReadFile("../../testdata/nmp.onnx")
	if err != nil {
		panic(err)
	}

	m, err := onnx.Parse(data)
	if err != nil {
		panic(err)
	}

	return m
}

// lastName returns the last ";"-separated segment of a tf2onnx node name.
func lastName(name string) string {
	return name[strings.LastIndex(name, ";")+1:]
}

func ExampleParse() {
	data, err := os.ReadFile("../../testdata/nmp.onnx")
	if err != nil {
		fmt.Println(err)

		return
	}

	m, err := onnx.Parse(data)
	if err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println(m.ProducerName, m.ProducerVersion, "IR", m.IRVersion)
	fmt.Println("opsets:", m.Opsets)
	fmt.Println(len(m.Graph.Nodes), "nodes,", len(m.Graph.Initializers), "initializers")

	in := m.Graph.Inputs[0]
	fmt.Println("input:", in.Name, in.Dims[1].Value, in.Dims[2].Value)

	for _, out := range m.Graph.Outputs {
		fmt.Println("output:", out.Name, out.Dims[1].Value, out.Dims[2].Value)
	}

	// Output:
	// tf2onnx 1.15.1 37820d IR 8
	// opsets: [{ 15} {ai.onnx.ml 2}]
	// 248 nodes, 102 initializers
	// input: serving_default_input_2:0 43844 1
	// output: StatefulPartitionedCall:2 172 88
	// output: StatefulPartitionedCall:1 172 88
	// output: StatefulPartitionedCall:0 172 264
}

func ExampleParse_error() {
	// A length-delimited graph field (7) claiming 16 bytes but carrying 2.
	_, err := onnx.Parse([]byte{0x3a, 0x10, 0x0a, 0x00})
	fmt.Println(err)

	// Output:
	// onnx: parse model: field 7: graph: length exceeds input: need 16 bytes, have 2
}

func ExampleTensor_NumElements() {
	w := onnx.Tensor{Dims: []int64{8, 8, 3, 39}}
	scalar := onnx.Tensor{}

	fmt.Println(w.NumElements(), scalar.NumElements())

	// Output:
	// 7488 1
}

func ExampleTensor_Float32s() {
	w := loadModel().Graph.Initializer("const_fold_opt__727")

	vals, err := w.Float32s()
	if err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println(w.Dims, len(vals))
	fmt.Printf("%.8f\n", vals[:4])

	// Output:
	// [8 8 3 39] 7488
	// [0.02223546 0.01556065 -0.01682326 0.00577071]
}

func ExampleTensor_Int64s() {
	g := &loadModel().Graph

	for _, name := range []string{"const_starts__320", "pad_const__82"} {
		vals, err := g.Initializer(name).Int64s()
		if err != nil {
			fmt.Println(err)

			return
		}

		fmt.Println(name, vals)
	}

	// Output:
	// const_starts__320 [0 -309 0 0]
	// pad_const__82 [0 0 128 0 0 128]
}

func ExampleGraph_Initializer() {
	g := &loadModel().Graph

	t := g.Initializer("const_starts__320")
	fmt.Println(t.Name, t.DataType == onnx.DataTypeInt64, t.Dims)
	fmt.Println(g.Initializer("no such tensor") == nil)

	// Output:
	// const_starts__320 true [4]
	// true
}

func ExampleGraph_Producer() {
	g := &loadModel().Graph

	for _, out := range g.Outputs {
		p := g.Producer(out.Name)
		fmt.Println(out.Name, "<-", p.OpType, lastName(p.Inputs[0]))
	}

	fmt.Println(g.Producer(g.Inputs[0].Name) == nil)

	// Output:
	// StatefulPartitionedCall:2 <- Reshape model_1/conv2d_5/Sigmoid
	// StatefulPartitionedCall:1 <- Reshape model_1/conv2d_3/Sigmoid
	// StatefulPartitionedCall:0 <- Reshape model_1/contours-reduced/Sigmoid
	// true
}

func ExampleGraph_Consumers() {
	g := &loadModel().Graph

	for _, n := range g.Consumers(g.Inputs[0].Name) {
		fmt.Println(n.OpType, lastName(n.Name))
	}

	for _, n := range g.Consumers("const_starts__320") {
		fmt.Println(n.OpType)
	}

	// Output:
	// Reshape model_1/flatten_audio_ch_1/reshape/Reshape
	// Slice
}

func ExampleNode_Attr() {
	g := &loadModel().Graph

	for i := range g.Nodes {
		n := &g.Nodes[i]
		if lastName(n.Name) != "model_1/conv2d_4/Conv2D1" {
			continue
		}

		fmt.Println(n.OpType)

		for _, name := range []string{"kernel_shape", "strides", "pads", "dilations"} {
			fmt.Println(name, n.Attr(name).Ints)
		}

		fmt.Println("group", n.Attr("group").I)
		fmt.Println("missing", n.Attr("auto_pad") == nil)
	}

	// Output:
	// Conv
	// kernel_shape [5 5]
	// strides [1 3]
	// pads [2 1 2 1]
	// dilations [1 1]
	// group 1
	// missing true
}
