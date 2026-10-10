package onnx

import (
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

const nmpPath = "../../testdata/nmp.onnx"

// loadNMP parses the committed basic-pitch model.
func loadNMP(t *testing.T) *Model {
	t.Helper()

	data, err := os.ReadFile(nmpPath)
	if err != nil {
		t.Fatalf("read model: %v", err)
	}

	m, err := Parse(data)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	return m
}

// TestParseNMP checks the reader against ground truth taken from the
// reference Python onnx package for basic-pitch v0.4.0's nmp.onnx.
func TestParseNMP(t *testing.T) {
	t.Parallel()

	m := loadNMP(t)
	g := &m.Graph

	if m.IRVersion != 8 {
		t.Errorf("IRVersion = %d, want 8", m.IRVersion)
	}

	if m.ProducerName != "tf2onnx" || m.ProducerVersion != "1.15.1 37820d" {
		t.Errorf("producer = %q %q", m.ProducerName, m.ProducerVersion)
	}

	wantOpsets := []OpsetID{{"", 15}, {"ai.onnx.ml", 2}}
	if !reflect.DeepEqual(m.Opsets, wantOpsets) {
		t.Errorf("Opsets = %v, want %v", m.Opsets, wantOpsets)
	}

	if len(g.Initializers) != 102 {
		t.Errorf("%d initializers, want 102", len(g.Initializers))
	}

	if len(g.Nodes) != 248 {
		t.Errorf("%d nodes, want 248", len(g.Nodes))
	}

	ops := map[string]int{}
	for i := range g.Nodes {
		ops[g.Nodes[i].OpType]++
	}

	if ops["Conv"] != 32 || ops["Neg"] != 9 {
		t.Errorf("Conv = %d, Neg = %d, want 32 and 9", ops["Conv"], ops["Neg"])
	}

	wantInputs := []ValueInfo{{
		Name: "serving_default_input_2:0", ElemType: DataTypeFloat,
		Dims: []Dim{{Param: g.Inputs[0].Dims[0].Param}, {Value: 43844}, {Value: 1}},
	}}
	if !reflect.DeepEqual(g.Inputs, wantInputs) || g.Inputs[0].Dims[0].Param == "" {
		t.Errorf("Inputs = %+v, want %+v with a symbolic batch dim", g.Inputs, wantInputs)
	}

	wantOutputs := []struct {
		name string
		last int64
	}{
		{"StatefulPartitionedCall:2", 88},
		{"StatefulPartitionedCall:1", 88},
		{"StatefulPartitionedCall:0", 264},
	}
	if len(g.Outputs) != len(wantOutputs) {
		t.Fatalf("%d outputs, want %d", len(g.Outputs), len(wantOutputs))
	}

	for i, w := range wantOutputs {
		o := g.Outputs[i]
		if o.Name != w.name || o.ElemType != DataTypeFloat || len(o.Dims) != 3 ||
			o.Dims[1].Value != 172 || o.Dims[2].Value != w.last || o.Dims[0].Param == "" {
			t.Errorf("output %d = %+v, want %s [?,172,%d]", i, o, w.name, w.last)
		}
	}
}

func TestParseNMPInitializers(t *testing.T) {
	t.Parallel()

	g := &loadNMP(t).Graph

	w := g.Initializer("const_fold_opt__727")
	if w == nil {
		t.Fatal("const_fold_opt__727 missing")
	}

	if w.DataType != DataTypeFloat || !reflect.DeepEqual(w.Dims, []int64{8, 8, 3, 39}) || len(w.RawData) == 0 {
		t.Fatalf("const_fold_opt__727: type %d dims %v raw %d bytes", w.DataType, w.Dims, len(w.RawData))
	}

	vals, err := w.Float32s()
	if err != nil {
		t.Fatal(err)
	}

	if len(vals) != 8*8*3*39 {
		t.Fatalf("%d values", len(vals))
	}

	for i, want := range []float32{0.02223546, 0.01556065, -0.01682326, 0.00577071} {
		if math.Abs(float64(vals[i]-want)) > 1e-7 {
			t.Errorf("value %d = %g, want %g", i, vals[i], want)
		}
	}

	for _, tc := range []struct {
		name string
		want []int64
	}{
		{"const_starts__320", []int64{0, -309, 0, 0}},
		{"pad_const__82", nil},
	} {
		tensor := g.Initializer(tc.name)
		if tensor == nil || tensor.DataType != DataTypeInt64 {
			t.Fatalf("%s missing or not INT64: %+v", tc.name, tensor)
		}

		got, err := tensor.Int64s()
		if err != nil {
			t.Fatal(err)
		}

		if tc.want != nil && !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s = %v, want %v", tc.name, got, tc.want)
		}

		if tc.name == "pad_const__82" &&
			(len(got) != 6 || !reflect.DeepEqual(got[:4], []int64{0, 0, 128, 0})) {
			t.Errorf("%s = %v, want 6 values beginning [0 0 128 0]", tc.name, got)
		}
	}
}

func TestParseNMPAttributes(t *testing.T) {
	t.Parallel()

	g := &loadNMP(t).Graph

	// tf2onnx names fused nodes by joining the TF op names with ";", e.g.
	// "model_1/normalized_log_1/Min;model_1/normalized_log_1/Min". Match the
	// last segment.
	node := func(name string) *Node {
		t.Helper()

		for i := range g.Nodes {
			n := g.Nodes[i].Name
			if n == name || strings.HasSuffix(n, ";"+name) {
				return &g.Nodes[i]
			}
		}

		t.Fatalf("node %s missing", name)

		return nil
	}

	conv := node("model_1/conv2d_4/Conv2D1")
	if conv.OpType != "Conv" {
		t.Errorf("op %q, want Conv", conv.OpType)
	}

	for _, tc := range []struct {
		name string
		want []int64
	}{
		{"strides", []int64{1, 3}},
		{"kernel_shape", []int64{5, 5}},
		{"pads", []int64{2, 1, 2, 1}},
		{"dilations", []int64{1, 1}},
	} {
		a := conv.Attr(tc.name)
		if a == nil || a.Type != AttributeInts || !reflect.DeepEqual(a.Ints, tc.want) {
			t.Errorf("Conv %s = %+v, want INTS %v", tc.name, a, tc.want)
		}
	}

	if a := conv.Attr("group"); a == nil || a.Type != AttributeInt || a.I != 1 {
		t.Errorf("Conv group = %+v, want INT 1", a)
	}

	pad := node("model_1/cq_t2010v2_1/reflection_pad1d/MirrorPad")
	if a := pad.Attr("mode"); a == nil || a.Type != AttributeString || string(a.S) != "reflect" {
		t.Errorf("Pad mode = %+v, want STRING reflect", a)
	}

	minNode := node("model_1/normalized_log_1/Min")
	if a := minNode.Attr("axes"); a == nil || a.Type != AttributeInts || !reflect.DeepEqual(a.Ints, []int64{1, 2}) {
		t.Errorf("ReduceMin axes = %+v, want INTS [1 2]", a)
	}
}

// TestParseNMPTopology checks that every node input is produced by a node, an
// initializer or a graph input, and that Producer/Consumers agree.
func TestParseNMPTopology(t *testing.T) {
	t.Parallel()

	g := &loadNMP(t).Graph

	inputs := map[string]bool{}
	for _, in := range g.Inputs {
		inputs[in.Name] = true
	}

	for i := range g.Nodes {
		n := &g.Nodes[i]
		for _, in := range n.Inputs {
			if in == "" || inputs[in] || g.Initializer(in) != nil {
				continue
			}

			p := g.Producer(in)
			if p == nil {
				t.Errorf("node %s: input %s has no producer", n.Name, in)

				continue
			}

			found := false

			for _, c := range g.Consumers(in) {
				found = found || c == n
			}

			if !found {
				t.Errorf("node %s missing from Consumers(%s)", n.Name, in)
			}
		}
	}

	for _, out := range g.Outputs {
		if g.Producer(out.Name) == nil {
			t.Errorf("graph output %s has no producer", out.Name)
		}
	}
}
