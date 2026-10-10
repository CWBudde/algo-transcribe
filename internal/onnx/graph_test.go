package onnx

import "testing"

// helperGraph is X -> conv0 -> Y -> neg0 -> Z, and add0(Y, Z, Y) -> out, with
// initializer W feeding conv0. It is the graph of buildModel.
func helperGraph(t *testing.T) *Graph {
	t.Helper()

	m, err := Parse(buildModel(false))
	if err != nil {
		t.Fatal(err)
	}

	return &m.Graph
}

func TestInitializer(t *testing.T) {
	t.Parallel()

	g := helperGraph(t)

	tests := []struct {
		name string
		want *Tensor
	}{
		{"W", &g.Initializers[0]},
		{"B", &g.Initializers[1]},
		{"X", nil},
		{"", nil},
	}

	for _, tc := range tests {
		if got := g.Initializer(tc.name); got != tc.want {
			t.Errorf("Initializer(%q) = %p, want %p", tc.name, got, tc.want)
		}
	}
}

func TestProducer(t *testing.T) {
	t.Parallel()

	g := helperGraph(t)

	tests := []struct {
		value string
		want  *Node
	}{
		{"Y", &g.Nodes[0]},
		{"Z", &g.Nodes[1]},
		{"out", &g.Nodes[2]},
		{"X", nil}, // graph input
		{"W", nil}, // initializer
		{"missing", nil},
	}

	for _, tc := range tests {
		if got := g.Producer(tc.value); got != tc.want {
			t.Errorf("Producer(%q) = %p, want %p", tc.value, got, tc.want)
		}
	}
}

func TestConsumers(t *testing.T) {
	t.Parallel()

	g := helperGraph(t)

	tests := []struct {
		value string
		want  []*Node
	}{
		{"X", []*Node{&g.Nodes[0]}},
		{"W", []*Node{&g.Nodes[0]}},
		{"Y", []*Node{&g.Nodes[1], &g.Nodes[2]}}, // add0 uses Y twice, listed once
		{"Z", []*Node{&g.Nodes[2]}},
		{"out", nil},
	}

	for _, tc := range tests {
		got := g.Consumers(tc.value)
		if len(got) != len(tc.want) {
			t.Errorf("Consumers(%q) has %d nodes, want %d", tc.value, len(got), len(tc.want))

			continue
		}

		for i := range got {
			if got[i] != tc.want[i] {
				t.Errorf("Consumers(%q)[%d] = %s, want %s", tc.value, i, got[i].Name, tc.want[i].Name)
			}
		}
	}
}

func TestAttr(t *testing.T) {
	t.Parallel()

	g := helperGraph(t)
	conv := &g.Nodes[0]

	for i, name := range []string{"strides", "pads", "group", "auto_pad", "alpha", "scales", "names", "value", "body"} {
		if got := conv.Attr(name); got != &conv.Attributes[i] {
			t.Errorf("Attr(%q) = %p, want %p", name, got, &conv.Attributes[i])
		}
	}

	if got := conv.Attr("missing"); got != nil {
		t.Errorf("Attr(missing) = %+v, want nil", got)
	}

	if got := g.Nodes[1].Attr("strides"); got != nil {
		t.Errorf("Neg has no attributes, got %+v", got)
	}
}
