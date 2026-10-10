package onnx

import "slices"

// Initializer returns the initializer named name, or nil if there is none.
// The pointer refers into g.Initializers.
func (g *Graph) Initializer(name string) *Tensor {
	for i := range g.Initializers {
		if g.Initializers[i].Name == name {
			return &g.Initializers[i]
		}
	}

	return nil
}

// Producer returns the node that lists value among its outputs, or nil if no
// node produces it (for example a graph input or an initializer). The pointer
// refers into g.Nodes.
func (g *Graph) Producer(value string) *Node {
	for i := range g.Nodes {
		if slices.Contains(g.Nodes[i].Outputs, value) {
			return &g.Nodes[i]
		}
	}

	return nil
}

// Consumers returns, in graph order, the nodes that list value among their
// inputs. The pointers refer into g.Nodes; a node consuming value twice is
// listed once.
func (g *Graph) Consumers(value string) []*Node {
	var out []*Node

	for i := range g.Nodes {
		if slices.Contains(g.Nodes[i].Inputs, value) {
			out = append(out, &g.Nodes[i])
		}
	}

	return out
}

// Attr returns the attribute named name, or nil if the node has none. The
// pointer refers into n.Attributes.
func (n *Node) Attr(name string) *Attribute {
	for i := range n.Attributes {
		if n.Attributes[i].Name == name {
			return &n.Attributes[i]
		}
	}

	return nil
}
