package onnx

import (
	"errors"
	"fmt"
)

// Errors for ONNX features the reader deliberately does not support.
var (
	// ErrExternalData reports a tensor whose data lives outside the model file.
	ErrExternalData = errors.New("external data not supported")
	// ErrSegment reports a segmented tensor (TensorProto.segment).
	ErrSegment = errors.New("tensor segments not supported")
)

// TensorProto.DataLocation value for data stored outside the model file.
const dataLocationExternal = 1

// Parse decodes a serialized ONNX ModelProto.
//
// It is a hand-written walk over the protobuf wire format and never panics on
// malformed input; truncated data, varint overflow, group wire types, wrong
// wire types for known fields, external tensor data and tensor segments are
// reported as errors. Fields the reader does not model are skipped, as are
// subgraph attributes (AttributeProto.g and .graphs). Repeated scalar fields
// accept packed and unpacked encodings, and a repeated occurrence of an
// embedded message is merged into the earlier one, as protobuf prescribes.
//
// The returned Model does not alias data: strings and byte slices (raw_data,
// string attributes) are copies.
func Parse(data []byte) (*Model, error) {
	m := &Model{}
	if err := parseModel(data, m); err != nil {
		return nil, fmt.Errorf("onnx: parse model: %w", err)
	}

	return m, nil
}

// parseModel decodes a ModelProto into m.
func parseModel(b []byte, m *Model) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var err error

		switch num {
		case 1: // ir_version
			m.IRVersion, err = d.readInt64(wt)
		case 2: // producer_name
			m.ProducerName, err = d.readString(wt)
		case 3: // producer_version
			m.ProducerVersion, err = d.readString(wt)
		case 7: // graph
			err = parseEmbedded(d, wt, "graph", func(b []byte) error { return parseGraph(b, &m.Graph) })
		case 8: // opset_import
			var o OpsetID

			err = parseEmbedded(d, wt, fmt.Sprintf("opset_import %d", len(m.Opsets)),
				func(b []byte) error { return parseOpset(b, &o) })
			m.Opsets = append(m.Opsets, o)
		default:
			return false, nil
		}

		return true, err
	})
}

// parseEmbedded reads a length-delimited embedded message and hands its bytes
// to parse, prefixing errors with what.
func parseEmbedded(d *decoder, wt int, what string, parse func([]byte) error) error {
	b, err := d.readMessage(wt)
	if err == nil {
		err = parse(b)
	}

	if err != nil {
		return fmt.Errorf("%s: %w", what, err)
	}

	return nil
}

// parseOpset decodes an OperatorSetIdProto.
func parseOpset(b []byte, o *OpsetID) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var err error

		switch num {
		case 1: // domain
			o.Domain, err = d.readString(wt)
		case 2: // version
			o.Version, err = d.readInt64(wt)
		default:
			return false, nil
		}

		return true, err
	})
}

// parseGraph decodes a GraphProto into g.
func parseGraph(b []byte, g *Graph) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var err error

		switch num {
		case 1: // node
			var n Node

			err = parseEmbedded(d, wt, fmt.Sprintf("node %d", len(g.Nodes)),
				func(b []byte) error { return parseNode(b, &n) })
			g.Nodes = append(g.Nodes, n)
		case 2: // name
			g.Name, err = d.readString(wt)
		case 5: // initializer
			var t Tensor

			err = parseEmbedded(d, wt, fmt.Sprintf("initializer %d", len(g.Initializers)),
				func(b []byte) error { return parseTensor(b, &t) })
			g.Initializers = append(g.Initializers, t)
		case 11: // input
			var v ValueInfo

			err = parseEmbedded(d, wt, fmt.Sprintf("input %d", len(g.Inputs)),
				func(b []byte) error { return parseValueInfo(b, &v) })
			g.Inputs = append(g.Inputs, v)
		case 12: // output
			var v ValueInfo

			err = parseEmbedded(d, wt, fmt.Sprintf("output %d", len(g.Outputs)),
				func(b []byte) error { return parseValueInfo(b, &v) })
			g.Outputs = append(g.Outputs, v)
		default:
			return false, nil
		}

		return true, err
	})
}

// parseNode decodes a NodeProto into n.
func parseNode(b []byte, n *Node) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var (
			s   string
			err error
		)

		switch num {
		case 1: // input
			s, err = d.readString(wt)
			n.Inputs = append(n.Inputs, s)
		case 2: // output
			s, err = d.readString(wt)
			n.Outputs = append(n.Outputs, s)
		case 3: // name
			n.Name, err = d.readString(wt)
		case 4: // op_type
			n.OpType, err = d.readString(wt)
		case 5: // attribute
			var a Attribute

			err = parseEmbedded(d, wt, fmt.Sprintf("attribute %d", len(n.Attributes)),
				func(b []byte) error { return parseAttribute(b, &a) })
			n.Attributes = append(n.Attributes, a)
		case 7: // domain
			n.Domain, err = d.readString(wt)
		default:
			return false, nil
		}

		return true, err
	})
}

// parseAttribute decodes an AttributeProto into a. Subgraphs (g, graphs) and
// tensor lists (tensors) are skipped.
func parseAttribute(b []byte, a *Attribute) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var err error

		switch num {
		case 1: // name
			a.Name, err = d.readString(wt)
		case 2: // f
			a.F, err = d.readFloat32(wt)
		case 3: // i
			a.I, err = d.readInt64(wt)
		case 4: // s
			a.S, err = d.readBytes(wt)
		case 5: // t
			if a.T == nil {
				a.T = &Tensor{}
			}

			err = parseEmbedded(d, wt, "t", func(b []byte) error { return parseTensor(b, a.T) })
		case 7: // floats
			a.Floats, err = d.appendFloat32s(a.Floats, wt)
		case 8: // ints
			a.Ints, err = d.appendInt64s(a.Ints, wt)
		case 9: // strings
			var s []byte

			s, err = d.readBytes(wt)
			a.Strings = append(a.Strings, s)
		case 20: // type
			var t int32

			t, err = d.readInt32(wt)
			a.Type = AttributeType(t)
		default:
			return false, nil
		}

		return true, err
	})
}

// parseTensor decodes a TensorProto into t.
func parseTensor(b []byte, t *Tensor) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var err error

		switch num {
		case 1: // dims
			t.Dims, err = d.appendInt64s(t.Dims, wt)
		case 2: // data_type
			var dt int32

			dt, err = d.readInt32(wt)
			t.DataType = DataType(dt)
		case 3: // segment
			err = ErrSegment
		case 4: // float_data
			t.FloatData, err = d.appendFloat32s(t.FloatData, wt)
		case 5: // int32_data
			t.Int32Data, err = d.appendInt32s(t.Int32Data, wt)
		case 7: // int64_data
			t.Int64Data, err = d.appendInt64s(t.Int64Data, wt)
		case 8: // name
			t.Name, err = d.readString(wt)
		case 9: // raw_data
			t.RawData, err = d.readBytes(wt)
		case 10: // double_data
			t.DoubleData, err = d.appendFloat64s(t.DoubleData, wt)
		case 13: // external_data
			err = ErrExternalData
		case 14: // data_location
			var loc int32

			loc, err = d.readInt32(wt)
			if err == nil && loc == dataLocationExternal {
				err = ErrExternalData
			}
		default:
			return false, nil
		}

		return true, err
	})
}

// parseValueInfo decodes a ValueInfoProto into v.
func parseValueInfo(b []byte, v *ValueInfo) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var err error

		switch num {
		case 1: // name
			v.Name, err = d.readString(wt)
		case 2: // type
			err = parseEmbedded(d, wt, "type", func(b []byte) error { return parseTypeProto(b, v) })
		default:
			return false, nil
		}

		return true, err
	})
}

// parseTypeProto decodes the tensor_type branch of a TypeProto into v; other
// type kinds (sequence, map, optional, sparse) are skipped.
func parseTypeProto(b []byte, v *ValueInfo) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		if num != 1 { // tensor_type
			return false, nil
		}

		return true, parseEmbedded(d, wt, "tensor_type", func(b []byte) error { return parseTensorType(b, v) })
	})
}

// parseTensorType decodes a TypeProto.Tensor into v.
func parseTensorType(b []byte, v *ValueInfo) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var err error

		switch num {
		case 1: // elem_type
			var et int32

			et, err = d.readInt32(wt)
			v.ElemType = DataType(et)
		case 2: // shape
			err = parseEmbedded(d, wt, "shape", func(b []byte) error { return parseShape(b, v) })
		default:
			return false, nil
		}

		return true, err
	})
}

// parseShape decodes a TensorShapeProto, appending its dimensions to v.Dims.
func parseShape(b []byte, v *ValueInfo) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		if num != 1 { // dim
			return false, nil
		}

		var dim Dim

		err := parseEmbedded(d, wt, fmt.Sprintf("dim %d", len(v.Dims)),
			func(b []byte) error { return parseDim(b, &dim) })
		v.Dims = append(v.Dims, dim)

		return true, err
	})
}

// parseDim decodes a TensorShapeProto.Dimension. dim_value and dim_param form
// a oneof, so setting one clears the other.
func parseDim(b []byte, dim *Dim) error {
	return walk(b, func(d *decoder, num, wt int) (bool, error) {
		var err error

		switch num {
		case 1: // dim_value
			dim.Value, err = d.readInt64(wt)
			dim.Param = ""
		case 2: // dim_param
			dim.Param, err = d.readString(wt)
			dim.Value = 0
		default:
			return false, nil
		}

		return true, err
	})
}
