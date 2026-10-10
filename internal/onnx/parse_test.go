package onnx

import (
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
)

// buildModel hand-encodes a ModelProto touching every decoded field and every
// skipped field the reader documents. With unk, unknown fields of all four
// skippable wire types are added at every message level.
func buildModel(unk bool) pb {
	dimN := pb{}.str(2, "N").unknown(unk)
	dim4 := pb{}.varint(1, 4).str(3, "denotation").unknown(unk)
	shape := pb{}.msg(1, dimN).msg(1, dim4).unknown(unk)
	tensorType := pb{}.varint(1, int64(DataTypeFloat)).msg(2, shape).unknown(unk)
	inputX := pb{}.str(1, "X").msg(2, pb{}.msg(1, tensorType).str(6, "denotation").unknown(unk)).
		str(3, "doc").unknown(unk)

	outShape := pb{}.msg(1, pb{}.varint(1, -1).unknown(unk)).unknown(unk)
	outType := pb{}.msg(1, pb{}.varint(1, int64(DataTypeInt64)).msg(2, outShape).unknown(unk)).unknown(unk)
	outputOut := pb{}.str(1, "out").msg(2, outType).unknown(unk)

	weights := pb{}.packedVarints(1, 2, 1, 1, 2).
		varint(2, int64(DataTypeFloat)).
		str(8, "W").
		bytes(9, rawLE(1, -2, 0.5, 3)).
		str(12, "doc").
		str(6, "string data is skipped").
		unknown(unk)
	bias := pb{}.varint(1, 2).
		varint(2, int64(DataTypeInt64)).
		str(8, "B").
		packedVarints(7, -1, 5).
		varint(14, 0). // data_location DEFAULT
		unknown(unk)

	attrTensor := pb{}.varint(1, 1).varint(2, int64(DataTypeDouble)).double(10, 2.5).unknown(unk)
	subgraph := pb{}.str(2, "sub").msg(1, pb{}.str(4, "Identity"))

	conv := pb{}.
		str(1, "X").str(1, "W").str(1, "B").
		str(2, "Y").
		str(3, "conv0").
		str(4, "Conv").
		str(6, "node doc").
		msg(5, pb{}.str(1, "strides").packedVarints(8, 1, 3).varint(20, int64(AttributeInts)).unknown(unk)).
		msg(5, pb{}.str(1, "pads").varint(8, -1).varint(8, 2).varint(20, int64(AttributeInts)).
			str(13, "attr doc").str(21, "ref").unknown(unk)).
		msg(5, pb{}.str(1, "group").varint(3, -7).varint(20, int64(AttributeInt)).unknown(unk)).
		msg(5, pb{}.str(1, "auto_pad").str(4, "SAME_UPPER").varint(20, int64(AttributeString)).unknown(unk)).
		msg(5, pb{}.str(1, "alpha").float(2, 0.25).varint(20, int64(AttributeFloat)).unknown(unk)).
		msg(5, pb{}.str(1, "scales").float(7, 1.5).packedFloats(7, -2, 4).varint(20, int64(AttributeFloats)).unknown(unk)).
		msg(5, pb{}.str(1, "names").str(9, "a").str(9, "bc").varint(20, int64(AttributeStrings)).unknown(unk)).
		msg(5, pb{}.str(1, "value").msg(5, attrTensor).varint(20, int64(AttributeTensor)).unknown(unk)).
		msg(5, pb{}.str(1, "body").msg(6, subgraph).msg(11, subgraph).msg(10, attrTensor).
			varint(20, int64(AttributeGraph)).unknown(unk)).
		unknown(unk)
	neg := pb{}.str(1, "Y").str(2, "Z").str(3, "neg0").str(4, "Neg").str(7, "ai.onnx.ml").unknown(unk)
	add := pb{}.str(1, "Y").str(1, "Z").str(1, "Y").str(2, "out").str(3, "add0").str(4, "Add").unknown(unk)

	graph := pb{}.
		msg(1, conv).
		str(2, "g").
		msg(5, weights).
		msg(1, neg).
		msg(5, bias).
		str(10, "graph doc").
		msg(11, inputX).
		msg(12, outputOut).
		msg(13, pb{}.str(1, "Y")).
		msg(14, pb{}.str(1, "q")).
		msg(15, pb{}.str(1, "sparse")).
		msg(1, add).
		unknown(unk)

	return pb{}.
		varint(1, 8).
		str(2, "hand").
		str(3, "1.0").
		str(4, "ai.test").
		varint(5, 3).
		str(6, "model doc").
		msg(7, graph).
		msg(8, pb{}.str(1, "").varint(2, 15).unknown(unk)).
		msg(8, pb{}.str(1, "ai.onnx.ml").varint(2, 2).unknown(unk)).
		msg(14, pb{}.str(1, "key").str(2, "value")).
		msg(20, pb{}.msg(1, pb{})).
		msg(25, pb{}.str(1, "fn")).
		unknown(unk)
}

// wantModel is what buildModel must decode to.
func wantModel() *Model {
	return &Model{
		IRVersion:       8,
		ProducerName:    "hand",
		ProducerVersion: "1.0",
		Opsets:          []OpsetID{{"", 15}, {"ai.onnx.ml", 2}},
		Graph: Graph{
			Name: "g",
			Nodes: []Node{
				{
					Name: "conv0", OpType: "Conv",
					Inputs: []string{"X", "W", "B"}, Outputs: []string{"Y"},
					Attributes: []Attribute{
						{Name: "strides", Type: AttributeInts, Ints: []int64{1, 3}},
						{Name: "pads", Type: AttributeInts, Ints: []int64{-1, 2}},
						{Name: "group", Type: AttributeInt, I: -7},
						{Name: "auto_pad", Type: AttributeString, S: []byte("SAME_UPPER")},
						{Name: "alpha", Type: AttributeFloat, F: 0.25},
						{Name: "scales", Type: AttributeFloats, Floats: []float32{1.5, -2, 4}},
						{Name: "names", Type: AttributeStrings, Strings: [][]byte{[]byte("a"), []byte("bc")}},
						{Name: "value", Type: AttributeTensor, T: &Tensor{
							Dims: []int64{1}, DataType: DataTypeDouble, DoubleData: []float64{2.5},
						}},
						{Name: "body", Type: AttributeGraph},
					},
				},
				{Name: "neg0", OpType: "Neg", Domain: "ai.onnx.ml", Inputs: []string{"Y"}, Outputs: []string{"Z"}},
				{Name: "add0", OpType: "Add", Inputs: []string{"Y", "Z", "Y"}, Outputs: []string{"out"}},
			},
			Initializers: []Tensor{
				{Name: "W", Dims: []int64{2, 1, 1, 2}, DataType: DataTypeFloat, RawData: rawLE(1, -2, 0.5, 3)},
				{Name: "B", Dims: []int64{2}, DataType: DataTypeInt64, Int64Data: []int64{-1, 5}},
			},
			Inputs: []ValueInfo{
				{Name: "X", ElemType: DataTypeFloat, Dims: []Dim{{Param: "N"}, {Value: 4}}},
			},
			Outputs: []ValueInfo{
				{Name: "out", ElemType: DataTypeInt64, Dims: []Dim{{Value: -1}}},
			},
		},
	}
}

func TestParseHandBuilt(t *testing.T) {
	t.Parallel()

	for _, unk := range []bool{false, true} {
		got, err := Parse(buildModel(unk))
		if err != nil {
			t.Fatalf("unknown=%v: %v", unk, err)
		}

		if want := wantModel(); !reflect.DeepEqual(got, want) {
			t.Errorf("unknown=%v:\n got %+v\nwant %+v", unk, got, want)
		}
	}
}

func TestParseEmpty(t *testing.T) {
	t.Parallel()

	got, err := Parse(nil)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(got, &Model{}) {
		t.Errorf("Parse(nil) = %+v, want zero Model", got)
	}
}

// TestParseDoesNotAlias checks that mutating the input after Parse leaves the
// decoded model untouched.
func TestParseDoesNotAlias(t *testing.T) {
	t.Parallel()

	data := buildModel(false)

	m, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}

	for i := range data {
		data[i] = 0xff
	}

	if want := wantModel(); !reflect.DeepEqual(m, want) {
		t.Errorf("model changed after mutating input:\n got %+v\nwant %+v", m, want)
	}
}

// TestParseMergesEmbedded checks protobuf merge semantics for a repeated
// occurrence of a singular embedded message (graph, attribute tensor) and
// last-one-wins for scalars.
func TestParseMergesEmbedded(t *testing.T) {
	t.Parallel()

	data := pb{}.
		msg(7, pb{}.str(2, "first").msg(1, pb{}.str(4, "A"))).
		varint(1, 3).
		msg(7, pb{}.str(2, "second").msg(1, pb{}.str(4, "B").
			msg(5, pb{}.str(1, "t").msg(5, pb{}.varint(1, 2)).msg(5, pb{}.varint(1, 3).varint(2, 7))))).
		varint(1, 4)

	m, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}

	if m.IRVersion != 4 || m.Graph.Name != "second" || len(m.Graph.Nodes) != 2 {
		t.Fatalf("got ir %d name %q %d nodes", m.IRVersion, m.Graph.Name, len(m.Graph.Nodes))
	}

	want := &Tensor{Dims: []int64{2, 3}, DataType: DataTypeInt64}
	if got := m.Graph.Nodes[1].Attributes[0].T; !reflect.DeepEqual(got, want) {
		t.Errorf("merged tensor = %+v, want %+v", got, want)
	}
}

// TestParseDimOneof checks that dim_value and dim_param replace each other.
func TestParseDimOneof(t *testing.T) {
	t.Parallel()

	dims := pb{}.
		msg(1, pb{}.varint(1, 5).str(2, "p")).
		msg(1, pb{}.str(2, "p").varint(1, 5))
	vi := pb{}.str(1, "v").msg(2, pb{}.msg(1, pb{}.msg(2, dims)))

	m, err := Parse(pb{}.msg(7, pb{}.msg(11, vi)))
	if err != nil {
		t.Fatal(err)
	}

	want := []Dim{{Param: "p"}, {Value: 5}}
	if got := m.Graph.Inputs[0].Dims; !reflect.DeepEqual(got, want) {
		t.Errorf("Dims = %+v, want %+v", got, want)
	}
}

// withTensor wraps a TensorProto as the only initializer of a model.
func withTensor(tensor pb) pb {
	return pb{}.msg(7, pb{}.msg(5, tensor))
}

// withAttr wraps an AttributeProto as the only attribute of the only node.
func withAttr(attr pb) pb {
	return pb{}.msg(7, pb{}.msg(1, pb{}.msg(5, attr)))
}

func TestParseRepeatedEncodings(t *testing.T) {
	t.Parallel()

	tensor := func(m *Model) *Tensor { return &m.Graph.Initializers[0] }
	attr := func(m *Model) *Attribute { return &m.Graph.Nodes[0].Attributes[0] }

	tests := []struct {
		name string
		data pb
		get  func(*Model) any
		want any
	}{
		{
			"dims unpacked", withTensor(pb{}.varint(1, 3).varint(1, 0).varint(1, 7)),
			func(m *Model) any { return tensor(m).Dims },
			[]int64{3, 0, 7},
		},
		{
			"dims packed", withTensor(pb{}.packedVarints(1, 3, 0, 7)),
			func(m *Model) any { return tensor(m).Dims },
			[]int64{3, 0, 7},
		},
		{
			"dims mixed", withTensor(pb{}.packedVarints(1, 3).varint(1, 0).packedVarints(1, 7)),
			func(m *Model) any { return tensor(m).Dims },
			[]int64{3, 0, 7},
		},
		{
			"float_data unpacked", withTensor(pb{}.float(4, 1.5).float(4, -0.25)),
			func(m *Model) any { return tensor(m).FloatData },
			[]float32{1.5, -0.25},
		},
		{
			"float_data packed", withTensor(pb{}.packedFloats(4, 1.5, -0.25)),
			func(m *Model) any { return tensor(m).FloatData },
			[]float32{1.5, -0.25},
		},
		{
			"int32_data unpacked", withTensor(pb{}.varint(5, -5).varint(5, math.MaxInt32)),
			func(m *Model) any { return tensor(m).Int32Data },
			[]int32{-5, math.MaxInt32},
		},
		{
			"int32_data packed", withTensor(pb{}.packedVarints(5, -5, math.MaxInt32)),
			func(m *Model) any { return tensor(m).Int32Data },
			[]int32{-5, math.MaxInt32},
		},
		{
			"int64_data unpacked", withTensor(pb{}.varint(7, math.MinInt64).varint(7, math.MaxInt64)),
			func(m *Model) any { return tensor(m).Int64Data },
			[]int64{math.MinInt64, math.MaxInt64},
		},
		{
			"int64_data packed", withTensor(pb{}.packedVarints(7, math.MinInt64, math.MaxInt64)),
			func(m *Model) any { return tensor(m).Int64Data },
			[]int64{math.MinInt64, math.MaxInt64},
		},
		{
			"double_data unpacked", withTensor(pb{}.double(10, 1e300).double(10, -2)),
			func(m *Model) any { return tensor(m).DoubleData },
			[]float64{1e300, -2},
		},
		{
			"double_data packed", withTensor(pb{}.packedDoubles(10, 1e300, -2)),
			func(m *Model) any { return tensor(m).DoubleData },
			[]float64{1e300, -2},
		},
		{
			"attr floats unpacked", withAttr(pb{}.float(7, 0.5).float(7, 2)),
			func(m *Model) any { return attr(m).Floats },
			[]float32{0.5, 2},
		},
		{
			"attr floats packed", withAttr(pb{}.packedFloats(7, 0.5, 2)),
			func(m *Model) any { return attr(m).Floats },
			[]float32{0.5, 2},
		},
		{
			"attr ints unpacked", withAttr(pb{}.varint(8, -1).varint(8, 1)),
			func(m *Model) any { return attr(m).Ints },
			[]int64{-1, 1},
		},
		{
			"attr ints packed", withAttr(pb{}.packedVarints(8, -1, 1)),
			func(m *Model) any { return attr(m).Ints },
			[]int64{-1, 1},
		},
		{
			"empty packed", withAttr(pb{}.packedVarints(8)),
			func(m *Model) any { return attr(m).Ints }, []int64(nil),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, err := Parse(tc.data)
			if err != nil {
				t.Fatal(err)
			}

			if got := tc.get(m); !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %v, want %v", got, tc.want)
			}
		})
	}
}

// TestParseNegativeInt32 checks that negative int32 enum fields, which
// protobuf encodes as 10-byte sign-extended varints, decode correctly.
func TestParseNegativeInt32(t *testing.T) {
	t.Parallel()

	m, err := Parse(withTensor(pb{}.varint(2, -3)))
	if err != nil {
		t.Fatal(err)
	}

	if got := m.Graph.Initializers[0].DataType; got != -3 {
		t.Errorf("DataType = %d, want -3", got)
	}

	if n := len(pb{}.varint(2, -3)); n != 11 {
		t.Errorf("encoded negative int32 field is %d bytes, want 1 + 10", n)
	}
}

func TestParseErrors(t *testing.T) {
	t.Parallel()

	overflow := append(pb{0x08}, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01)
	tenthByte := append(pb{0x08}, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02)

	tests := []struct {
		name string
		data pb
		is   error  // if non-nil, errors.Is must hold
		msg  string // substring of the error text
	}{
		{"truncated tag", pb{0x80}, errTruncated, "tag"},
		{"truncated varint value", pb{0x08, 0x80}, errTruncated, "field 1"},
		{"varint overflow 11 bytes", overflow, errVarintOverflow, ""},
		{"varint overflow tenth byte", tenthByte, errVarintOverflow, ""},
		{"field number zero", pb{}.key(0, wireVarint).rawVarint(1), errBadField, ""},
		{"field number too large", pb{}.key(maxFieldNumber+1, wireVarint).rawVarint(1), errBadField, ""},
		{"start group", pb{}.key(9, wireStartGroup), errGroup, "field 9"},
		{"end group", pb{}.key(9, wireEndGroup), errGroup, ""},
		{"wire type 6", pb{}.key(9, 6), errBadWireType, ""},
		{"wire type 7", pb{}.key(9, 7), errBadWireType, ""},
		{"truncated length", pb{}.key(2, wireBytes), errTruncated, "length"},
		{"truncated length-delimited", append(pb{}.key(2, wireBytes).rawVarint(5), 'a', 'b'), errBadLength, ""},
		{"length beyond buffer", pb{}.key(2, wireBytes).rawVarint(math.MaxUint64), errBadLength, ""},
		{"truncated fixed32", append(pb{}.key(900, wireFixed32), 1, 2, 3), errTruncated, ""},
		{"truncated fixed64", append(pb{}.key(900, wireFixed64), 1, 2, 3, 4, 5, 6, 7), errTruncated, ""},
		{"wrong wire type scalar", pb{}.str(1, "8"), errBadWireType, "field 1"},
		{"wrong wire type message", pb{}.varint(7, 1), errBadWireType, ""},
		{"wrong wire type string", pb{}.varint(2, 1), errBadWireType, ""},
		{"wrong wire type float attr", withAttr(pb{}.varint(2, 1)), errBadWireType, "attribute 0"},
		{"wrong wire type repeated varint", withTensor(pb{}.float(1, 1)), errBadWireType, ""},
		{"wrong wire type repeated float", withTensor(pb{}.varint(4, 1)), errBadWireType, ""},
		{"wrong wire type repeated double", withTensor(pb{}.float(10, 1)), errBadWireType, ""},
		{"packed float length", withTensor(pb{}.bytes(4, []byte{1, 2, 3})), nil, "multiple of 4"},
		{"packed double length", withTensor(pb{}.bytes(10, []byte{1, 2, 3, 4})), nil, "multiple of 8"},
		{"truncated packed varint", withTensor(pb{}.bytes(1, []byte{0x80})), errTruncated, "packed"},
		{"external data field", withTensor(pb{}.str(8, "w").msg(13, pb{}.str(1, "location"))), ErrExternalData, "initializer 0"},
		{"data_location external", withTensor(pb{}.varint(14, 1)), ErrExternalData, "external data not supported"},
		{"segment", withTensor(pb{}.msg(3, pb{}.varint(1, 0).varint(2, 4))), ErrSegment, ""},
		{"nested truncation", pb{}.msg(7, append(pb{}.key(1, wireBytes).rawVarint(3), 0x22)), errBadLength, "graph"},
		{"error inside attribute tensor", withAttr(pb{}.msg(5, pb{}.varint(14, 1))), ErrExternalData, "attribute 0: field 5: t"},
		{"error inside value info", pb{}.msg(7, pb{}.msg(12, pb{}.msg(2, pb{}.msg(1, pb{}.msg(2, pb{}.msg(1, pb{0x08})))))), errTruncated, "output 0"},
		{"error inside opset", pb{}.msg(8, pb{}.str(2, "x")), errBadWireType, "opset_import 0"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m, err := Parse(tc.data)
			if err == nil {
				t.Fatalf("Parse succeeded: %+v", m)
			}

			if m != nil {
				t.Errorf("Parse returned a model alongside error %v", err)
			}

			if !strings.HasPrefix(err.Error(), "onnx: ") {
				t.Errorf("error %q lacks the onnx: prefix", err)
			}

			if tc.is != nil && !errors.Is(err, tc.is) {
				t.Errorf("error %q is not %q", err, tc.is)
			}

			if !strings.Contains(err.Error(), tc.msg) {
				t.Errorf("error %q does not mention %q", err, tc.msg)
			}
		})
	}
}

// TestParseTruncatedPrefixes parses every prefix of a valid model: all must
// either succeed or fail cleanly.
func TestParseTruncatedPrefixes(t *testing.T) {
	t.Parallel()

	data := buildModel(true)
	for i := range data {
		_, _ = Parse(data[:i])
	}
}

func FuzzParse(f *testing.F) {
	f.Add([]byte(buildModel(false)))
	f.Add([]byte(buildModel(true)))
	f.Add([]byte(withTensor(pb{}.packedVarints(1, 2, 2).varint(2, int64(DataTypeFloat)).bytes(9, rawLE(1, 2, 3, 4)))))
	f.Add([]byte(withTensor(pb{}.varint(1, 2).varint(2, int64(DataTypeInt32)).packedVarints(5, -1, 1))))

	f.Fuzz(func(t *testing.T, data []byte) {
		m, err := Parse(data)
		if err != nil {
			if m != nil {
				t.Fatalf("model returned with error %v", err)
			}

			return
		}

		g := &m.Graph
		for i := range g.Initializers {
			tensor := &g.Initializers[i]
			_ = tensor.NumElements()
			_, _ = tensor.Float32s()
			_, _ = tensor.Int64s()
			_ = g.Initializer(tensor.Name)
		}

		for i := range g.Nodes {
			n := &g.Nodes[i]
			for _, out := range n.Outputs {
				_ = g.Producer(out)
				_ = g.Consumers(out)
			}

			for _, a := range n.Attributes {
				_ = n.Attr(a.Name)
				if a.T != nil {
					_, _ = a.T.Float32s()
					_, _ = a.T.Int64s()
				}
			}
		}
	})
}
