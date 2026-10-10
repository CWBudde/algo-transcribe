// Package onnx is a minimal, dependency-free reader for ONNX model files.
//
// It walks the protobuf wire format of ModelProto just far enough to recover
// the graph: initializers (TensorProto with dims, data type and raw_data or
// typed data fields), nodes (op type, inputs, outputs, attributes) and the
// graph inputs and outputs. It exists for the weight extractor in
// internal/cmd/extractweights and is not a general ONNX implementation:
// external data, sparse tensors, subgraph attributes and training info are
// skipped or rejected.
package onnx

// DataType is the TensorProto.DataType enum.
type DataType int32

// TensorProto.DataType values used by the reader.
const (
	DataTypeUndefined DataType = 0
	DataTypeFloat     DataType = 1
	DataTypeUint8     DataType = 2
	DataTypeInt8      DataType = 3
	DataTypeUint16    DataType = 4
	DataTypeInt16     DataType = 5
	DataTypeInt32     DataType = 6
	DataTypeInt64     DataType = 7
	DataTypeString    DataType = 8
	DataTypeBool      DataType = 9
	DataTypeFloat16   DataType = 10
	DataTypeDouble    DataType = 11
	DataTypeUint32    DataType = 12
	DataTypeUint64    DataType = 13
)

// AttributeType is the AttributeProto.AttributeType enum.
type AttributeType int32

// AttributeProto.AttributeType values.
const (
	AttributeUndefined AttributeType = 0
	AttributeFloat     AttributeType = 1
	AttributeInt       AttributeType = 2
	AttributeString    AttributeType = 3
	AttributeTensor    AttributeType = 4
	AttributeGraph     AttributeType = 5
	AttributeFloats    AttributeType = 6
	AttributeInts      AttributeType = 7
	AttributeStrings   AttributeType = 8
	AttributeTensors   AttributeType = 9
	AttributeGraphs    AttributeType = 10
)

// Model is the subset of ModelProto the reader decodes.
type Model struct {
	IRVersion       int64
	ProducerName    string
	ProducerVersion string
	Opsets          []OpsetID
	Graph           Graph
}

// OpsetID is an OperatorSetIdProto.
type OpsetID struct {
	Domain  string
	Version int64
}

// Graph is the subset of GraphProto the reader decodes.
type Graph struct {
	Name         string
	Nodes        []Node
	Initializers []Tensor
	Inputs       []ValueInfo
	Outputs      []ValueInfo
}

// Node is a NodeProto.
type Node struct {
	Name       string
	OpType     string
	Domain     string
	Inputs     []string
	Outputs    []string
	Attributes []Attribute
}

// Attribute is an AttributeProto. Only the field matching Type is meaningful.
type Attribute struct {
	Name    string
	Type    AttributeType
	F       float32
	I       int64
	S       []byte
	T       *Tensor
	Floats  []float32
	Ints    []int64
	Strings [][]byte
}

// Tensor is a TensorProto. Data lives either in RawData (little-endian) or in
// the typed field matching DataType.
type Tensor struct {
	Name       string
	Dims       []int64
	DataType   DataType
	RawData    []byte
	FloatData  []float32
	Int32Data  []int32
	Int64Data  []int64
	DoubleData []float64
}

// ValueInfo is a ValueInfoProto restricted to tensor types.
type ValueInfo struct {
	Name     string
	ElemType DataType
	Dims     []Dim
}

// Dim is one TensorShapeProto.Dimension: either a fixed Value or a symbolic
// Param (then Value is 0).
type Dim struct {
	Value int64
	Param string
}
