package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
)

// NumElements returns the number of elements described by Dims: their
// product, or 1 for a scalar without dims. It returns -1 if a dimension is
// negative or the product overflows int64.
func (t *Tensor) NumElements() int64 {
	n := int64(1)

	for _, d := range t.Dims {
		if d < 0 {
			return -1
		}

		if d != 0 && n > math.MaxInt64/d {
			return -1
		}

		n *= d
	}

	return n
}

// Float32s returns the values of a DataTypeFloat tensor as a fresh slice,
// decoded from RawData (little-endian) or FloatData. The number of values
// must match NumElements.
func (t *Tensor) Float32s() ([]float32, error) {
	if t.DataType != DataTypeFloat {
		return nil, fmt.Errorf("onnx: tensor %q: Float32s on data type %d", t.Name, t.DataType)
	}

	n, err := t.checkData(4, len(t.FloatData))
	if err != nil {
		return nil, err
	}

	out := make([]float32, n)
	if len(t.RawData) == 0 {
		copy(out, t.FloatData)

		return out, nil
	}

	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.RawData[4*i:]))
	}

	return out, nil
}

// Int64s returns the values of a DataTypeInt64 or DataTypeInt32 tensor as a
// fresh int64 slice, decoded from RawData (little-endian) or from Int64Data or
// Int32Data respectively. int32 values are sign-extended. The number of values
// must match NumElements.
func (t *Tensor) Int64s() ([]int64, error) {
	switch t.DataType {
	case DataTypeInt64:
		n, err := t.checkData(8, len(t.Int64Data))
		if err != nil {
			return nil, err
		}

		out := make([]int64, n)
		if len(t.RawData) == 0 {
			copy(out, t.Int64Data)

			return out, nil
		}

		for i := range out {
			out[i] = int64(binary.LittleEndian.Uint64(t.RawData[8*i:]))
		}

		return out, nil
	case DataTypeInt32:
		n, err := t.checkData(4, len(t.Int32Data))
		if err != nil {
			return nil, err
		}

		out := make([]int64, n)

		for i := range out {
			if len(t.RawData) == 0 {
				out[i] = int64(t.Int32Data[i])
			} else {
				out[i] = int64(int32(binary.LittleEndian.Uint32(t.RawData[4*i:])))
			}
		}

		return out, nil
	default:
		return nil, fmt.Errorf("onnx: tensor %q: Int64s on data type %d", t.Name, t.DataType)
	}
}

// checkData validates that the tensor holds exactly NumElements values,
// either in RawData (elemSize bytes each) or in a typed field with typedLen
// entries, and returns that count.
func (t *Tensor) checkData(elemSize, typedLen int) (int, error) {
	n := t.NumElements()
	if n < 0 {
		return 0, fmt.Errorf("onnx: tensor %q: invalid dims %v", t.Name, t.Dims)
	}

	if len(t.RawData) == 0 {
		if int64(typedLen) != n {
			return 0, fmt.Errorf("onnx: tensor %q: %d typed values for %d elements", t.Name, typedLen, n)
		}

		return typedLen, nil
	}

	if typedLen != 0 {
		return 0, fmt.Errorf("onnx: tensor %q: both raw_data and typed data set", t.Name)
	}

	if len(t.RawData)%elemSize != 0 || int64(len(t.RawData)/elemSize) != n {
		return 0, fmt.Errorf("onnx: tensor %q: raw_data has %d bytes, want %d elements of %d bytes",
			t.Name, len(t.RawData), n, elemSize)
	}

	return int(n), nil
}
