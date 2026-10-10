package onnx

import (
	"encoding/binary"
	"math"
	"reflect"
	"strings"
	"testing"
)

func TestNumElements(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		dims []int64
		want int64
	}{
		{"scalar nil dims", nil, 1},
		{"scalar empty dims", []int64{}, 1},
		{"vector", []int64{5}, 5},
		{"4d", []int64{8, 8, 3, 39}, 7488},
		{"zero dim", []int64{3, 0, 4}, 0},
		{"negative dim", []int64{2, -1}, -1},
		{"overflow", []int64{math.MaxInt64, 2}, -1},
		{"zero after huge", []int64{math.MaxInt64, 0}, 0},
	}

	for _, tc := range tests {
		if got := (&Tensor{Dims: tc.dims}).NumElements(); got != tc.want {
			t.Errorf("%s: NumElements(%v) = %d, want %d", tc.name, tc.dims, got, tc.want)
		}
	}
}

func TestFloat32s(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		tensor Tensor
		want   []float32
		errMsg string
	}{
		{
			name:   "raw_data",
			tensor: Tensor{Dims: []int64{2, 2}, DataType: DataTypeFloat, RawData: rawLE(1, -2, 0.5, 3)},
			want:   []float32{1, -2, 0.5, 3},
		},
		{
			name:   "float_data",
			tensor: Tensor{Dims: []int64{3}, DataType: DataTypeFloat, FloatData: []float32{1, 2, 3}},
			want:   []float32{1, 2, 3},
		},
		{
			name:   "scalar raw",
			tensor: Tensor{DataType: DataTypeFloat, RawData: rawLE(7)},
			want:   []float32{7},
		},
		{
			name:   "empty",
			tensor: Tensor{Dims: []int64{0, 4}, DataType: DataTypeFloat},
			want:   []float32{},
		},
		{
			name:   "raw_data length mismatch",
			tensor: Tensor{Dims: []int64{3}, DataType: DataTypeFloat, RawData: rawLE(1, 2)},
			errMsg: "raw_data has 8 bytes",
		},
		{
			name:   "raw_data not a multiple of 4",
			tensor: Tensor{Dims: []int64{1}, DataType: DataTypeFloat, RawData: []byte{1, 2, 3, 4, 5}},
			errMsg: "raw_data has 5 bytes",
		},
		{
			name:   "float_data length mismatch",
			tensor: Tensor{Dims: []int64{3}, DataType: DataTypeFloat, FloatData: []float32{1}},
			errMsg: "1 typed values for 3 elements",
		},
		{
			name:   "no data",
			tensor: Tensor{Dims: []int64{2}, DataType: DataTypeFloat},
			errMsg: "0 typed values for 2 elements",
		},
		{
			name:   "both raw and typed",
			tensor: Tensor{DataType: DataTypeFloat, RawData: rawLE(1), FloatData: []float32{1}},
			errMsg: "both raw_data and typed data",
		},
		{
			name:   "wrong data type",
			tensor: Tensor{Name: "x", DataType: DataTypeInt64, Int64Data: []int64{1}},
			errMsg: `tensor "x": Float32s on data type 7`,
		},
		{
			name:   "invalid dims",
			tensor: Tensor{Dims: []int64{-1}, DataType: DataTypeFloat},
			errMsg: "invalid dims",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := tc.tensor.Float32s()
			checkResult(t, got, tc.want, err, tc.errMsg)
		})
	}
}

func TestInt64s(t *testing.T) {
	t.Parallel()

	int32Raw := binary.LittleEndian.AppendUint32(nil, uint32(0xfffffffb)) // -5
	int32Raw = binary.LittleEndian.AppendUint32(int32Raw, 7)

	tests := []struct {
		name   string
		tensor Tensor
		want   []int64
		errMsg string
	}{
		{
			name:   "int64 raw_data",
			tensor: Tensor{Dims: []int64{4}, DataType: DataTypeInt64, RawData: rawLE64(0, -309, 0, math.MinInt64)},
			want:   []int64{0, -309, 0, math.MinInt64},
		},
		{
			name:   "int64_data",
			tensor: Tensor{Dims: []int64{2}, DataType: DataTypeInt64, Int64Data: []int64{-1, 1}},
			want:   []int64{-1, 1},
		},
		{
			name:   "int32 raw_data sign-extended",
			tensor: Tensor{Dims: []int64{2}, DataType: DataTypeInt32, RawData: int32Raw},
			want:   []int64{-5, 7},
		},
		{
			name:   "int32_data widened",
			tensor: Tensor{DataType: DataTypeInt32, Int32Data: []int32{math.MinInt32}},
			want:   []int64{math.MinInt32},
		},
		{
			name:   "int64 raw_data length mismatch",
			tensor: Tensor{Dims: []int64{2}, DataType: DataTypeInt64, RawData: rawLE64(1)},
			errMsg: "raw_data has 8 bytes",
		},
		{
			name:   "int32_data length mismatch",
			tensor: Tensor{Dims: []int64{2}, DataType: DataTypeInt32, Int32Data: []int32{1, 2, 3}},
			errMsg: "3 typed values for 2 elements",
		},
		{
			name:   "int64 tensor with int32_data only",
			tensor: Tensor{DataType: DataTypeInt64, Int32Data: []int32{1}},
			errMsg: "0 typed values for 1 elements",
		},
		{
			name:   "wrong data type",
			tensor: Tensor{Name: "f", DataType: DataTypeFloat, FloatData: []float32{1}},
			errMsg: `tensor "f": Int64s on data type 1`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := tc.tensor.Int64s()
			checkResult(t, got, tc.want, err, tc.errMsg)
		})
	}
}

// checkResult compares a conversion result against either want or an
// expected error substring.
func checkResult[T any](t *testing.T, got, want []T, err error, errMsg string) {
	t.Helper()

	if errMsg != "" {
		if err == nil || !strings.Contains(err.Error(), errMsg) || !strings.HasPrefix(err.Error(), "onnx: ") {
			t.Fatalf("err = %v, want onnx: ... %q", err, errMsg)
		}

		if got != nil {
			t.Errorf("got %v alongside an error", got)
		}

		return
	}

	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// TestConversionsReturnFreshSlices checks that callers may modify the results
// without touching the tensor.
func TestConversionsReturnFreshSlices(t *testing.T) {
	t.Parallel()

	f := Tensor{Dims: []int64{2}, DataType: DataTypeFloat, FloatData: []float32{1, 2}}
	fs, _ := f.Float32s()
	fs[0] = 99

	i := Tensor{Dims: []int64{2}, DataType: DataTypeInt64, Int64Data: []int64{1, 2}}
	is, _ := i.Int64s()
	is[0] = 99

	if f.FloatData[0] != 1 || i.Int64Data[0] != 1 {
		t.Errorf("conversion aliased tensor data: %v %v", f.FloatData, i.Int64Data)
	}
}
