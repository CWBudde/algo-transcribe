package nn

import (
	"strings"
	"testing"
)

func TestTensorNew(t *testing.T) {
	cases := []struct {
		h, w, c int
		size    int
		str     string
	}{
		{h: 1, w: 1, c: 1, size: 1, str: "(1,1,1)"},
		{h: 172, w: 264, c: 8, size: 172 * 264 * 8, str: "(172,264,8)"},
		{h: 3, w: 4, c: 5, size: 60, str: "(3,4,5)"},
	}

	for _, tc := range cases {
		t.Run(tc.str, func(t *testing.T) {
			x := NewTensor(tc.h, tc.w, tc.c)

			if want := (Shape{H: tc.h, W: tc.w, C: tc.c}); x.Shape != want {
				t.Fatalf("shape %v, want %v", x.Shape, want)
			}

			if x.Size() != tc.size || len(x.Data) != tc.size {
				t.Fatalf("Size() = %d, len(Data) = %d, want %d", x.Size(), len(x.Data), tc.size)
			}

			if x.String() != tc.str {
				t.Errorf("String() = %q, want %q", x.String(), tc.str)
			}

			for i, v := range x.Data {
				if v != 0 {
					t.Fatalf("Data[%d] = %v, want 0", i, v)
				}
			}

			if err := x.check("t"); err != nil {
				t.Errorf("check: %v", err)
			}
		})
	}
}

func TestTensorFrom(t *testing.T) {
	cases := []struct {
		name    string
		n       int
		nilData bool
		h, w, c int
		wantErr string
	}{
		{name: "valid", n: 24, h: 2, w: 3, c: 4},
		{name: "valid-1x1x1", n: 1, h: 1, w: 1, c: 1},
		{name: "too-short", n: 23, h: 2, w: 3, c: 4, wantErr: "23 elements"},
		{name: "too-long", n: 25, h: 2, w: 3, c: 4, wantErr: "25 elements"},
		{name: "nil-data", nilData: true, h: 2, w: 3, c: 4, wantErr: "0 elements"},
		{name: "H=0", n: 0, h: 0, w: 3, c: 4, wantErr: "invalid shape"},
		{name: "W=0", n: 0, h: 2, w: 0, c: 4, wantErr: "invalid shape"},
		{name: "C=0", n: 0, h: 2, w: 3, c: 0, wantErr: "invalid shape"},
		{name: "negative", n: 6, h: -2, w: -3, c: 1, wantErr: "invalid shape"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var data []float32
			if !tc.nilData {
				data = make([]float32, tc.n)
			}

			x, err := TensorFrom(data, tc.h, tc.w, tc.c)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("TensorFrom succeeded with shape %v", x.Shape)
				}

				if x != nil {
					t.Errorf("non-nil tensor with error %v", err)
				}

				if !strings.HasPrefix(err.Error(), "nn: tensor: ") || !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("error %q, want prefix %q and %q", err, "nn: tensor: ", tc.wantErr)
				}

				return
			}

			if err != nil {
				t.Fatal(err)
			}

			if want := (Shape{H: tc.h, W: tc.w, C: tc.c}); x.Shape != want {
				t.Fatalf("shape %v, want %v", x.Shape, want)
			}

			// No copy: the tensor aliases data.
			data[len(data)-1] = 3.5
			if x.Data[len(x.Data)-1] != 3.5 {
				t.Error("TensorFrom copied the data")
			}
		})
	}
}

func TestTensorLayout(t *testing.T) {
	x := NewTensor(3, 4, 5)

	cases := []struct {
		y, xx, c int
		want     int
	}{
		{0, 0, 0, 0},
		{0, 0, 4, 4},
		{0, 1, 0, 5},
		{0, 3, 4, 19},
		{1, 0, 0, 20},
		{1, 2, 3, (1*4+2)*5 + 3},
		{2, 3, 4, 59},
	}

	for _, tc := range cases {
		if got := x.Index(tc.y, tc.xx, tc.c); got != tc.want {
			t.Errorf("Index(%d,%d,%d) = %d, want %d", tc.y, tc.xx, tc.c, got, tc.want)
		}
	}

	// Set writes the element Index points to and nothing else; At reads it
	// back.
	for i, tc := range cases {
		v := float32(i + 1)
		x.Set(tc.y, tc.xx, tc.c, v)

		if x.Data[tc.want] != v {
			t.Errorf("Set(%d,%d,%d) wrote Data[%d] = %v, want %v", tc.y, tc.xx, tc.c, tc.want, x.Data[tc.want], v)
		}

		if got := x.At(tc.y, tc.xx, tc.c); got != v {
			t.Errorf("At(%d,%d,%d) = %v, want %v", tc.y, tc.xx, tc.c, got, v)
		}
	}

	nonZero := 0

	for _, v := range x.Data {
		if v != 0 {
			nonZero++
		}
	}

	if nonZero != len(cases) {
		t.Errorf("%d elements set, want %d", nonZero, len(cases))
	}

	// Full sweep: At sees exactly the value stored at (y*W+x)*C+c.
	for i := range x.Data {
		x.Data[i] = float32(i)
	}

	for y := range x.H {
		for xx := range x.W {
			for c := range x.C {
				if got, want := x.At(y, xx, c), float32((y*x.W+xx)*x.C+c); got != want {
					t.Fatalf("At(%d,%d,%d) = %v, want %v", y, xx, c, got, want)
				}
			}
		}
	}
}

func TestTensorRows(t *testing.T) {
	x := NewTensor(4, 3, 2)
	for i := range x.Data {
		x.Data[i] = float32(i)
	}

	cases := []struct {
		r0, r1 int
	}{
		{0, 4},
		{0, 1},
		{1, 3},
		{3, 4},
		{2, 2},
		{4, 4},
	}

	n := x.W * x.C

	for _, tc := range cases {
		rows := x.Rows(tc.r0, tc.r1)
		if len(rows) != (tc.r1-tc.r0)*n {
			t.Errorf("Rows(%d,%d) has %d values, want %d", tc.r0, tc.r1, len(rows), (tc.r1-tc.r0)*n)

			continue
		}

		for i, v := range rows {
			if want := float32(tc.r0*n + i); v != want {
				t.Errorf("Rows(%d,%d)[%d] = %v, want %v", tc.r0, tc.r1, i, v, want)
			}
		}
	}

	// Rows aliases Data.
	x.Rows(2, 3)[1] = -1
	if x.At(2, 0, 1) != -1 {
		t.Error("Rows does not alias Data")
	}
}

func TestTensorCheck(t *testing.T) {
	var nilTensor *Tensor

	cases := []struct {
		name    string
		x       *Tensor
		wantErr string
	}{
		{name: "valid", x: NewTensor(2, 2, 2)},
		{name: "nil", x: nilTensor, wantErr: "nil tensor"},
		{name: "zero", x: &Tensor{}, wantErr: "elements for shape"},
		{name: "short", x: &Tensor{Shape: Shape{2, 2, 2}, Data: make([]float32, 7)}, wantErr: "7 elements"},
		{name: "long", x: &Tensor{Shape: Shape{2, 2, 2}, Data: make([]float32, 9)}, wantErr: "9 elements"},
		{name: "negative", x: &Tensor{Shape: Shape{-1, -1, 1}, Data: make([]float32, 1)}, wantErr: "elements for shape"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.x.check("op: x")
			if tc.wantErr == "" {
				if err != nil {
					t.Fatal(err)
				}

				return
			}

			if err == nil {
				t.Fatal("check succeeded")
			}

			if !strings.HasPrefix(err.Error(), "nn: op: x: ") || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error %q, want prefix %q and %q", err, "nn: op: x: ", tc.wantErr)
			}
		})
	}
}
