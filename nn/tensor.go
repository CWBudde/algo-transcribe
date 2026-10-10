package nn

import "fmt"

// Shape is the shape of an NHWC tensor with a batch size of one.
type Shape struct {
	H, W, C int
}

// Size returns the number of elements, H*W*C.
func (s Shape) Size() int { return s.H * s.W * s.C }

func (s Shape) String() string { return fmt.Sprintf("(%d,%d,%d)", s.H, s.W, s.C) }

func (s Shape) valid() bool { return s.H > 0 && s.W > 0 && s.C > 0 }

// Tensor is a dense float32 tensor in NHWC layout with a batch size of one.
// Element (y, x, c) is stored at Data[(y*W+x)*C+c], so one row y of the
// tensor is the contiguous slice Data[y*W*C : (y+1)*W*C].
//
// In basic-pitch, H is time and W is frequency.
type Tensor struct {
	Shape

	Data []float32
}

// NewTensor allocates a zeroed tensor of shape (h, w, c).
func NewTensor(h, w, c int) *Tensor {
	s := Shape{H: h, W: w, C: c}

	return &Tensor{Shape: s, Data: make([]float32, s.Size())}
}

// TensorFrom wraps data as a tensor of shape (h, w, c) without copying.
func TensorFrom(data []float32, h, w, c int) (*Tensor, error) {
	s := Shape{H: h, W: w, C: c}
	if !s.valid() {
		return nil, fmt.Errorf("nn: tensor: invalid shape %v", s)
	}

	if len(data) != s.Size() {
		return nil, fmt.Errorf("nn: tensor: %d elements for shape %v, want %d", len(data), s, s.Size())
	}

	return &Tensor{Shape: s, Data: data}, nil
}

// Index returns the offset of element (y, x, c) in Data.
func (t *Tensor) Index(y, x, c int) int { return (y*t.W+x)*t.C + c }

// At returns element (y, x, c).
func (t *Tensor) At(y, x, c int) float32 { return t.Data[t.Index(y, x, c)] }

// Set sets element (y, x, c) to v.
func (t *Tensor) Set(y, x, c int, v float32) { t.Data[t.Index(y, x, c)] = v }

// Rows returns the contiguous data of rows [r0, r1).
func (t *Tensor) Rows(r0, r1 int) []float32 {
	n := t.W * t.C

	return t.Data[r0*n : r1*n]
}

func (t *Tensor) check(name string) error {
	if t == nil {
		return fmt.Errorf("nn: %s: nil tensor", name)
	}

	if !t.valid() || len(t.Data) != t.Size() {
		return fmt.Errorf("nn: %s: %d elements for shape %v", name, len(t.Data), t.Shape)
	}

	return nil
}
