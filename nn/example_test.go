package nn_test

import (
	"fmt"

	"github.com/cwbudde/algo-transcribe/nn"
)

func ExampleNewTensor() {
	t := nn.NewTensor(2, 3, 4)
	fmt.Println(t.Shape, len(t.Data), t.Data[0])
	// Output:
	// (2,3,4) 24 0
}

func ExampleTensorFrom() {
	data := []float32{1, 2, 3, 4, 5, 6}

	t, err := nn.TensorFrom(data, 1, 3, 2)
	if err != nil {
		fmt.Println(err)

		return
	}

	// No copy: the tensor shares data.
	data[5] = 60

	fmt.Println(t.Shape, t.At(0, 2, 1))

	_, err = nn.TensorFrom(data, 2, 3, 2)
	fmt.Println(err)
	// Output:
	// (1,3,2) 60
	// nn: tensor: 6 elements for shape (2,3,2), want 12
}

func ExampleTensor() {
	t := nn.NewTensor(2, 3, 2)
	t.Set(1, 2, 0, 7)

	// Element (y, x, c) lives at (y*W+x)*C+c.
	fmt.Println(t.Index(1, 2, 0), t.At(1, 2, 0), t.Data[10])
	// Output:
	// 10 7 7
}

func ExampleTensor_Rows() {
	t := nn.NewTensor(3, 2, 2)
	for i := range t.Data {
		t.Data[i] = float32(i)
	}

	// Rows are contiguous, W*C values each.
	fmt.Println(t.Rows(1, 2))
	fmt.Println(len(t.Rows(0, 3)))
	// Output:
	// [4 5 6 7]
	// 12
}

func ExampleShape_Size() {
	s := nn.Shape{H: 172, W: 88, C: 33}
	fmt.Println(s.Size())
	// Output:
	// 499488
}

func ExampleShape_String() {
	fmt.Println(nn.Shape{H: 172, W: 264, C: 8})
	// Output:
	// (172,264,8)
}

func ExampleNewConv2D() {
	// A 3×3 box filter of ones over a 3×3 single-channel input. "same"
	// padding keeps the output 3×3; corners see 4 inputs, edges 6, the
	// centre 9.
	in := nn.Shape{H: 3, W: 3, C: 1}
	weights := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}

	conv, err := nn.NewConv2D(in, 3, 3, 1, weights, nil)
	if err != nil {
		fmt.Println(err)

		return
	}

	src := nn.NewTensor(3, 3, 1)
	for i := range src.Data {
		src.Data[i] = 1
	}

	dst := nn.NewTensor(conv.OutShape().H, conv.OutShape().W, conv.OutShape().C)
	if err := conv.Forward(dst, src); err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println(conv.InShape(), "->", conv.OutShape())

	for y := range 3 {
		fmt.Println(dst.Rows(y, y+1))
	}
	// Output:
	// (3,3,1) -> (3,3,1)
	// [4 6 4]
	// [6 9 6]
	// [4 6 4]
}

func ExampleWithStride() {
	// TensorFlow "same" padding with stride 3 on frequency: the total is
	// (88-1)*3 + k - 264 = k-3. An odd total puts the extra zero after the
	// data, so k=4 pads 0 left and 1 right, k=6 pads 1 left and 2 right.
	// basic-pitch uses k=7 (conv2d_2) and k=5 (conv2d_4).
	in := nn.Shape{H: 1, W: 264, C: 1}

	for _, k := range []int{4, 5, 6, 7} {
		conv, err := nn.NewConv2D(in, 1, k, 1, make([]float32, k), nil, nn.WithStride(1, 3))
		if err != nil {
			fmt.Println(err)

			return
		}

		_, _, left, right := conv.Padding()
		fmt.Printf("k=%d: pad left %d right %d, out W %d\n", k, left, right, conv.OutShape().W)
	}
	// Output:
	// k=4: pad left 0 right 1, out W 88
	// k=5: pad left 1 right 1, out W 88
	// k=6: pad left 1 right 2, out W 88
	// k=7: pad left 2 right 2, out W 88
}

func ExampleConv2D_Forward() {
	// A 1×2 kernel with two input and two output channels and a bias.
	// HWIO: weights[((ky*kw+kx)*cin+ci)*cout+co].
	in := nn.Shape{H: 1, W: 2, C: 2}
	weights := []float32{
		1, 0, // kx=0, ci=0 -> co=0, co=1
		0, 1, // kx=0, ci=1
		0.5, 0, // kx=1, ci=0
		0, 0.5, // kx=1, ci=1
	}
	bias := []float32{0.25, -0.25}

	conv, err := nn.NewConv2D(in, 1, 2, 2, weights, bias)
	if err != nil {
		fmt.Println(err)

		return
	}

	src, err := nn.TensorFrom([]float32{1, 2, 3, 4}, 1, 2, 2)
	if err != nil {
		fmt.Println(err)

		return
	}

	dst := nn.NewTensor(1, 2, 2)
	if err := conv.Forward(dst, src); err != nil {
		fmt.Println(err)

		return
	}

	// k=2 pads 0 before and 1 after, so x=1 sees only itself.
	for x := range 2 {
		fmt.Printf("x=%d: %.4g %.4g\n", x, dst.At(0, x, 0), dst.At(0, x, 1))
	}
	// Output:
	// x=0: 2.75 3.75
	// x=1: 3.25 3.75
}

func ExampleConv2D_ForwardRows() {
	// Sum over three neighbouring rows, computed two output rows per step
	// as a budgeted caller would.
	in := nn.Shape{H: 5, W: 2, C: 1}

	conv, err := nn.NewConv2D(in, 3, 1, 1, []float32{1, 1, 1}, nil)
	if err != nil {
		fmt.Println(err)

		return
	}

	src := nn.NewTensor(5, 2, 1)
	for i := range src.Data {
		src.Data[i] = float32(i)
	}

	dst := nn.NewTensor(5, 2, 1)

	const step = 2

	for r0 := 0; r0 < dst.H; r0 += step {
		r1 := min(r0+step, dst.H)
		if err := conv.ForwardRows(dst, src, r0, r1); err != nil {
			fmt.Println(err)

			return
		}

		fmt.Printf("rows [%d,%d): %.4g\n", r0, r1, dst.Rows(r0, r1))
	}
	// Output:
	// rows [0,2): [2 4 6 9]
	// rows [2,4): [12 15 18 21]
	// rows [4,5): [14 16]
}

func ExampleConv2D_MACsPerRow() {
	// conv2d_1 of basic-pitch: (172,264,8) input, 3×39 kernel, 8 filters.
	in := nn.Shape{H: 172, W: 264, C: 8}

	conv, err := nn.NewConv2D(in, 3, 39, 8, make([]float32, 3*39*8*8), nil)
	if err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println(conv.MACsPerRow(), conv.MACsPerRow()*conv.OutShape().H)
	// Output:
	// 1976832 340015104
}

func ExampleReLU() {
	x := []float32{-2, -0.5, 0, 0.5, 2}
	nn.ReLU(x)
	fmt.Println(x)
	// Output:
	// [0 0 0 0.5 2]
}

func ExampleSigmoid() {
	x := []float32{-20, -1, 0, 1, 20, 100}
	nn.Sigmoid(x)
	fmt.Printf("%.4g\n", x)
	// Output:
	// [2.061e-09 0.2689 0.5 0.7311 1 1]
}

func ExampleConcatChannels() {
	// basic-pitch feeds [note (172,88,1), conv2d_4 (172,88,32)] to the onset
	// convolution; here with a tiny 1×2 frame.
	note, _ := nn.TensorFrom([]float32{0.9, 0.1}, 1, 2, 1)
	feat, _ := nn.TensorFrom([]float32{1, 2, 3, 4}, 1, 2, 2)
	dst := nn.NewTensor(1, 2, 3)

	if err := nn.ConcatChannels(dst, note, feat); err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println(dst.Data)

	fmt.Println(nn.ConcatChannels(nn.NewTensor(1, 2, 4), note, feat))
	// Output:
	// [0.9 1 2 0.1 3 4]
	// nn: concat: sources have 3 channels, dst has 4
}

func ExampleConcatChannelsRows() {
	a := nn.NewTensor(4, 1, 1)
	b := nn.NewTensor(4, 1, 1)

	for y := range 4 {
		a.Data[y] = float32(y)
		b.Data[y] = float32(10 * y)
	}

	dst := nn.NewTensor(4, 1, 2)

	// Only rows [1, 3) are written; the others stay zero.
	if err := nn.ConcatChannelsRows(dst, 1, 3, a, b); err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println(dst.Data)
	// Output:
	// [0 0 1 10 2 20 0 0]
}

func ExampleFoldBatchNorm() {
	// A 1×1 convolution with two input channels and one output channel,
	// followed by BatchNorm. scale = 4/sqrt(3+1) = 2.
	weights := []float32{1, 2}
	bias := []float32{0} // no conv bias: pass zeros
	bn := nn.BatchNorm{
		Gamma:    []float32{4},
		Beta:     []float32{0.5},
		Mean:     []float32{1},
		Variance: []float32{3},
		Epsilon:  1, // the model's epsilon; Keras defaults to 1e-3
	}

	if err := nn.FoldBatchNorm(weights, bias, bn); err != nil {
		fmt.Println(err)

		return
	}

	// bias = (0-1)*2 + 0.5.
	fmt.Printf("weights %.4g, bias %.4g\n", weights, bias)
	// Output:
	// weights [2 4], bias [-1.5]
}
