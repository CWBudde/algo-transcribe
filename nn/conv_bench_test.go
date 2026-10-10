package nn

import (
	"math/rand/v2"
	"testing"
)

// bpLayer is one convolution of the basic-pitch CNN (docs/port-spec.md §4).
type bpLayer struct {
	name             string
	in               Shape
	kh, kw, cout     int
	strideH, strideW int
}

// bpLayers lists the six basic-pitch convolutions for one 172-frame window.
var bpLayers = []bpLayer{
	{"conv2d_1", Shape{H: 172, W: 264, C: 8}, 3, 39, 8, 1, 1},
	{"contours-reduced", Shape{H: 172, W: 264, C: 8}, 5, 5, 1, 1, 1},
	{"conv2d_2", Shape{H: 172, W: 264, C: 1}, 7, 7, 32, 1, 3},
	{"conv2d_3", Shape{H: 172, W: 88, C: 32}, 7, 3, 1, 1, 1},
	{"conv2d_4", Shape{H: 172, W: 264, C: 8}, 5, 5, 32, 1, 3},
	{"conv2d_5", Shape{H: 172, W: 88, C: 33}, 3, 3, 1, 1, 1},
}

func benchSetup(b *testing.B, l bpLayer) (conv *Conv2D, dst, src *Tensor) {
	b.Helper()

	rng := rand.New(rand.NewPCG(1, 2))
	k := l.kh * l.kw * l.in.C

	weights := make([]float32, k*l.cout)
	for i := range weights {
		weights[i] = rng.Float32()*2 - 1
	}

	bias := make([]float32, l.cout)
	for i := range bias {
		bias[i] = rng.Float32()*2 - 1
	}

	conv, err := NewConv2D(l.in, l.kh, l.kw, l.cout, weights, bias, WithStride(l.strideH, l.strideW))
	if err != nil {
		b.Fatal(err)
	}

	src = NewTensor(l.in.H, l.in.W, l.in.C)
	for i := range src.Data {
		src.Data[i] = rng.Float32()
	}

	out := conv.OutShape()

	return conv, NewTensor(out.H, out.W, out.C), src
}

// BenchmarkConv2DWindow runs every basic-pitch convolution over a full
// 172-row window and reports multiply-accumulates per second.
func BenchmarkConv2DWindow(b *testing.B) {
	for _, l := range bpLayers {
		b.Run(l.name, func(b *testing.B) {
			conv, dst, src := benchSetup(b, l)
			macs := float64(conv.MACsPerRow() * conv.OutShape().H)

			b.ReportAllocs()

			for b.Loop() {
				if err := conv.Forward(dst, src); err != nil {
					b.Fatal(err)
				}
			}

			b.ReportMetric(macs*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
		})
	}
}

// BenchmarkConv2DRow measures one output row, the smallest step a caller
// of ForwardRows can take.
func BenchmarkConv2DRow(b *testing.B) {
	for _, l := range bpLayers {
		b.Run(l.name, func(b *testing.B) {
			conv, dst, src := benchSetup(b, l)
			rows := conv.OutShape().H
			macs := float64(conv.MACsPerRow())

			b.ReportAllocs()

			y := 0
			for b.Loop() {
				if err := conv.ForwardRows(dst, src, y, y+1); err != nil {
					b.Fatal(err)
				}

				y = (y + 1) % rows
			}

			b.ReportMetric(macs*float64(b.N)/b.Elapsed().Seconds()/1e9, "GMAC/s")
		})
	}
}
