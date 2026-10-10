package main

import (
	"bytes"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/algo-transcribe/internal/onnx"
)

const (
	modelPath    = "../../../testdata/nmp.onnx"
	committedDir = "../../../basicpitch"
)

func readModel(t *testing.T) []byte {
	t.Helper()

	data, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}

	return data
}

func parseModel(t *testing.T) *onnx.Model {
	t.Helper()

	m, err := onnx.Parse(readModel(t))
	if err != nil {
		t.Fatal(err)
	}

	return m
}

// TestExtractMatchesCommitted proves that regenerating from the committed
// model reproduces the committed weights.bin and weights.go byte for byte.
func TestExtractMatchesCommitted(t *testing.T) {
	t.Parallel()

	bin, src, err := extract(readModel(t))
	if err != nil {
		t.Fatal(err)
	}

	for _, f := range []struct {
		name string
		got  []byte
	}{{binFile, bin}, {srcFile, src}} {
		want, err := os.ReadFile(filepath.Join(committedDir, f.name))
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(f.got, want) {
			t.Errorf("regenerated %s (%d bytes) differs from the committed one (%d bytes); run go generate ./basicpitch",
				f.name, len(f.got), len(want))
		}
	}
}

func TestRunWritesFiles(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	if err := run(modelPath, dir); err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{binFile, srcFile} {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}

		want, err := os.ReadFile(filepath.Join(committedDir, name))
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.Equal(got, want) {
			t.Errorf("run wrote a %s that differs from the committed one", name)
		}
	}

	if err := run(filepath.Join(dir, "missing.onnx"), dir); err == nil {
		t.Error("run on a missing model succeeded")
	}
}

func TestExtractRejectsOtherModels(t *testing.T) {
	t.Parallel()

	model := readModel(t)

	flipped := slices.Clone(model)
	flipped[len(flipped)/2] ^= 1

	for _, tc := range []struct {
		name  string
		model []byte
	}{
		{"one bit flipped", flipped},
		{"truncated", model[:len(model)-1]},
		{"empty", nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, _, err := extract(tc.model)
			if err == nil || !strings.Contains(err.Error(), "SHA-256") {
				t.Errorf("extract: err = %v, want a SHA-256 mismatch", err)
			}
		})
	}
}

// TestMapModelIgnoresInitializerNamesAndOrder shuffles and renames every
// initializer (rewriting the node inputs to match) and checks that the
// extracted tensors do not change: the mapping follows node inputs only.
func TestMapModelIgnoresInitializerNamesAndOrder(t *testing.T) {
	t.Parallel()

	want, err := mapModel(parseModel(t))
	if err != nil {
		t.Fatal(err)
	}

	for seed := range uint64(3) {
		m := parseModel(t)
		g := &m.Graph
		rng := rand.New(rand.NewPCG(seed, 0x5eed))

		rng.Shuffle(len(g.Initializers), func(i, j int) {
			g.Initializers[i], g.Initializers[j] = g.Initializers[j], g.Initializers[i]
		})

		rename := map[string]string{}

		for i := range g.Initializers {
			name := fmt.Sprintf("w%03d", rng.IntN(1000)*1000+i)
			rename[g.Initializers[i].Name] = name
			g.Initializers[i].Name = name
		}

		for i := range g.Nodes {
			for j, in := range g.Nodes[i].Inputs {
				if name, ok := rename[in]; ok {
					g.Nodes[i].Inputs[j] = name
				}
			}
		}

		got, err := mapModel(m)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}

		if !reflect.DeepEqual(got, want) {
			t.Errorf("seed %d: shuffled and renamed initializers change the extracted tensors", seed)
		}
	}
}

// convByKernel returns the Conv whose kernel initializer has the given dims.
func convByKernel(t *testing.T, g *onnx.Graph, dims ...int64) *onnx.Node {
	t.Helper()

	for i := range g.Nodes {
		n := &g.Nodes[i]
		if n.OpType != "Conv" || len(n.Inputs) < 2 {
			continue
		}

		if k := g.Initializer(n.Inputs[1]); k != nil && slices.Equal(k.Dims, dims) {
			return n
		}
	}

	t.Fatalf("no Conv with kernel dims %v", dims)

	return nil
}

// producerOf returns the producer of n's first input.
func producerOf(t *testing.T, g *onnx.Graph, n *onnx.Node) *onnx.Node {
	t.Helper()

	p := g.Producer(n.Inputs[0])
	if p == nil {
		t.Fatalf("%s %q has no producer", n.OpType, n.Name)
	}

	return p
}

func TestMapModelRejectsMismatches(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		mutate  func(t *testing.T, g *onnx.Graph)
		wantErr string
	}{
		{
			name: "conv2d_2 stride",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()
				convByKernel(t, g, 32, 1, 7, 7).Attr("strides").Ints = []int64{1, 1}
			},
			wantErr: "strides [1 1], want [1 3]",
		},
		{
			name: "conv2d_4 pads",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()
				convByKernel(t, g, 32, 8, 5, 5).Attr("pads").Ints = []int64{2, 2, 2, 2}
			},
			wantErr: "pads [2 2 2 2], want [2 1 2 1]",
		},
		{
			name: "conv2d_1 dilations",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()
				convByKernel(t, g, 8, 8, 3, 39).Attr("dilations").Ints = []int64{2, 1}
			},
			wantErr: "dilations [2 1]",
		},
		{
			name: "contours-reduced group",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()
				convByKernel(t, g, 1, 8, 5, 5).Attr("group").I = 2
			},
			wantErr: "group 2",
		},
		{
			name: "conv2d_5 auto_pad",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				n := convByKernel(t, g, 1, 33, 3, 3)
				n.Attributes = append(n.Attributes, onnx.Attribute{
					Name: "auto_pad", Type: onnx.AttributeString, S: []byte("SAME_UPPER"),
				})
			},
			wantErr: `auto_pad "SAME_UPPER"`,
		},
		{
			name: "conv2d_3 unknown attribute",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				n := convByKernel(t, g, 1, 32, 7, 3)
				n.Attributes = append(n.Attributes, onnx.Attribute{Name: "fancy", Type: onnx.AttributeInt})
			},
			wantErr: `unexpected attribute "fancy"`,
		},
		{
			name: "conv2d_3 kernel shape",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				n := convByKernel(t, g, 1, 32, 7, 3)
				g.Initializer(n.Inputs[1]).Dims = []int64{1, 32, 3, 7}
			},
			wantErr: "has dims [1 32 3 7], want [1 32 7 3]",
		},
		{
			name: "conv2d_1 kernel data type",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				n := convByKernel(t, g, 8, 8, 3, 39)
				g.Initializer(n.Inputs[1]).DataType = onnx.DataTypeDouble
			},
			wantErr: "want float",
		},
		{
			name: "contour sigmoid missing",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				sig := producerOf(t, g, g.Producer(outContour))
				if sig.OpType != "Sigmoid" {
					t.Fatalf("contour output reads %s, want Sigmoid", sig.OpType)
				}

				sig.OpType = "Tanh"
			},
			wantErr: "stops at Tanh",
		},
		{
			name: "conv2d_1 relu replaced by sigmoid",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()
				producerOf(t, g, convByKernel(t, g, 1, 8, 5, 5)).OpType = "Sigmoid"
			},
			wantErr: "want exactly one Relu",
		},
		{
			name: "onset concat inputs swapped",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				concat := producerOf(t, g, convByKernel(t, g, 1, 33, 3, 3))
				concat.Inputs[0], concat.Inputs[1] = concat.Inputs[1], concat.Inputs[0]
			},
			wantErr: "onset concat input 0",
		},
		{
			name: "conv2d_4 reads another tensor",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				conv4 := convByKernel(t, g, 32, 8, 5, 5)
				conv4.Inputs[0] = producerOf(t, g, convByKernel(t, g, 1, 8, 5, 5)).Outputs[0]
			},
			wantErr: "want the same harmonic stack",
		},
		{
			name: "conv2d_2 reads conv2d_1 instead of the contour",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				conv2 := convByKernel(t, g, 32, 1, 7, 7)
				conv2.Inputs[0] = producerOf(t, g, convByKernel(t, g, 1, 8, 5, 5)).Outputs[0]
			},
			wantErr: "conv2d_2 input",
		},
		{
			name: "graph output renamed",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				g.Outputs[1].Name = "renamed"
			},
			wantErr: `no output "StatefulPartitionedCall:1"`,
		},
		{
			name: "imaginary CQT conv not negated",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				for i := range g.Nodes {
					if g.Nodes[i].OpType == "Neg" {
						g.Nodes[i].OpType = "Identity"

						return
					}
				}
			},
			wantErr: "uses a second kernel initializer",
		},
		{
			name: "CQT bias nonzero",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				b := g.Initializer(convByKernel(t, g, 36, 1, 1, 256).Inputs[2])
				b.RawData = slices.Clone(b.RawData)
				b.RawData[3] = 0x3f // 0.5 in the first float
			},
			wantErr: "nonzero bias",
		},
		{
			name: "low-pass stride",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()
				convByKernel(t, g, 1, 1, 1, 256).Attr("strides").Ints = []int64{1, 1}
			},
			wantErr: "low-pass",
		},
		{
			name: "sqrt lengths missing",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				for i := range g.Initializers {
					if slices.Equal(g.Initializers[i].Dims, []int64{309, 1, 1}) {
						g.Initializers[i].Dims = []int64{309, 1}
					}
				}
			},
			wantErr: "sqrt lengths",
		},
		{
			name: "harmonic stack channels swapped",
			mutate: func(t *testing.T, g *onnx.Graph) {
				t.Helper()

				stack := producerOf(t, g, producerOf(t, g, producerOf(t, g, convByKernel(t, g, 8, 8, 3, 39))))
				if stack.OpType != "Concat" {
					t.Fatalf("conv2d_1 reads %s, want Concat", stack.OpType)
				}

				stack.Inputs[0], stack.Inputs[1] = stack.Inputs[1], stack.Inputs[0]
			},
			wantErr: "harmonic stack input 1 is not produced by an Add",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := parseModel(t)
			tc.mutate(t, &m.Graph)

			_, err := mapModel(m)
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("mapModel: err = %v, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestOIHWToHWIO(t *testing.T) {
	t.Parallel()

	const o, i, h, w = 2, 3, 2, 4

	src := make([]float32, o*i*h*w)
	for n := range src {
		src[n] = float32(n)
	}

	dst, err := oihwToHWIO(src, o, i, h, w)
	if err != nil {
		t.Fatal(err)
	}

	for oo := range o {
		for ii := range i {
			for hh := range h {
				for ww := range w {
					got := dst[((hh*w+ww)*i+ii)*o+oo]
					if want := float32(((oo*i+ii)*h+hh)*w + ww); got != want {
						t.Errorf("hwio[%d][%d][%d][%d] = %g, want %g", hh, ww, ii, oo, got, want)
					}
				}
			}
		}
	}

	// Hand-checked corners: hwio[0][0][0][1] = oihw[1][0][0][0] = 24 and
	// hwio[1][3][2][0] = oihw[0][2][1][3] = 23.
	if dst[1] != 24 || dst[((1*w+3)*i+2)*o] != 23 {
		t.Errorf("corner values %g, %g; want 24, 23", dst[1], dst[((1*w+3)*i+2)*o])
	}

	if _, err := oihwToHWIO(src[1:], o, i, h, w); err == nil {
		t.Error("oihwToHWIO accepted a short kernel")
	}
}
