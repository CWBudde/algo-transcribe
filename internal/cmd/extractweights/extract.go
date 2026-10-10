package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"

	"github.com/cwbudde/algo-transcribe/internal/onnx"
)

// modelSHA256 pins the basic-pitch v0.4.0 nmp.onnx (testdata/README.md).
const modelSHA256 = "2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec"

// Output file names inside the -out directory.
const (
	binFile = "weights.bin"
	srcFile = "weights.go"
)

// namedTensor is one tensor of weights.bin: its manifest name, the Go
// constant that names it in weights.go, its shape and its values. Kernels are
// already in HWIO.
type namedTensor struct {
	name  string
	ident string
	shape []int
	data  []float32
}

// extract turns the bytes of nmp.onnx into the contents of weights.bin and
// weights.go. It is a pure function: the same model always yields the same
// bytes.
func extract(model []byte) (bin, src []byte, err error) {
	sum := sha256.Sum256(model)
	if got := hex.EncodeToString(sum[:]); got != modelSHA256 {
		return nil, nil, fmt.Errorf("extractweights: model SHA-256 is %s, want %s", got, modelSHA256)
	}

	m, err := onnx.Parse(model)
	if err != nil {
		return nil, nil, fmt.Errorf("extractweights: %w", err)
	}

	tensors, err := mapModel(m)
	if err != nil {
		return nil, nil, err
	}

	src, err = generateSource(tensors, modelSHA256)
	if err != nil {
		return nil, nil, err
	}

	return encodeBin(tensors), src, nil
}

// encodeBin concatenates the tensors as little-endian float32, copying the
// bits unchanged.
func encodeBin(tensors []namedTensor) []byte {
	out := make([]byte, 0, 4*total(tensors))
	for _, t := range tensors {
		for _, v := range t.data {
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(v))
		}
	}

	return out
}

// oihwToHWIO reorders an ONNX Conv kernel of shape [o,i,h,w] into the HWIO
// layout of package nn: hwio[h][w][i][o] = oihw[o][i][h][w].
func oihwToHWIO(src []float32, o, i, h, w int) ([]float32, error) {
	if len(src) != o*i*h*w {
		return nil, fmt.Errorf("extractweights: OIHW kernel has %d values, want %d*%d*%d*%d", len(src), o, i, h, w)
	}

	dst := make([]float32, len(src))

	for oo := range o {
		for ii := range i {
			for hh := range h {
				for ww := range w {
					dst[((hh*w+ww)*i+ii)*o+oo] = src[((oo*i+ii)*h+hh)*w+ww]
				}
			}
		}
	}

	return dst, nil
}
