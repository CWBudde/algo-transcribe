package fixture

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"slices"
	"strings"
)

// Float32 reads the float32 array called name and returns its data in C order
// together with a copy of its shape.
func (c *Clip) Float32(name string) (data []float32, shape []int, err error) {
	raw, a, err := c.arrayBytes(name, DTypeFloat32, 4)
	if err != nil {
		return nil, nil, err
	}

	data = make([]float32, len(raw)/4)
	for i := range data {
		data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}

	return data, slices.Clone(a.Shape), nil
}

// Float64 reads the float64 array called name and returns its data in C order
// together with a copy of its shape.
func (c *Clip) Float64(name string) (data []float64, shape []int, err error) {
	raw, a, err := c.arrayBytes(name, DTypeFloat64, 8)
	if err != nil {
		return nil, nil, err
	}

	data = make([]float64, len(raw)/8)
	for i := range data {
		data[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[8*i:]))
	}

	return data, slices.Clone(a.Shape), nil
}

// Window returns window w of a per-window float32 array, that is the slice
// along its first axis, and the shape without the leading dimension. It reads
// the whole array on every call; loop over windows with [Clip.Float32] instead
// when every window is needed.
func (c *Clip) Window(name string, w int) ([]float32, []int, error) {
	data, shape, err := c.Float32(name)
	if err != nil {
		return nil, nil, err
	}

	if len(shape) < 2 {
		return nil, nil, fmt.Errorf("fixture: array %q has shape %v, want at least 2 dimensions", name, shape)
	}

	if w < 0 || w >= shape[0] {
		return nil, nil, fmt.Errorf("fixture: array %q: window %d out of range [0, %d)", name, w, shape[0])
	}

	stride := len(data) / shape[0]

	return data[w*stride : (w+1)*stride : (w+1)*stride], shape[1:], nil
}

// HarmonicStack is the name of the harmonic-stack array. It is stored with
// axes (window, time, harmonic, freq) because that layout compresses much
// better than the model's NHWC (time, freq, harmonic).
const HarmonicStack = "harmonic_stack"

// harmonicStackAxes is the stored axis order of [HarmonicStack].
var harmonicStackAxes = []string{"window", "time", "harmonic", "freq"}

// HarmonicStackNHWC returns window w of the harmonic stack in the model's
// NHWC layout: shape (time, freq, harmonic), that is (172, 264, 8) for real
// fixtures, with element (t, f, h) at index (t*freq+f)*harmonic+h. The shape
// is returned alongside. It reads the whole array on every call.
func (c *Clip) HarmonicStackNHWC(w int) ([]float32, []int, error) {
	if a, ok := c.Manifest.Arrays[HarmonicStack]; ok && !slices.Equal(a.Axes, harmonicStackAxes) {
		return nil, nil, fmt.Errorf("fixture: array %q has axes %q, want %q", HarmonicStack, a.Axes, harmonicStackAxes)
	}

	win, shape, err := c.Window(HarmonicStack, w)
	if err != nil {
		return nil, nil, err
	}

	nT, nH, nF := shape[0], shape[1], shape[2]
	out := make([]float32, len(win))

	for t := range nT {
		for h := range nH {
			src := win[(t*nH+h)*nF : (t*nH+h+1)*nF]
			for f, v := range src {
				out[(t*nF+f)*nH+h] = v
			}
		}
	}

	return out, []int{nT, nF, nH}, nil
}

// arrayBytes reads, decompresses and verifies the bytes of an array.
func (c *Clip) arrayBytes(name, dtype string, elemSize int) ([]byte, Array, error) {
	a, ok := c.Manifest.Arrays[name]
	if !ok {
		return nil, Array{}, fmt.Errorf("fixture: unknown array %q in clip %q", name, c.Manifest.Clip)
	}

	if a.DType != dtype {
		return nil, Array{}, fmt.Errorf("fixture: array %q has dtype %s, not %s", name, a.DType, dtype)
	}

	count, err := elements(a.Shape)
	if err != nil {
		return nil, Array{}, fmt.Errorf("fixture: array %q: %w", name, err)
	}

	raw, err := c.readFile(a.File)
	if err != nil {
		return nil, Array{}, err
	}

	if strings.HasSuffix(a.File, ".gz") {
		raw, err = gunzip(raw)
		if err != nil {
			return nil, Array{}, fmt.Errorf("fixture: array %q: gzip: %w", name, err)
		}
	}

	if want := count * elemSize; len(raw) != want {
		return nil, Array{}, fmt.Errorf("fixture: array %q: size %d bytes, shape %v needs %d", name, len(raw), a.Shape, want)
	}

	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != a.SHA256 {
		return nil, Array{}, fmt.Errorf("fixture: array %q: sha256 %s, manifest says %s", name, got, a.SHA256)
	}

	return raw, a, nil
}

// elements returns the product of shape and rejects counts that overflow.
func elements(shape []int) (int, error) {
	n := 1

	for _, d := range shape {
		if d <= 0 || n > math.MaxInt32/d {
			return 0, fmt.Errorf("shape %v is invalid or too large", shape)
		}

		n *= d
	}

	return n, nil
}

func gunzip(b []byte) ([]byte, error) {
	zr, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	out, err := io.ReadAll(zr)
	if err != nil {
		return nil, err
	}

	err = zr.Close()
	if err != nil {
		return nil, err
	}

	return out, nil
}
