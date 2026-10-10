package basicpitch

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"slices"
	"testing"
)

// pinnedModelSHA256 is the SHA-256 recorded in testdata/README.md.
const pinnedModelSHA256 = "2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec"

// Front-end parameters of docs/port-spec.md §3.1.
const (
	cqtSampleRate = 22050.0
	cqtFFTLen     = 256
	cqtBinsPerOct = 36
	cqtNBins      = 309
)

func loadTestWeights(t *testing.T) map[string]weightTensor {
	t.Helper()

	w, err := loadWeights()
	if err != nil {
		t.Fatalf("loadWeights: %v", err)
	}

	return w
}

func TestWeightManifestShapes(t *testing.T) {
	t.Parallel()

	// HWIO kernel shapes and bias lengths of docs/port-spec.md §4, then the
	// front-end constants of §3.
	want := []struct {
		name  string
		shape []int
	}{
		{weightConv2d1Kernel, []int{3, 39, 8, 8}},
		{weightConv2d1Bias, []int{8}},
		{weightContoursReducedKernel, []int{5, 5, 8, 1}},
		{weightContoursReducedBias, []int{1}},
		{weightConv2d2Kernel, []int{7, 7, 1, 32}},
		{weightConv2d2Bias, []int{32}},
		{weightConv2d3Kernel, []int{7, 3, 32, 1}},
		{weightConv2d3Bias, []int{1}},
		{weightConv2d4Kernel, []int{5, 5, 8, 32}},
		{weightConv2d4Bias, []int{32}},
		{weightConv2d5Kernel, []int{3, 3, 33, 1}},
		{weightConv2d5Bias, []int{1}},
		{weightCQTKernelReal, []int{36, 256}},
		{weightCQTKernelImag, []int{36, 256}},
		{weightCQTLowpass, []int{256}},
		{weightCQTSqrtLengths, []int{309}},
		{weightBatchNormScale, []int{1}},
		{weightBatchNormOffset, []int{1}},
	}

	if len(weightManifest) != len(want) {
		t.Fatalf("manifest has %d entries, want %d", len(weightManifest), len(want))
	}

	offset := 0

	for i, w := range want {
		e := weightManifest[i]
		if e.Name != w.name || !slices.Equal(e.Shape, w.shape) {
			t.Errorf("entry %d: %s %v, want %s %v", i, e.Name, e.Shape, w.name, w.shape)
		}

		n := 1
		for _, d := range w.shape {
			n *= d
		}

		if e.Offset != offset || e.Length != n {
			t.Errorf("%s: offset %d length %d, want %d %d", e.Name, e.Offset, e.Length, offset, n)
		}

		offset += n
	}

	if offset != weightsTotal || len(weightsBin) != 4*weightsTotal {
		t.Errorf("total %d values, weightsTotal %d, weights.bin %d bytes; want %d bytes",
			offset, weightsTotal, len(weightsBin), 4*offset)
	}
}

func TestWeightParameterCounts(t *testing.T) {
	t.Parallel()

	w := loadTestWeights(t)

	count := func(names ...string) int {
		n := 0
		for _, name := range names {
			n += len(w[name].Data)
		}

		return n
	}

	// The six layers: 7,496 + 201 + 1,600 + 673 + 6,432 + 298 = 16,700.
	layers := count(
		weightConv2d1Kernel, weightConv2d1Bias,
		weightContoursReducedKernel, weightContoursReducedBias,
		weightConv2d2Kernel, weightConv2d2Bias,
		weightConv2d3Kernel, weightConv2d3Bias,
		weightConv2d4Kernel, weightConv2d4Bias,
		weightConv2d5Kernel, weightConv2d5Bias,
	)
	if layers != 16700 {
		t.Errorf("layer parameters = %d, want 16700", layers)
	}

	// Keras counts 16,782 trainable parameters (port-spec §4): the 16,700
	// above, gamma and beta of the CQT BatchNorm (surviving as its scale and
	// offset, 2), and gamma and beta of the 8- and 32-channel BatchNorms after
	// conv2d_1 and conv2d_4 (2*8 + 2*32 = 80), which tf2onnx folded into those
	// kernels and biases.
	bn := count(weightBatchNormScale, weightBatchNormOffset)

	const foldedBNGammaBeta = 2*8 + 2*32
	if got := layers + bn + foldedBNGammaBeta; got != 16782 {
		t.Errorf("trainable parameters = %d + %d + %d = %d, want 16782", layers, bn, foldedBNGammaBeta, got)
	}

	// Front-end constants: 2 kernels of 36x256, a 256-tap low-pass and 309
	// sqrt-length factors = 18,997, plus the 2 BatchNorm values.
	front := count(weightCQTKernelReal, weightCQTKernelImag, weightCQTLowpass, weightCQTSqrtLengths)
	if front != 2*36*256+256+309 {
		t.Errorf("front-end constants = %d, want 18997", front)
	}

	if total := layers + front + bn; total != weightsTotal {
		t.Errorf("total = %d, want weightsTotal %d", total, weightsTotal)
	}

	if weightsTotal != 35699 {
		t.Errorf("weightsTotal = %d, want 16700 + 18997 + 2 = 35699", weightsTotal)
	}
}

func TestWeightsModelSHA256(t *testing.T) {
	t.Parallel()

	data, err := os.ReadFile("../testdata/nmp.onnx")
	if err != nil {
		t.Fatal(err)
	}

	sum := sha256.Sum256(data)
	got := hex.EncodeToString(sum[:])

	if got != pinnedModelSHA256 {
		t.Errorf("SHA-256 of testdata/nmp.onnx = %s, want %s", got, pinnedModelSHA256)
	}

	if nmpONNXSHA256 != pinnedModelSHA256 {
		t.Errorf("weights.go records model SHA-256 %s, want %s", nmpONNXSHA256, pinnedModelSHA256)
	}
}

// hwio returns kernel[h][w][i][o] of an HWIO tensor.
func hwio(k weightTensor, h, w, i, o int) float32 {
	s := k.Shape

	return k.Data[((h*s[1]+w)*s[2]+i)*s[3]+o]
}

func TestWeightSpotValues(t *testing.T) {
	t.Parallel()

	w := loadTestWeights(t)

	// Expected values were read from nmp.onnx with the reference Python onnx
	// package. Kernels: oihw[0][0][0][0..3], which is hwio[0][0..3][0][0].
	kernels := []struct {
		name string
		want []float32
	}{
		{weightConv2d1Kernel, []float32{0.02223546, 0.01556065, -0.01682326, 0.00577071}},
		{weightContoursReducedKernel, []float32{0.01667988, -0.00744746, 0.00815329, 0.01231149}},
		{weightConv2d2Kernel, []float32{-0.11708884, -0.11443283, 0.03919401, 0.1296901}},
		{weightConv2d3Kernel, []float32{0.00148134, -0.00502468, 0.00196456, -0.00447498}},
		{weightConv2d4Kernel, []float32{-0.21944936, -0.18605125, 0.12032231, -0.1096387}},
		{weightConv2d5Kernel, []float32{0.02961926, 0.24009527, 0.01838084, 0.02228916}},
	}

	for _, k := range kernels {
		for wi, want := range k.want {
			if got := hwio(w[k.name], 0, wi, 0, 0); !close32(got, want, 1e-7) {
				t.Errorf("%s hwio[0][%d][0][0] = %.9g, want %.9g", k.name, wi, got, want)
			}
		}
	}

	prefixes := []struct {
		name string
		want []float32
	}{
		{weightConv2d1Bias, []float32{0.7729597, -1.2325631, -1.2648083, 0.98053026}},
		{weightConv2d2Bias, []float32{0.3099294, 0.5528378, 0.31296661, 0.4430207}},
		{weightConv2d4Bias, []float32{-0.5544527, 0.45351636, 1.3920046, 2.681076}},
		{weightContoursReducedBias, []float32{-0.3945435}},
		{weightConv2d3Bias, []float32{-0.6187599}},
		{weightConv2d5Bias, []float32{-0.36014846}},
		{weightCQTLowpass, []float32{-0.00011111, -0.00011279, 0.00011467, 0.0001172}},
		{weightCQTSqrtLengths, []float32{203.08865, 201.14174, 199.21596, 197.30687}},
		{weightBatchNormScale, []float32{2.480741}},
		{weightBatchNormOffset, []float32{-0.8769183}},
	}

	for _, p := range prefixes {
		for i, want := range p.want {
			// The reference values are printed to 8 significant digits.
			if got := w[p.name].Data[i]; !close32(got, want, 1e-7*max(1, math.Abs(float64(want)))) {
				t.Errorf("%s[%d] = %.9g, want %.9g", p.name, i, got, want)
			}
		}
	}
}

// close32 reports whether got and want differ by at most tol.
func close32(got, want float32, tol float64) bool {
	return math.Abs(float64(got)-float64(want)) <= tol
}

// cqtKernel computes the complex top-octave kernel of bin k as nnAudio's
// create_cqt_kernels does (port-spec §3.1): its length l, its start in the
// 256-sample frame and its L1-normalised values.
func cqtKernel(k int) (l, start int, re, im []float64) {
	q := 1 / (math.Pow(2, 1.0/cqtBinsPerOct) - 1)
	fminT := 7040 * math.Pow(2, -15.0/cqtBinsPerOct)
	f := fminT * math.Pow(2, float64(k)/cqtBinsPerOct)

	lf := math.Ceil(q * cqtSampleRate / f)
	l = int(lf)
	start = int(math.Ceil(cqtFFTLen/2-lf/2)) - l%2

	re = make([]float64, l)
	im = make([]float64, l)
	norm := 0.0
	n0 := math.Floor(-lf / 2)

	for j := range l {
		win := 0.5 - 0.5*math.Cos(2*math.Pi*float64(j)/lf) // periodic Hann
		phase := 2 * math.Pi * f * (n0 + float64(j)) / cqtSampleRate
		re[j] = win * math.Cos(phase) / lf
		im[j] = win * math.Sin(phase) / lf
		norm += math.Hypot(re[j], im[j])
	}

	for j := range l {
		re[j] /= norm
		im[j] /= norm
	}

	return l, start, re, im
}

func TestCQTKernelsMatchFormula(t *testing.T) {
	t.Parallel()

	w := loadTestWeights(t)
	kr, ki := w[weightCQTKernelReal], w[weightCQTKernelImag]

	maxDiff := 0.0

	for k := range cqtBinsPerOct {
		l, start, re, im := cqtKernel(k)
		row := k * cqtFFTLen

		l1 := 0.0

		for n := range cqtFFTLen {
			r, i := float64(kr.Data[row+n]), float64(ki.Data[row+n])
			l1 += math.Hypot(r, i)

			if n < start || n >= start+l {
				if r != 0 || i != 0 {
					t.Errorf("bin %d: sample %d outside support [%d,%d) is (%g, %g)", k, n, start, start+l, r, i)
				}

				continue
			}

			maxDiff = max(maxDiff, math.Abs(r-re[n-start]), math.Abs(i-im[n-start]))
		}

		// The periodic Hann window is zero at its first sample, so the first
		// support sample is exactly zero; the second and the last are not.
		// The nonzero support is therefore [start+1, start+l).
		first, second, last := row+start, row+start+1, row+start+l-1
		if kr.Data[first] != 0 || ki.Data[first] != 0 ||
			kr.Data[second] == 0 && ki.Data[second] == 0 ||
			kr.Data[last] == 0 && ki.Data[last] == 0 {
			t.Errorf("bin %d: support of length %d does not start at %d", k, l, start)
		}

		if math.Abs(l1-1) > 1e-5 {
			t.Errorf("bin %d: L1 norm %.9g, want 1", k, l1)
		}
	}

	// Measured: 9.2e-10, the float32 rounding of values below 0.02 (half an
	// ulp there is 9.3e-10).
	t.Logf("max |ONNX - formula| over the CQT kernels: %.3g", maxDiff)

	if maxDiff > 1e-9 {
		t.Errorf("CQT kernels differ from the formula by %.3g, want <= 1e-9", maxDiff)
	}
}

func TestCQTLowpass(t *testing.T) {
	t.Parallel()

	h := loadTestWeights(t)[weightCQTLowpass].Data

	sum := 0.0
	for n, v := range h {
		sum += float64(v)

		// Even-length firwin2 designs are type II linear phase: exactly
		// symmetric.
		if v != h[len(h)-1-n] {
			t.Errorf("low-pass not symmetric: h[%d] = %g, h[%d] = %g", n, v, len(h)-1-n, h[len(h)-1-n])
		}
	}

	// firwin2's Hamming-windowed design does not hit unit DC gain exactly:
	// the taps sum to 0.99977926, 2.2e-4 below 1. Pin the measured value.
	t.Logf("low-pass DC gain: %.9g", sum)

	if math.Abs(sum-0.99977926) > 1e-7 {
		t.Errorf("low-pass DC gain %.9g, want 0.99977926", sum)
	}
}

func TestCQTSqrtLengths(t *testing.T) {
	t.Parallel()

	s := loadTestWeights(t)[weightCQTSqrtLengths].Data
	q := 1 / (math.Pow(2, 1.0/cqtBinsPerOct) - 1)

	maxRel := 0.0

	for k := range cqtNBins {
		f := 27.5 * math.Pow(2, float64(k)/cqtBinsPerOct)
		want := math.Sqrt(math.Ceil(q * cqtSampleRate / f))
		maxRel = max(maxRel, math.Abs(float64(s[k])-want)/want)
	}

	// Measured: 5.8e-8, within float32 rounding (half an ulp is 2^-24 = 6e-8
	// relative).
	t.Logf("max relative |ONNX - formula| over the sqrt lengths: %.3g", maxRel)

	if maxRel > 0x1p-24 {
		t.Errorf("sqrt lengths differ from the formula by %.3g relative, want <= 2^-24", maxRel)
	}
}

func TestLoadWeightsCopies(t *testing.T) {
	t.Parallel()

	a := loadTestWeights(t)
	a[weightConv2d1Bias].Data[0] = 42
	a[weightConv2d1Bias].Shape[0] = 42

	b := loadTestWeights(t)
	if b[weightConv2d1Bias].Data[0] == 42 || weightManifest[1].Shape[0] != 8 {
		t.Error("loadWeights returned shared storage")
	}
}
