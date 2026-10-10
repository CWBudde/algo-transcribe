package nn

import (
	"errors"
	"fmt"
	"math"
)

// ReLU applies max(v, 0) to every element of x in place.
//
// It only rewrites negative values, so NaN passes through unchanged (a
// comparison with NaN is false) and -0 stays -0. Neither occurs in a model
// that is fed finite audio.
func ReLU(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

// Sigmoid applies 1/(1+exp(-v)) to every element of x in place.
//
// The exponential is evaluated in float64 and the result rounded once to
// float32, so the outputs match onnxruntime well inside the 1e-4 parity
// tolerance of docs/port-spec.md §6. Large magnitudes saturate cleanly:
// exp overflows to +Inf for very negative v, giving exactly 0, and
// underflows to 0 for very positive v, giving exactly 1; ±Inf map to 0 and
// 1 as well. NaN stays NaN.
func Sigmoid(x []float32) {
	for i, v := range x {
		x[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
}

// ConcatChannels concatenates srcs along the channel (last) axis into dst.
// It is ConcatChannelsRows(dst, 0, dst.H, srcs...).
//
// basic-pitch concatenates the note posteriorgram (172,88,1) with the
// output of conv2d_4 (172,88,32) into the (172,88,33) input of the onset
// convolution (docs/port-spec.md §4), in that order.
func ConcatChannels(dst *Tensor, srcs ...*Tensor) error {
	if err := dst.check("concat: dst"); err != nil {
		return err
	}

	return ConcatChannelsRows(dst, 0, dst.H, srcs...)
}

// ConcatChannelsRows concatenates rows [r0, r1) of srcs along the channel
// (last) axis into the same rows of dst. Every source must have the H and
// W of dst, and their channel counts must add up to dst.C; the channels of
// srcs[0] come first. Other rows of dst are left untouched, so a caller
// can split the concatenation into steps of any size. It does not
// allocate.
func ConcatChannelsRows(dst *Tensor, r0, r1 int, srcs ...*Tensor) error {
	if err := dst.check("concat: dst"); err != nil {
		return err
	}

	c := 0

	for i, s := range srcs {
		if err := s.check("concat: src"); err != nil {
			return fmt.Errorf("%w (srcs[%d])", err, i)
		}

		if s.H != dst.H || s.W != dst.W {
			return fmt.Errorf("nn: concat: srcs[%d] shape %v does not match dst %v in H and W", i, s.Shape, dst.Shape)
		}

		c += s.C
	}

	if c != dst.C {
		return fmt.Errorf("nn: concat: sources have %d channels, dst has %d", c, dst.C)
	}

	if r0 < 0 || r1 > dst.H || r0 > r1 {
		return fmt.Errorf("nn: concat: rows [%d,%d) outside [0,%d)", r0, r1, dst.H)
	}

	for p := r0 * dst.W; p < r1*dst.W; p++ {
		d := dst.Data[p*dst.C : (p+1)*dst.C]
		off := 0

		for _, s := range srcs {
			off += copy(d[off:], s.Data[p*s.C:(p+1)*s.C])
		}
	}

	return nil
}

// BatchNorm holds the parameters of an inference-mode batch normalisation
// over cout channels:
//
//	y = Gamma*(x-Mean)/sqrt(Variance+Epsilon) + Beta
//
// All four slices have one element per channel. Epsilon is the model's
// value; Keras defaults to 1e-3, but the caller passes whatever the
// exported graph uses rather than relying on a default here.
type BatchNorm struct {
	Gamma, Beta, Mean, Variance []float32
	Epsilon                     float32
}

// FoldBatchNorm folds bn into the preceding convolution's weights and bias
// in place, so that the convolution alone computes BN(conv(x)+bias).
//
// weights is in HWIO layout, so the output channel varies fastest:
// weights[i] belongs to channel i % cout with cout = len(bn.Gamma). For
// every channel
//
//	scale = Gamma/sqrt(Variance+Epsilon)
//	w     = w*scale
//	bias  = (bias-Mean)*scale + Beta
//
// with scale and bias evaluated in float64 and rounded once to float32. A
// convolution without bias passes a zeroed bias slice of length cout.
//
// All four slices of bn and bias must have the same length cout > 0, len(weights)
// must be a positive multiple of cout, and Variance+Epsilon must be
// positive. On error nothing is modified. It does not allocate.
func FoldBatchNorm(weights, bias []float32, bn BatchNorm) error {
	cout := len(bn.Gamma)
	if cout == 0 {
		return errors.New("nn: fold batchnorm: no channels")
	}

	if len(bn.Beta) != cout || len(bn.Mean) != cout || len(bn.Variance) != cout {
		return fmt.Errorf("nn: fold batchnorm: parameter lengths gamma %d, beta %d, mean %d, variance %d differ",
			cout, len(bn.Beta), len(bn.Mean), len(bn.Variance))
	}

	if len(bias) != cout {
		return fmt.Errorf("nn: fold batchnorm: %d biases, want %d", len(bias), cout)
	}

	if len(weights) == 0 || len(weights)%cout != 0 {
		return fmt.Errorf("nn: fold batchnorm: %d weights are not a positive multiple of %d channels", len(weights), cout)
	}

	for co, v := range bn.Variance {
		// The negated comparison also rejects NaN.
		if !(float64(v)+float64(bn.Epsilon) > 0) {
			return fmt.Errorf("nn: fold batchnorm: channel %d: variance %g + epsilon %g is not positive", co, v, bn.Epsilon)
		}
	}

	for co := range cout {
		scale := float64(bn.Gamma[co]) / math.Sqrt(float64(bn.Variance[co])+float64(bn.Epsilon))

		for i := co; i < len(weights); i += cout {
			weights[i] = float32(float64(weights[i]) * scale)
		}

		bias[co] = float32((float64(bias[co])-float64(bn.Mean[co]))*scale + float64(bn.Beta[co]))
	}

	return nil
}
