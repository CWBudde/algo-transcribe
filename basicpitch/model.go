package basicpitch

import (
	"errors"
	"fmt"

	"github.com/cwbudde/algo-transcribe/nn"
)

// Shapes of the CNN for one window (docs/port-spec.md §4). Time is the
// tensor height and frequency the width.
const (
	modelFrames      = 172 // ANNOT_N_FRAMES, frames per window
	modelContourBins = 264 // 88 semitones at 3 bins each
	modelNoteBins    = 88  // ANNOTATIONS_N_SEMITONES
	modelHarmonics   = 8   // channels of the harmonic stack
)

// modelStepMACs is the multiply-accumulate budget of one [ForwardJob] step.
// A step computes as many output rows of one layer as fit into it, but at
// least one row. conv2d_1 needs 1,976,832 MACs for one row and conv2d_4
// 563,200, so both advance a row per step; the single-channel layers, which
// run at a third of the GEMM throughput (PLAN 1.1), take 7 to 40 rows.
const modelStepMACs = 1 << 20

// ErrJobSuperseded is returned by [ForwardJob.Step] when another forward
// pass has started on the job's model since the job was (re)started. The
// model's activation buffers now belong to that pass, so the job cannot
// continue and its outputs are no longer valid, even if it had completed.
// [ForwardJob.Reset] starts the job over. Test for it with [errors.Is].
var ErrJobSuperseded = errors.New("basicpitch: model: forward job superseded by a newer one on the same model")

// modelStage is one convolution of the CNN with its activation, run over
// the rows of one window.
type modelStage struct {
	name string
	conv *nn.Conv2D
	// src is the input tensor, or nil for the harmonic stack of the job.
	src *nn.Tensor
	dst *nn.Tensor
	act func([]float32)
	// cat, if set, receives the channel concatenation of the note
	// posteriorgram and dst, row block by row block.
	cat *nn.Tensor
	// rows is the number of output rows per step.
	rows int
}

// Model is the basic-pitch CNN of docs/port-spec.md §4 with the embedded
// weights. It maps the harmonic stack of one window, (172, 264, 8) in NHWC
// with time as height, to the contour (172, 264, 1), note (172, 88, 1) and
// onset (172, 88, 1) posteriorgrams:
//
//	conv2d_1          3×39, 8→8,  BN folded, ReLU      on the harmonic stack
//	contours-reduced  5×5,  8→1,  sigmoid              → contour
//	conv2d_2          7×7,  1→32, stride (1,3), ReLU   on the contour
//	conv2d_3          7×3, 32→1,  sigmoid              → note
//	conv2d_4          5×5,  8→32, stride (1,3), BN folded, ReLU on the harmonic stack
//	concat            [note, conv2d_4] → 33 channels
//	conv2d_5          3×3, 33→1,  sigmoid              → onset
//
// The ONNX graph already has the BatchNorms of conv2d_1 and conv2d_4 folded
// into their kernels and biases, so the weights are used as extracted.
//
// A Model owns its activation buffers and the scratch space of its
// convolutions, so after [NewModel] a forward pass does not allocate. For
// the same reason a Model is not safe for concurrent use, and it runs one
// forward pass at a time: starting a new one supersedes every earlier
// [ForwardJob], finished or not (see [ErrJobSuperseded]). Use one Model per
// goroutine.
type Model struct {
	stages [6]modelStage

	conv1   *nn.Tensor // (172, 264, 8)
	contour *nn.Tensor // (172, 264, 1)
	conv2   *nn.Tensor // (172, 88, 32)
	note    *nn.Tensor // (172, 88, 1)
	conv4   *nn.Tensor // (172, 88, 32)
	concat  *nn.Tensor // (172, 88, 33)
	onset   *nn.Tensor // (172, 88, 1)

	// gen counts the forward passes started on the model; a job whose
	// generation differs has been superseded.
	gen uint64
	// job is the job [Model.Forward] reuses, so Forward does not allocate.
	job ForwardJob
	// steps is the number of steps of one forward pass.
	steps int
}

// NewModel builds the CNN from the weights embedded in the package.
func NewModel() (*Model, error) {
	w, err := loadWeights()
	if err != nil {
		return nil, fmt.Errorf("basicpitch: model: %w", err)
	}

	m := &Model{
		conv1:   nn.NewTensor(modelFrames, modelContourBins, modelHarmonics),
		contour: nn.NewTensor(modelFrames, modelContourBins, 1),
		conv2:   nn.NewTensor(modelFrames, modelNoteBins, 32),
		note:    nn.NewTensor(modelFrames, modelNoteBins, 1),
		conv4:   nn.NewTensor(modelFrames, modelNoteBins, 32),
		concat:  nn.NewTensor(modelFrames, modelNoteBins, 33),
		onset:   nn.NewTensor(modelFrames, modelNoteBins, 1),
	}

	stack := nn.Shape{H: modelFrames, W: modelContourBins, C: modelHarmonics}

	// The order is the evaluation order: every stage reads only tensors
	// that earlier stages have completed.
	specs := [len(m.stages)]struct {
		name   string
		kernel string
		bias   string
		in     nn.Shape
		src    *nn.Tensor
		dst    *nn.Tensor
		act    func([]float32)
		cat    *nn.Tensor
		stride int
	}{
		{"conv2d_1", weightConv2d1Kernel, weightConv2d1Bias, stack, nil, m.conv1, nn.ReLU, nil, 1},
		{"contours-reduced", weightContoursReducedKernel, weightContoursReducedBias, stack, m.conv1, m.contour, nn.Sigmoid, nil, 1},
		{"conv2d_2", weightConv2d2Kernel, weightConv2d2Bias, m.contour.Shape, m.contour, m.conv2, nn.ReLU, nil, 3},
		{"conv2d_3", weightConv2d3Kernel, weightConv2d3Bias, m.conv2.Shape, m.conv2, m.note, nn.Sigmoid, nil, 1},
		{"conv2d_4", weightConv2d4Kernel, weightConv2d4Bias, stack, nil, m.conv4, nn.ReLU, m.concat, 3},
		{"conv2d_5", weightConv2d5Kernel, weightConv2d5Bias, m.concat.Shape, m.concat, m.onset, nn.Sigmoid, nil, 1},
	}

	for i, s := range specs {
		k, b := w[s.kernel], w[s.bias]
		if len(k.Shape) != 4 {
			return nil, fmt.Errorf("basicpitch: model: %s: kernel shape %v is not HWIO", s.name, k.Shape)
		}

		conv, err := nn.NewConv2D(s.in, k.Shape[0], k.Shape[1], k.Shape[3], k.Data, b.Data, nn.WithStride(1, s.stride))
		if err != nil {
			return nil, fmt.Errorf("basicpitch: model: %s: %w", s.name, err)
		}

		if conv.OutShape() != s.dst.Shape {
			return nil, fmt.Errorf("basicpitch: model: %s: output shape %v, want %v", s.name, conv.OutShape(), s.dst.Shape)
		}

		rows := max(1, modelStepMACs/conv.MACsPerRow())
		m.stages[i] = modelStage{name: s.name, conv: conv, src: s.src, dst: s.dst, act: s.act, cat: s.cat, rows: rows}
		m.steps += (modelFrames + rows - 1) / rows
	}

	m.job.m = m

	return m, nil
}

// Forward runs the CNN on the harmonic stack of one window, a (172, 264, 8)
// tensor, and returns the contour (172, 264, 1), note (172, 88, 1) and onset
// (172, 88, 1) posteriorgrams.
//
// The returned tensors belong to the model: they stay valid until the next
// forward pass on m starts, and the caller must not modify them. Forward
// supersedes every earlier [ForwardJob] of m. It does not allocate.
func (m *Model) Forward(stack *nn.Tensor) (contour, note, onset *nn.Tensor, err error) {
	if err := m.job.Reset(stack); err != nil {
		return nil, nil, nil, err
	}

	for {
		done, err := m.job.Step()
		if err != nil {
			return nil, nil, nil, err
		}

		if done {
			return m.contour, m.note, m.onset, nil
		}
	}
}

// ForwardJob is a forward pass of a [Model] split into bounded steps. Each
// [ForwardJob.Step] computes a block of output rows of one layer, together
// with its activation, so a caller can interleave inference with other work
// (AGENTS.md rule 4). The layers run in the order of [Model]; a step costs
// at most about one row of conv2d_1, roughly 2 million multiply-accumulates.
//
// The job reads the harmonic stack it was started with until it is done, so
// the caller must not modify the stack in the meantime.
type ForwardJob struct {
	m     *Model
	gen   uint64
	stack *nn.Tensor
	stage int
	row   int
	steps int
}

// NewForwardJob starts a forward pass of m on the harmonic stack of one
// window, a (172, 264, 8) tensor. It supersedes every earlier job of m.
// Use [ForwardJob.Reset] to start further passes without allocating.
func (m *Model) NewForwardJob(stack *nn.Tensor) (*ForwardJob, error) {
	j := &ForwardJob{m: m}
	if err := j.Reset(stack); err != nil {
		return nil, err
	}

	return j, nil
}

// Reset starts a new forward pass of the job's model on stack, discarding
// the progress of the current one. It supersedes every other job of the
// same model and does not allocate.
func (j *ForwardJob) Reset(stack *nn.Tensor) error {
	if stack == nil {
		return errors.New("basicpitch: model: nil harmonic stack")
	}

	want := nn.Shape{H: modelFrames, W: modelContourBins, C: modelHarmonics}
	if stack.Shape != want || len(stack.Data) != want.Size() {
		return fmt.Errorf("basicpitch: model: harmonic stack has shape %v and %d elements, want %v",
			stack.Shape, len(stack.Data), want)
	}

	j.m.gen++
	j.gen = j.m.gen
	j.stack = stack
	j.stage = 0
	j.row = 0
	j.steps = 0

	return nil
}

// Step advances the forward pass by one row block of one layer and reports
// whether the pass is complete. Once it is, further calls return true
// without doing any work, as long as no other pass has started on the
// model. If one has since this job was (re)started, Step returns
// [ErrJobSuperseded], whether or not the job had completed: its outputs
// are being overwritten.
func (j *ForwardJob) Step() (done bool, err error) {
	m := j.m
	if j.gen != m.gen {
		return false, ErrJobSuperseded
	}

	if j.stage == len(m.stages) {
		return true, nil
	}

	s := &m.stages[j.stage]
	r0, r1 := j.row, min(j.row+s.rows, modelFrames)

	src := s.src
	if src == nil {
		src = j.stack
	}

	if err := s.conv.ForwardRows(s.dst, src, r0, r1); err != nil {
		return false, fmt.Errorf("basicpitch: model: %s: %w", s.name, err)
	}

	s.act(s.dst.Rows(r0, r1))

	if s.cat != nil {
		// The note posteriorgram is complete: conv2d_3 runs before conv2d_4.
		if err := nn.ConcatChannelsRows(s.cat, r0, r1, m.note, s.dst); err != nil {
			return false, fmt.Errorf("basicpitch: model: %s: %w", s.name, err)
		}
	}

	j.steps++
	j.row = r1

	if r1 == modelFrames {
		j.stage++
		j.row = 0
	}

	return j.stage == len(m.stages), nil
}

// Progress returns the number of steps taken and the total number of steps
// of a forward pass. The total is the same for every pass of a model.
func (j *ForwardJob) Progress() (done, total int) { return j.steps, j.m.steps }

// Outputs returns the contour (172, 264, 1), note (172, 88, 1) and onset
// (172, 88, 1) posteriorgrams. They are complete once [ForwardJob.Step] has
// reported done, and they stay valid until the next forward pass on the
// model starts. They belong to the model; the caller must not modify them.
func (j *ForwardJob) Outputs() (contour, note, onset *nn.Tensor) {
	return j.m.contour, j.m.note, j.m.onset
}
