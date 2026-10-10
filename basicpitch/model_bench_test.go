package basicpitch

import (
	"testing"
	"time"

	"github.com/cwbudde/algo-transcribe/internal/fixture"
	"github.com/cwbudde/algo-transcribe/nn"
)

// windowAudioSeconds is the new audio one window contributes: the hop of
// 36164 samples at 22050 Hz (docs/port-spec.md §2).
const windowAudioSeconds = 36164.0 / 22050.0

// benchModel builds a model and loads the harmonic stack of window 0 of the
// vocal fixture, so the benchmarks run on real data. It skips the benchmark
// if the fixtures cannot be loaded.
func benchModel(b *testing.B) (*Model, *nn.Tensor) {
	b.Helper()

	clip, err := fixture.Open("vocal")
	if err != nil {
		b.Skipf("fixture vocal: %v", err)
	}

	data, shape, err := clip.HarmonicStackNHWC(0)
	if err != nil {
		b.Skipf("fixture vocal: harmonic stack: %v", err)
	}

	stack, err := nn.TensorFrom(data, shape[0], shape[1], shape[2])
	if err != nil {
		b.Fatal(err)
	}

	m, err := NewModel()
	if err != nil {
		b.Fatal(err)
	}

	return m, stack
}

// BenchmarkModelForward runs the CNN over one full window and reports the
// realtime factor: seconds of new audio per window over seconds of compute.
func BenchmarkModelForward(b *testing.B) {
	m, stack := benchModel(b)

	b.ReportAllocs()

	for b.Loop() {
		if _, _, _, err := m.Forward(stack); err != nil {
			b.Fatal(err)
		}
	}

	perWindow := b.Elapsed().Seconds() / float64(b.N)
	b.ReportMetric(perWindow*1e3, "ms/window")
	b.ReportMetric(windowAudioSeconds/perWindow, "x-realtime")
}

// BenchmarkForwardJobStep measures one step of each stage of the CNN. The
// job is advanced to the stage, so every input of the stage is a real
// activation, and then the same kind of step is repeated by rewinding the
// job. Only full row blocks are timed; a trailing partial block is cheaper.
func BenchmarkForwardJobStep(b *testing.B) {
	m, stack := benchModel(b)

	for i := range m.stages {
		s := &m.stages[i]

		b.Run(s.name, func(b *testing.B) {
			j, err := m.NewForwardJob(stack)
			if err != nil {
				b.Fatal(err)
			}

			for j.stage < i {
				if _, err := j.Step(); err != nil {
					b.Fatal(err)
				}
			}

			blocks := modelFrames / s.rows

			b.ReportAllocs()

			k := 0
			for b.Loop() {
				j.stage, j.row = i, k*s.rows
				if _, err := j.Step(); err != nil {
					b.Fatal(err)
				}

				k = (k + 1) % blocks
			}

			b.ReportMetric(float64(s.rows), "rows/step")
		})
	}
}

// BenchmarkForwardJobMaxStep runs full forward passes step by step, timing
// every step, and reports the longest single step seen over all passes
// (wall time, so it includes scheduler and GC jitter), the stage it belonged
// to and the mean step.
func BenchmarkForwardJobMaxStep(b *testing.B) {
	m, stack := benchModel(b)

	j, err := m.NewForwardJob(stack)
	if err != nil {
		b.Fatal(err)
	}

	var (
		maxStep  time.Duration
		maxStage int
		total    time.Duration
		steps    int
	)

	b.ReportAllocs()

	for b.Loop() {
		if err := j.Reset(stack); err != nil {
			b.Fatal(err)
		}

		for {
			stage := j.stage
			t0 := time.Now()
			done, err := j.Step()
			d := time.Since(t0)

			if err != nil {
				b.Fatal(err)
			}

			total += d
			steps++

			if d > maxStep {
				maxStep, maxStage = d, stage
			}

			if done {
				break
			}
		}
	}

	b.ReportMetric(float64(maxStep.Nanoseconds()), "max-step-ns")
	b.ReportMetric(float64(maxStage), "max-step-stage")
	b.ReportMetric(float64(total.Nanoseconds())/float64(steps), "mean-step-ns")
}
