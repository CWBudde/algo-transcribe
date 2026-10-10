package basicpitch

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/algo-transcribe/internal/fixture"
	"github.com/cwbudde/algo-transcribe/nn"
)

// modelParityTol is the model-output tolerance of docs/port-spec.md §6:
// ≤ 1e-4 absolute, given the fixture harmonic stack. Never loosen it.
const modelParityTol = 1e-4

// modelPinnedSteps is the number of ForwardJob steps of one forward pass
// with the current step budget. It changes whenever modelStepMACs (or the
// per-row cost of a layer) changes.
const modelPinnedSteps = 395

func newTestModel(t *testing.T) *Model {
	t.Helper()

	m, err := NewModel()
	if err != nil {
		t.Fatalf("NewModel: %v", err)
	}

	return m
}

// testStack returns window w of the harmonic stack of clip in NHWC.
func testStack(t *testing.T, clip *fixture.Clip, w int) *nn.Tensor {
	t.Helper()

	data, shape, err := clip.HarmonicStackNHWC(w)
	if err != nil {
		t.Fatalf("harmonic stack window %d: %v", w, err)
	}

	if !slices.Equal(shape, []int{modelFrames, modelContourBins, modelHarmonics}) {
		t.Fatalf("harmonic stack window %d has shape %v", w, shape)
	}

	stack, err := nn.TensorFrom(data, modelFrames, modelContourBins, modelHarmonics)
	if err != nil {
		t.Fatal(err)
	}

	return stack
}

// testWindowArray loads a (window, time, freq) fixture array and checks its
// shape.
func testWindowArray(t *testing.T, clip *fixture.Clip, name string, bins int) []float32 {
	t.Helper()

	data, shape, err := clip.Float32(name)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}

	want := []int{clip.Manifest.NWindows, modelFrames, bins}
	if !slices.Equal(shape, want) {
		t.Fatalf("%s has shape %v, want %v", name, shape, want)
	}

	return data
}

// maxAbsDiff returns the largest absolute difference of got and want and
// fails the test if any value of got is outside [0, 1] or not finite.
func maxAbsDiff(t *testing.T, label string, got, want []float32) float64 {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", label, len(got), len(want))
	}

	var worst float64

	for i, g := range got {
		if !(g >= 0 && g <= 1) {
			t.Fatalf("%s: value %d = %v is not a sigmoid output in [0, 1]", label, i, g)
		}

		worst = max(worst, math.Abs(float64(g)-float64(want[i])))
	}

	return worst
}

func TestModelParity(t *testing.T) {
	t.Parallel()

	names, err := fixture.Names()
	if err != nil {
		t.Fatal(err)
	}

	if len(names) == 0 {
		t.Fatal("no fixture clips")
	}

	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			clip := fixture.MustOpen(t, name)

			// Every window of every clip, including synth window 3, whose
			// harmonic stack is all zeros: it checks the bias-only path of
			// the whole network.
			outputs := []struct {
				name  string
				bins  int
				want  []float32
				worst float64
			}{
				{name: "contour", bins: modelContourBins},
				{name: "note", bins: modelNoteBins},
				{name: "onset", bins: modelNoteBins},
			}

			// Each array is read once per clip; Clip.Window would reread it
			// for every window.
			for i := range outputs {
				outputs[i].want = testWindowArray(t, clip, outputs[i].name, outputs[i].bins)
			}

			m := newTestModel(t)

			for w := range clip.Manifest.NWindows {
				contour, note, onset, err := m.Forward(testStack(t, clip, w))
				if err != nil {
					t.Fatalf("window %d: Forward: %v", w, err)
				}

				for i, got := range []*nn.Tensor{contour, note, onset} {
					o := &outputs[i]
					if got.Shape != (nn.Shape{H: modelFrames, W: o.bins, C: 1}) {
						t.Fatalf("window %d: %s has shape %v", w, o.name, got.Shape)
					}

					n := modelFrames * o.bins
					d := maxAbsDiff(t, o.name, got.Data, o.want[w*n:(w+1)*n])

					if d > modelParityTol {
						t.Errorf("window %d: %s: max abs diff %.3g > %g", w, o.name, d, modelParityTol)
					}

					o.worst = max(o.worst, d)
				}
			}

			for _, o := range outputs {
				t.Logf("%s %s: max abs diff %.3g over %d windows", name, o.name, o.worst, clip.Manifest.NWindows)
			}
		})
	}
}

func TestModelSteps(t *testing.T) {
	t.Parallel()

	m := newTestModel(t)

	// Every stage advances by its row block, at least one row, and the
	// total is the sum of the per-stage step counts.
	var total int

	for _, s := range m.stages {
		rows := max(1, modelStepMACs/s.conv.MACsPerRow())
		if s.rows != rows {
			t.Errorf("%s: %d rows per step, want %d", s.name, s.rows, rows)
		}

		total += (modelFrames + rows - 1) / rows
	}

	if total != modelPinnedSteps {
		t.Errorf("sum of per-stage steps = %d, want %d", total, modelPinnedSteps)
	}

	if m.steps != total {
		t.Errorf("model steps = %d, want %d", m.steps, total)
	}
}

func TestForwardJobMatchesForward(t *testing.T) {
	t.Parallel()

	stack := testStack(t, fixture.MustOpen(t, "vocal"), 0)

	ref := newTestModel(t)

	c, n, o, err := ref.Forward(stack)
	if err != nil {
		t.Fatal(err)
	}

	m := newTestModel(t)

	job, err := m.NewForwardJob(stack)
	if err != nil {
		t.Fatal(err)
	}

	if done, total := job.Progress(); done != 0 || total != modelPinnedSteps {
		t.Fatalf("fresh job progress = %d/%d, want 0/%d", done, total, modelPinnedSteps)
	}

	for step := 1; ; step++ {
		done, err := job.Step()
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}

		got, total := job.Progress()
		if got != step || total != modelPinnedSteps {
			t.Fatalf("after step %d: progress %d/%d", step, got, total)
		}

		if done != (step == total) {
			t.Fatalf("step %d of %d reported done = %v", step, total, done)
		}

		if done {
			break
		}
	}

	// Further steps are no-ops.
	for range 3 {
		done, err := job.Step()
		if err != nil || !done {
			t.Fatalf("Step after done = %v, %v; want true, nil", done, err)
		}
	}

	if got, _ := job.Progress(); got != modelPinnedSteps {
		t.Errorf("progress after extra steps = %d, want %d", got, modelPinnedSteps)
	}

	jc, jn, jo := job.Outputs()
	assertBitEqual(t, "contour", jc, c)
	assertBitEqual(t, "note", jn, n)
	assertBitEqual(t, "onset", jo, o)
}

func assertBitEqual(t *testing.T, label string, got, want *nn.Tensor) {
	t.Helper()

	if got.Shape != want.Shape || len(got.Data) != len(want.Data) {
		t.Fatalf("%s: shape %v (%d values), want %v (%d values)",
			label, got.Shape, len(got.Data), want.Shape, len(want.Data))
	}

	for i := range got.Data {
		if math.Float32bits(got.Data[i]) != math.Float32bits(want.Data[i]) {
			t.Fatalf("%s: value %d = %v (%#08x), want %v (%#08x)", label, i,
				got.Data[i], math.Float32bits(got.Data[i]), want.Data[i], math.Float32bits(want.Data[i]))
		}
	}
}

// snapshot copies the three outputs so a later pass cannot change them.
func snapshot(contour, note, onset *nn.Tensor) [3]*nn.Tensor {
	var s [3]*nn.Tensor

	for i, t := range []*nn.Tensor{contour, note, onset} {
		s[i] = &nn.Tensor{Shape: t.Shape, Data: slices.Clone(t.Data)}
	}

	return s
}

func TestForwardJobSuperseded(t *testing.T) {
	t.Parallel()

	clip := fixture.MustOpen(t, "synth")
	stackA, stackB := testStack(t, clip, 0), testStack(t, clip, 1)

	ref := newTestModel(t)

	c, n, o, err := ref.Forward(stackA)
	if err != nil {
		t.Fatal(err)
	}

	wantA := snapshot(c, n, o)

	tests := []struct {
		name string
		// supersede starts another pass on m.
		supersede func(t *testing.T, m *Model)
	}{
		{"Forward", func(t *testing.T, m *Model) {
			t.Helper()

			if _, _, _, err := m.Forward(stackB); err != nil {
				t.Fatal(err)
			}
		}},
		{"NewForwardJob", func(t *testing.T, m *Model) {
			t.Helper()

			b, err := m.NewForwardJob(stackB)
			if err != nil {
				t.Fatal(err)
			}

			if _, err := b.Step(); err != nil {
				t.Fatal(err)
			}
		}},
		{"Reset", func(t *testing.T, m *Model) {
			t.Helper()

			b, err := m.NewForwardJob(stackA)
			if err != nil {
				t.Fatal(err)
			}

			// NewForwardJob superseded job A already; resetting B onto
			// another stack supersedes it again.
			if err := b.Reset(stackB); err != nil {
				t.Fatal(err)
			}
		}},
	}

	// steps is how far job A gets before it is superseded; a completed job
	// is superseded too, because the new pass overwrites its outputs.
	for _, steps := range []int{5, modelPinnedSteps} {
		for _, tc := range tests {
			t.Run(fmt.Sprintf("%s/after%d", tc.name, steps), func(t *testing.T) {
				t.Parallel()

				m := newTestModel(t)

				a, err := m.NewForwardJob(stackA)
				if err != nil {
					t.Fatal(err)
				}

				for i := range steps {
					done, err := a.Step()
					if err != nil {
						t.Fatal(err)
					}

					if want := i == modelPinnedSteps-1; done != want {
						t.Fatalf("step %d: done = %v, want %v", i, done, want)
					}
				}

				tc.supersede(t, m)

				done, err := a.Step()
				if !errors.Is(err, ErrJobSuperseded) || done {
					t.Fatalf("Step of superseded job = %v, %v; want false, ErrJobSuperseded", done, err)
				}

				if got, _ := a.Progress(); got != steps {
					t.Errorf("superseded job progress = %d, want %d", got, steps)
				}

				// Reset revives the job, and the pass it then runs is complete.
				if err := a.Reset(stackA); err != nil {
					t.Fatal(err)
				}

				if got, _ := a.Progress(); got != 0 {
					t.Errorf("progress after Reset = %d, want 0", got)
				}

				runJob(t, a)

				ac, an, ao := a.Outputs()
				assertBitEqual(t, "contour", ac, wantA[0])
				assertBitEqual(t, "note", an, wantA[1])
				assertBitEqual(t, "onset", ao, wantA[2])
			})
		}
	}
}

// runJob steps j to completion and checks that it takes the pinned number of
// steps.
func runJob(t *testing.T, j *ForwardJob) {
	t.Helper()

	for {
		done, err := j.Step()
		if err != nil {
			t.Fatal(err)
		}

		if done {
			break
		}
	}

	if got, total := j.Progress(); got != total || total != modelPinnedSteps {
		t.Fatalf("finished job progress = %d/%d, want %d/%d", got, total, modelPinnedSteps, modelPinnedSteps)
	}
}

func TestModelInvalidStack(t *testing.T) {
	t.Parallel()

	good := nn.Shape{H: modelFrames, W: modelContourBins, C: modelHarmonics}

	tests := []struct {
		name  string
		stack *nn.Tensor
		want  string
	}{
		{"nil", nil, "nil harmonic stack"},
		{"one channel", nn.NewTensor(modelFrames, modelContourBins, 1), "harmonic stack has shape"},
		{"171 frames", nn.NewTensor(modelFrames-1, modelContourBins, modelHarmonics), "harmonic stack has shape"},
		{"note bins", nn.NewTensor(modelFrames, modelNoteBins, modelHarmonics), "harmonic stack has shape"},
		{"short data", &nn.Tensor{Shape: good, Data: make([]float32, good.Size()-1)}, "harmonic stack has shape"},
		{"long data", &nn.Tensor{Shape: good, Data: make([]float32, good.Size()+1)}, "harmonic stack has shape"},
		{"no data", &nn.Tensor{Shape: good}, "harmonic stack has shape"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := newTestModel(t)

			check := func(api string, err error) {
				t.Helper()

				if err == nil {
					t.Fatalf("%s: no error", api)
				}

				if !strings.HasPrefix(err.Error(), "basicpitch: model: ") || !strings.Contains(err.Error(), tc.want) {
					t.Errorf("%s: error %q, want it to contain %q", api, err, tc.want)
				}
			}

			c, n, o, err := m.Forward(tc.stack)
			check("Forward", err)

			if c != nil || n != nil || o != nil {
				t.Error("Forward returned outputs with an error")
			}

			j, err := m.NewForwardJob(tc.stack)
			check("NewForwardJob", err)

			if j != nil {
				t.Error("NewForwardJob returned a job with an error")
			}

			valid, err := m.NewForwardJob(nn.NewTensor(good.H, good.W, good.C))
			if err != nil {
				t.Fatal(err)
			}

			check("Reset", valid.Reset(tc.stack))

			// A rejected Reset leaves the job untouched and current.
			if _, err := valid.Step(); err != nil {
				t.Errorf("Step after rejected Reset: %v", err)
			}
		})
	}
}

// TestModelZeroAllocs must not run in parallel: AllocsPerRun counts the
// allocations of the whole process.
func TestModelZeroAllocs(t *testing.T) {
	stack := testStack(t, fixture.MustOpen(t, "vocal"), 1)
	m := newTestModel(t)

	var err error

	allocs := testing.AllocsPerRun(1, func() {
		_, _, _, err = m.Forward(stack)
	})

	if err != nil {
		t.Fatal(err)
	}

	if allocs != 0 {
		t.Errorf("Forward: %v allocations per run, want 0", allocs)
	}

	job, err := m.NewForwardJob(stack)
	if err != nil {
		t.Fatal(err)
	}

	allocs = testing.AllocsPerRun(1, func() {
		err = job.Reset(stack)

		for err == nil {
			var done bool

			done, err = job.Step()
			if done {
				break
			}
		}
	})

	if err != nil {
		t.Fatal(err)
	}

	if allocs != 0 {
		t.Errorf("Reset and Step loop: %v allocations per run, want 0", allocs)
	}

	if done, total := job.Progress(); done != total {
		t.Errorf("job progress %d/%d after the measured loop", done, total)
	}
}

func TestModelDeterministic(t *testing.T) {
	t.Parallel()

	clip := fixture.MustOpen(t, "synth")
	stack, other := testStack(t, clip, 1), testStack(t, clip, 0)

	m := newTestModel(t)

	c, n, o, err := m.Forward(stack)
	if err != nil {
		t.Fatal(err)
	}

	first := snapshot(c, n, o)

	// A pass on another window in between must not leak into the next one.
	if _, _, _, err := m.Forward(other); err != nil {
		t.Fatal(err)
	}

	c, n, o, err = m.Forward(stack)
	if err != nil {
		t.Fatal(err)
	}

	assertBitEqual(t, "contour (same model)", c, first[0])
	assertBitEqual(t, "note (same model)", n, first[1])
	assertBitEqual(t, "onset (same model)", o, first[2])

	c, n, o, err = newTestModel(t).Forward(stack)
	if err != nil {
		t.Fatal(err)
	}

	assertBitEqual(t, "contour (fresh model)", c, first[0])
	assertBitEqual(t, "note (fresh model)", n, first[1])
	assertBitEqual(t, "onset (fresh model)", o, first[2])
}
