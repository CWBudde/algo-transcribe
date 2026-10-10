package basicpitch_test

import (
	"fmt"

	"github.com/cwbudde/algo-transcribe/basicpitch"
	"github.com/cwbudde/algo-transcribe/internal/fixture"
	"github.com/cwbudde/algo-transcribe/nn"
)

// exampleStack returns window w of the harmonic stack of a fixture clip as a
// (172, 264, 8) tensor.
func exampleStack(clip string, w int) (*nn.Tensor, error) {
	c, err := fixture.Open(clip)
	if err != nil {
		return nil, err
	}

	data, shape, err := c.HarmonicStackNHWC(w)
	if err != nil {
		return nil, err
	}

	return nn.TensorFrom(data, shape[0], shape[1], shape[2])
}

// countAbove returns the number of values of t above threshold.
func countAbove(t *nn.Tensor, threshold float32) int {
	n := 0

	for _, v := range t.Data {
		if v > threshold {
			n++
		}
	}

	return n
}

func ExampleNewModel() {
	m, err := basicpitch.NewModel()
	if err != nil {
		fmt.Println(err)

		return
	}

	// Silence: the harmonic stack of an all-zero window.
	contour, note, onset, err := m.Forward(nn.NewTensor(172, 264, 8))
	if err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println("contour", contour.Shape, "note", note.Shape, "onset", onset.Shape)
	fmt.Println("active notes:", countAbove(note, 0.5))
	// Output:
	// contour (172,264,1) note (172,88,1) onset (172,88,1)
	// active notes: 0
}

func ExampleModel_Forward() {
	m, err := basicpitch.NewModel()
	if err != nil {
		fmt.Println(err)

		return
	}

	stack, err := exampleStack("vocal", 0)
	if err != nil {
		fmt.Println(err)

		return
	}

	_, note, onset, err := m.Forward(stack)
	if err != nil {
		fmt.Println(err)

		return
	}

	// The strongest note activation of the window, as frame and MIDI pitch
	// (note bin 0 is MIDI 21, A0).
	best := 0

	for i, v := range note.Data {
		if v > note.Data[best] {
			best = i
		}
	}

	fmt.Println("note activations > 0.5:", countAbove(note, 0.5))
	fmt.Println("onset activations > 0.5:", countAbove(onset, 0.5))
	fmt.Printf("strongest: frame %d, MIDI %d\n", best/note.W, 21+best%note.W)
	// Output:
	// note activations > 0.5: 69
	// onset activations > 0.5: 41
	// strongest: frame 122, MIDI 60
}

func ExampleModel_NewForwardJob() {
	m, err := basicpitch.NewModel()
	if err != nil {
		fmt.Println(err)

		return
	}

	stack, err := exampleStack("vocal", 0)
	if err != nil {
		fmt.Println(err)

		return
	}

	job, err := m.NewForwardJob(stack)
	if err != nil {
		fmt.Println(err)

		return
	}

	// Each Step is a bounded unit of work; a caller with a time budget can
	// stop between any two steps and continue later.
	for {
		done, err := job.Step()
		if err != nil {
			fmt.Println(err)

			return
		}

		if done {
			break
		}
	}

	steps, total := job.Progress()
	_, note, _ := job.Outputs()

	fmt.Printf("%d of %d steps\n", steps, total)
	fmt.Println("note activations > 0.5:", countAbove(note, 0.5))
	// Output:
	// 395 of 395 steps
	// note activations > 0.5: 69
}

func ExampleForwardJob_Step() {
	m, err := basicpitch.NewModel()
	if err != nil {
		fmt.Println(err)

		return
	}

	job, err := m.NewForwardJob(nn.NewTensor(172, 264, 8))
	if err != nil {
		fmt.Println(err)

		return
	}

	// Run a few steps per "tick", as a budget loop would.
	const stepsPerTick = 100

	ticks := 0

	for done := false; !done; {
		ticks++

		for range stepsPerTick {
			done, err = job.Step()
			if err != nil {
				fmt.Println(err)

				return
			}

			if done {
				break
			}
		}

		steps, total := job.Progress()
		fmt.Printf("tick %d: %d/%d\n", ticks, steps, total)
	}
	// Output:
	// tick 1: 100/395
	// tick 2: 200/395
	// tick 3: 300/395
	// tick 4: 395/395
}

func ExampleForwardJob_Reset() {
	m, err := basicpitch.NewModel()
	if err != nil {
		fmt.Println(err)

		return
	}

	clip, err := fixture.Open("vocal")
	if err != nil {
		fmt.Println(err)

		return
	}

	var job *basicpitch.ForwardJob

	// One job serves every window: Reset starts the next pass without
	// allocating.
	for w := range clip.Manifest.NWindows {
		stack, err := exampleStack("vocal", w)
		if err != nil {
			fmt.Println(err)

			return
		}

		if job == nil {
			job, err = m.NewForwardJob(stack)
		} else {
			err = job.Reset(stack)
		}

		if err != nil {
			fmt.Println(err)

			return
		}

		for done := false; !done; {
			done, err = job.Step()
			if err != nil {
				fmt.Println(err)

				return
			}
		}

		_, note, _ := job.Outputs()
		fmt.Printf("window %d: %d note activations > 0.5\n", w, countAbove(note, 0.5))
	}
	// Output:
	// window 0: 69 note activations > 0.5
	// window 1: 66 note activations > 0.5
	// window 2: 20 note activations > 0.5
}
