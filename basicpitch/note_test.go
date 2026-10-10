package basicpitch

import (
	"slices"
	"testing"

	"github.com/cwbudde/algo-transcribe/internal/fixture"
)

// TestModelOutputToNotesFixtures runs the whole note creation on the fixture
// model outputs and compares the result with upstream's note events, and the
// overlap-dropped events with the notes and bends of upstream's MIDI object.
func TestModelOutputToNotesFixtures(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"synth", "vocal"} {
		clip := fixture.MustOpen(t, name)

		for _, variant := range noteVariants {
			t.Run(name+"/"+variant, func(t *testing.T) {
				t.Parallel()

				frames, onsets, nFrames := fixturePosteriorgrams(t, clip)

				contours, _, err := clip.Float32("unwrapped_contour")
				if err != nil {
					t.Fatal(err)
				}

				ns, err := clip.Notes(variant)
				if err != nil {
					t.Fatal(err)
				}

				var d noteDecoder

				events, err := d.modelOutputToNotes(contours, frames, onsets, nFrames,
					fixtureNoteParams(ns.Options), ns.Options.IncludePitchBends)
				if err != nil {
					t.Fatal(err)
				}

				want := make([]noteEvent, len(ns.NoteEvents))
				for i, e := range ns.NoteEvents {
					want[i] = noteEvent{
						Start: e.StartS, End: e.EndS, Pitch: e.Pitch,
						Amplitude: float32(e.Amplitude), Bends: e.PitchBends,
					}

					if float64(want[i].Amplitude) != e.Amplitude {
						t.Fatalf("fixture amplitude %v is not a float32", e.Amplitude)
					}
				}

				checkEvents(t, events, want)

				// Without multiple_pitch_bends upstream writes one instrument
				// whose notes are the sorted, overlap-dropped events.
				dropped := dropOverlappingPitchBends(events)
				if len(ns.MIDI.Instruments) != 1 || len(ns.MIDI.Instruments[0].Notes) != len(dropped) {
					t.Fatalf("MIDI has %d instruments, want one with %d notes", len(ns.MIDI.Instruments), len(dropped))
				}

				bends := 0

				for i, n := range ns.MIDI.Instruments[0].Notes {
					e := dropped[i]
					if e.Start != n.Start || e.End != n.End || e.Pitch != n.Pitch {
						t.Errorf("sorted note %d: %v-%v pitch %d, MIDI has %v-%v pitch %d",
							i, e.Start, e.End, e.Pitch, n.Start, n.End, n.Pitch)
					}

					bends += len(e.Bends)
				}

				if got := len(ns.MIDI.Instruments[0].PitchBends); got != bends {
					t.Errorf("%d bends kept, MIDI has %d", bends, got)
				}

				t.Logf("%d events exact, %d keep bends after the overlap drop", len(events),
					len(slices.DeleteFunc(slices.Clone(dropped), func(e noteEvent) bool { return e.Bends == nil })))
			})
		}
	}
}

func TestModelOutputToNotesWithoutBends(t *testing.T) {
	t.Parallel()

	clip := fixture.MustOpen(t, "vocal")
	frames, onsets, nFrames := fixturePosteriorgrams(t, clip)

	var d noteDecoder

	events, err := d.modelOutputToNotes(nil, frames, onsets, nFrames,
		noteParams{onsetThresh: 0.5, frameThresh: 0.3, minNoteLen: 11, inferOnsets: true, melodiaTrick: true, energyTol: noteEnergyTol},
		false)
	if err != nil {
		t.Fatal(err)
	}

	if len(events) == 0 {
		t.Fatal("no events")
	}

	for i, e := range events {
		if e.Bends != nil {
			t.Errorf("event %d: bends %v, want nil (None)", i, e.Bends)
		}
	}
}

func TestModelOutputToNotesErrors(t *testing.T) {
	t.Parallel()

	var d noteDecoder

	p := noteParams{onsetThresh: 0.5, frameThresh: 0.3, energyTol: noteEnergyTol}
	frames := make([]float32, 4*noteBins)
	onsets := make([]float32, 4*noteBins)

	_, err := d.modelOutputToNotes(nil, frames[:1], onsets, 4, p, true)
	if err == nil {
		t.Error("short frames: no error")
	}

	// A note needs a contour of the right size for its bends.
	frames2, onsets2 := runMatrix(40, []cellRun{{t0: 1, t1: 39, f: 40, v: 0.9}}), make([]float32, 40*noteBins)
	onsets2[5*noteBins+40] = 0.9

	_, err = d.modelOutputToNotes(make([]float32, 3), frames2, onsets2, 40, p, true)
	if err == nil {
		t.Error("short contour: no error")
	}
}
