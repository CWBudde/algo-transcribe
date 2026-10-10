package basicpitch

import (
	"math"
	"slices"
	"testing"

	"github.com/cwbudde/algo-transcribe/internal/fixture"
)

// bendClips and bendVariants are the fixture clips and note variants the
// pitch-bend, time and overlap tests run on.
var (
	bendClips    = []string{"synth", "vocal"}
	bendVariants = []string{"default", "freq_range", "low_thresholds", "no_infer_onsets", "no_melodia"}
)

func TestPitchBendGaussianWindow(t *testing.T) {
	t.Parallel()

	// scipy.signal.windows.gaussian(51, std=5): n = arange(51) - 25,
	// w = exp(-n**2 / (2*5*5)).
	inexact := 0

	for k, w := range pbGaussian {
		n := float64(k - pbTolerance)
		got := math.Exp(-(n * n) / 50)

		if got != w {
			inexact++
		}

		if d := ulpDistance(got, w); d > 1 {
			t.Errorf("k=%d: math.Exp gives %x, table %x (%d ulp)", k, got, w, d)
		}

		if w != pbGaussian[pbWindowLen-1-k] {
			t.Errorf("k=%d: window not symmetric", k)
		}
	}

	if pbGaussian[pbTolerance] != 1 {
		t.Errorf("centre = %v, want 1", pbGaussian[pbTolerance])
	}

	t.Logf("math.Exp differs from the SciPy table in %d of %d values", inexact, pbWindowLen)
}

func ulpDistance(a, b float64) uint64 {
	ua, ub := math.Float64bits(a), math.Float64bits(b)
	if ua > ub {
		return ua - ub
	}

	return ub - ua
}

func TestPitchContourBin(t *testing.T) {
	t.Parallel()

	for p := noteMIDIOffset; p <= pbMaxMIDI; p++ {
		if got := pitchContourBin(p); got != 3*(p-noteMIDIOffset) {
			t.Errorf("pitch %d: bin %d, want %d", p, got, 3*(p-noteMIDIOffset))
		}

		fs, fe, gs, ge, shift := pitchBendWindow(p)
		if fe-fs != ge-gs || fe <= fs {
			t.Errorf("pitch %d: contour [%d,%d) and gaussian [%d,%d) differ", p, fs, fe, gs, ge)
		}

		if gs < 0 || ge > pbWindowLen || fs < 0 || fe > noteContourBins {
			t.Errorf("pitch %d: slices out of range: [%d,%d) [%d,%d)", p, fs, fe, gs, ge)
		}

		// pb_shift is the position of the note's own bin in the window.
		if fs+shift != 3*(p-noteMIDIOffset) || gs+shift != pbTolerance {
			t.Errorf("pitch %d: pb_shift %d does not point at the note bin", p, shift)
		}
	}
}

func TestPitchBendWindowEdges(t *testing.T) {
	t.Parallel()

	// Values printed by upstream's arithmetic in NumPy.
	tests := []struct {
		pitch                 int
		fs, fe, gs, ge, shift int
	}{
		{21, 0, 26, 25, 51, 0},
		{22, 0, 29, 22, 51, 3},
		{29, 0, 50, 1, 51, 24},
		{30, 2, 53, 0, 51, 25},
		{100, 212, 263, 0, 51, 25},
		{101, 215, 264, 0, 49, 25},
		{108, 236, 264, 0, 28, 25},
	}

	for _, tc := range tests {
		fs, fe, gs, ge, shift := pitchBendWindow(tc.pitch)
		if fs != tc.fs || fe != tc.fe || gs != tc.gs || ge != tc.ge || shift != tc.shift {
			t.Errorf("pitch %d: got %d %d %d %d %d, want %d %d %d %d %d", tc.pitch,
				fs, fe, gs, ge, shift, tc.fs, tc.fe, tc.gs, tc.ge, tc.shift)
		}
	}
}

// fixtureFrameNotes returns the fixture's output_to_notes_polyphonic notes
// as frame notes without bends.
func fixtureFrameNotes(ns *fixture.NoteSet) []frameNote {
	notes := make([]frameNote, len(ns.Notes))
	for i, n := range ns.Notes {
		notes[i] = frameNote{Start: n.StartFrame, End: n.EndFrame, Pitch: n.Pitch, Amplitude: float32(n.Amplitude)}
	}

	return notes
}

// fixtureEvents runs getPitchBends and frameNotesToEvents on the fixture
// notes of variant.
func fixtureEvents(t *testing.T, clip *fixture.Clip, variant string) (*fixture.NoteSet, []noteEvent) {
	t.Helper()

	contour, shape, err := clip.Float32("unwrapped_contour")
	if err != nil {
		t.Fatal(err)
	}

	ns, err := clip.Notes(variant)
	if err != nil {
		t.Fatal(err)
	}

	notes := fixtureFrameNotes(ns)

	err = getPitchBends(contour, shape[0], notes)
	if err != nil {
		t.Fatalf("getPitchBends: %v", err)
	}

	events, err := frameNotesToEvents(notes, shape[0])
	if err != nil {
		t.Fatalf("frameNotesToEvents: %v", err)
	}

	return ns, events
}

func TestPitchBendsFixtures(t *testing.T) {
	t.Parallel()

	for _, name := range bendClips {
		clip := fixture.MustOpen(t, name)

		for _, variant := range bendVariants {
			t.Run(name+"/"+variant, func(t *testing.T) {
				t.Parallel()

				ns, events := fixtureEvents(t, clip, variant)
				if !ns.Options.IncludePitchBends {
					t.Fatalf("variant without pitch bends")
				}

				if len(events) != len(ns.NoteEvents) {
					t.Fatalf("%d events, want %d", len(events), len(ns.NoteEvents))
				}

				bends := 0

				for i, want := range ns.NoteEvents {
					got := events[i]
					if got.Start != want.StartS || got.End != want.EndS {
						t.Errorf("event %d: times %v-%v, want %v-%v", i, got.Start, got.End, want.StartS, want.EndS)
					}

					if got.Pitch != want.Pitch || float64(got.Amplitude) != want.Amplitude {
						t.Errorf("event %d: pitch %d amp %v, want %d %v", i, got.Pitch, got.Amplitude, want.Pitch, want.Amplitude)
					}

					if !slices.Equal(got.Bends, want.PitchBends) || want.PitchBends == nil {
						t.Errorf("event %d: bends %v, want %v", i, got.Bends, want.PitchBends)
					}

					bends += len(got.Bends)
				}

				t.Logf("%d notes, %d bends exact", len(events), bends)
			})
		}
	}
}

func TestNoteBendsEdgeCases(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())

	tests := []struct {
		name  string
		pitch int
		// rows maps a frame to (contour bin, value) pairs; other bins are 0.
		rows [][][2]float32
		want []int
	}{
		{
			name: "pitch 21 peak on bin", pitch: 21,
			rows: [][][2]float32{{{0, 0.9}, {20, 0.5}}},
			want: []int{0},
		},
		{
			// The window starts at bin 0 with gaussian[25:]: bin 3 is +3.
			name: "pitch 21 peak above", pitch: 21,
			rows: [][][2]float32{{{0, 0.1}, {3, 0.9}}},
			want: []int{3},
		},
		{
			// Bin 25 is the last bin of the window; bin 26 is outside it.
			name: "pitch 21 window edge", pitch: 21,
			rows: [][][2]float32{{{25, 0.9}, {26, 1}}},
			want: []int{25},
		},
		{
			// Bin 261; the window ends at bin 263 with gaussian[:28].
			name: "pitch 108 peak below", pitch: 108,
			rows: [][][2]float32{{{236, 1}}},
			want: []int{-25},
		},
		{
			name: "pitch 108 top bin", pitch: 108,
			rows: [][][2]float32{{{263, 0.9}, {261, 0.5}}},
			want: []int{2},
		},
		{
			name: "single frame", pitch: 60,
			rows: [][][2]float32{{{117, 0.8}}},
			want: []int{0},
		},
		{
			// All-zero rows: argmax is the first bin, so the bend is -pb_shift.
			name: "zeros pitch 60", pitch: 60,
			rows: [][][2]float32{nil, nil},
			want: []int{-25, -25},
		},
		{
			name: "zeros pitch 21", pitch: 21,
			rows: [][][2]float32{nil},
			want: []int{0},
		},
		{
			name: "zeros pitch 22", pitch: 22,
			rows: [][][2]float32{nil},
			want: []int{-3},
		},
		{
			// np.argmax returns the first NaN, whatever the maximum.
			name: "NaN", pitch: 60,
			rows: [][][2]float32{{{117, 1}, {120, nan}, {125, nan}}, {{92, nan}}},
			want: []int{3, -25},
		},
		{
			// Equal products: the first occurrence wins.
			name: "tie", pitch: 60,
			rows: [][][2]float32{{{116, 0.5}, {118, 0.5}}},
			want: []int{-1},
		},
		{
			// The Gaussian weighting can make a lower, closer peak win.
			name: "weighting", pitch: 60,
			rows: [][][2]float32{{{117, 0.5}, {127, 0.9}}},
			want: []int{0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			nFrames := len(tc.rows) + 2
			contours := make([]float32, nFrames*noteContourBins)

			for r, row := range tc.rows {
				for _, cell := range row {
					contours[(r+1)*noteContourBins+int(cell[0])] = cell[1]
				}
			}

			notes := []frameNote{{Start: 1, End: 1 + len(tc.rows), Pitch: tc.pitch, Amplitude: 0.5}}

			err := getPitchBends(contours, nFrames, notes)
			if err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(notes[0].Bends, tc.want) {
				t.Errorf("bends %v, want %v", notes[0].Bends, tc.want)
			}
		})
	}
}

func TestGetPitchBendsSharedBuffer(t *testing.T) {
	t.Parallel()

	const nFrames = 6

	contours := make([]float32, nFrames*noteContourBins)
	notes := []frameNote{
		{Start: 0, End: 2, Pitch: 60},
		{Start: 3, End: 3, Pitch: 61},
		{Start: 2, End: 5, Pitch: 62},
	}

	err := getPitchBends(contours, nFrames, notes)
	if err != nil {
		t.Fatal(err)
	}

	if notes[1].Bends == nil || len(notes[1].Bends) != 0 {
		t.Errorf("empty note: bends %#v, want empty non-nil", notes[1].Bends)
	}

	// A capped capacity: appending to one note cannot overwrite the next.
	if cap(notes[0].Bends) != len(notes[0].Bends) {
		t.Errorf("first note: cap %d, len %d", cap(notes[0].Bends), len(notes[0].Bends))
	}

	if !slices.Equal(notes[2].Bends, []int{-25, -25, -25}) {
		t.Errorf("third note: bends %v", notes[2].Bends)
	}
}

func TestGetPitchBendsErrors(t *testing.T) {
	t.Parallel()

	const nFrames = 4

	contours := make([]float32, nFrames*noteContourBins)

	tests := []struct {
		name     string
		contours []float32
		note     frameNote
	}{
		{"short contour", contours[1:], frameNote{Start: 0, End: 1, Pitch: 60}},
		{"pitch low", contours, frameNote{Start: 0, End: 1, Pitch: 20}},
		{"pitch high", contours, frameNote{Start: 0, End: 1, Pitch: 109}},
		{"negative start", contours, frameNote{Start: -1, End: 1, Pitch: 60}},
		{"end before start", contours, frameNote{Start: 2, End: 1, Pitch: 60}},
		{"end past frames", contours, frameNote{Start: 0, End: nFrames + 1, Pitch: 60}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			notes := []frameNote{{Start: 0, End: 1, Pitch: 60}, tc.note}

			err := getPitchBends(tc.contours, nFrames, notes)
			if err == nil {
				t.Fatal("no error")
			}

			if notes[0].Bends != nil {
				t.Error("notes were modified before the error")
			}
		})
	}
}

func TestModelFramesToTimeFixtures(t *testing.T) {
	t.Parallel()

	if got, want := math.Float64bits(timeWindowOffset), math.Float64bits(0x1.525d69b7026c5p-7); got != want {
		t.Errorf("window offset %x, want %x", timeWindowOffset, 0x1.525d69b7026c5p-7)
	}

	for _, name := range bendClips {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			want, shape, err := fixture.MustOpen(t, name).Float64("frame_times")
			if err != nil {
				t.Fatal(err)
			}

			got := modelFramesToTime(shape[0])
			if len(got) != len(want) {
				t.Fatalf("%d times, want %d", len(got), len(want))
			}

			for i := range want {
				if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
					t.Errorf("frame %d: %x, want %x", i, got[i], want[i])
				}
			}
		})
	}
}

func TestModelFramesToTime(t *testing.T) {
	t.Parallel()

	if got := modelFramesToTime(0); len(got) != 0 {
		t.Errorf("modelFramesToTime(0) = %v", got)
	}

	times := modelFramesToTime(timeAnnotFrames + 2)
	hop := float64(timeFFTHop) / timeSampleRate

	// Within the first 172 frames there is no window correction.
	if times[timeAnnotFrames-1] != float64((timeAnnotFrames-1)*timeFFTHop)/timeSampleRate {
		t.Errorf("frame 171: %v", times[timeAnnotFrames-1])
	}

	// Frame 172 moves back by the window offset (about 10.3 ms), so the
	// step from 171 is one hop minus the offset.
	step := times[timeAnnotFrames] - times[timeAnnotFrames-1]
	if math.Abs(step-(hop-timeWindowOffset)) > 1e-15 {
		t.Errorf("step 171->172 = %v, want %v", step, hop-timeWindowOffset)
	}

	if step <= 0 {
		t.Errorf("times not increasing at the boundary: %v", step)
	}
}

func TestFrameNotesToEventsErrors(t *testing.T) {
	t.Parallel()

	for _, n := range []frameNote{{Start: -1, End: 2}, {Start: 0, End: 10}, {Start: 10, End: 10}} {
		_, err := frameNotesToEvents([]frameNote{n}, 10)
		if err == nil {
			t.Errorf("%+v: no error", n)
		}
	}

	events, err := frameNotesToEvents([]frameNote{{Start: 0, End: 9, Pitch: 60, Amplitude: 0.25, Bends: []int{1}}}, 10)
	if err != nil {
		t.Fatal(err)
	}

	want := noteEvent{Start: 0, End: modelFrameTime(9), Pitch: 60, Amplitude: 0.25, Bends: []int{1}}
	if events[0].Start != want.Start || events[0].End != want.End || events[0].Pitch != want.Pitch ||
		events[0].Amplitude != want.Amplitude || !slices.Equal(events[0].Bends, want.Bends) {
		t.Errorf("got %+v, want %+v", events[0], want)
	}
}

func TestDropOverlappingPitchBends(t *testing.T) {
	t.Parallel()

	b := func(v ...int) []int { return v }
	nan := float32(math.NaN())

	tests := []struct {
		name string
		in   []noteEvent
		want []noteEvent
	}{
		{name: "empty"},
		{
			name: "no overlap, sorted",
			in: []noteEvent{
				{Start: 2, End: 3, Pitch: 60, Bends: b(1)},
				{Start: 0, End: 1, Pitch: 62, Bends: b(2)},
			},
			want: []noteEvent{
				{Start: 0, End: 1, Pitch: 62, Bends: b(2)},
				{Start: 2, End: 3, Pitch: 60, Bends: b(1)},
			},
		},
		{
			// start j == end i does not overlap: the loop breaks on >=.
			name: "touching",
			in: []noteEvent{
				{Start: 1, End: 2, Pitch: 60, Bends: b(1)},
				{Start: 0, End: 1, Pitch: 62, Bends: b(2)},
			},
			want: []noteEvent{
				{Start: 0, End: 1, Pitch: 62, Bends: b(2)},
				{Start: 1, End: 2, Pitch: 60, Bends: b(1)},
			},
		},
		{
			name: "chain",
			in: []noteEvent{
				{Start: 0, End: 2, Pitch: 60, Bends: b(1)},
				{Start: 1, End: 3, Pitch: 61, Bends: b(2)},
				{Start: 2.5, End: 4, Pitch: 62, Bends: b(3)},
				{Start: 4, End: 5, Pitch: 63, Bends: b(4)},
			},
			want: []noteEvent{
				{Start: 0, End: 2, Pitch: 60},
				{Start: 1, End: 3, Pitch: 61},
				{Start: 2.5, End: 4, Pitch: 62},
				{Start: 4, End: 5, Pitch: 63, Bends: b(4)},
			},
		},
		{
			// A long note overlaps a later one although the note between
			// them does not; the inner loop only breaks on the first
			// non-overlapping start.
			name: "long note",
			in: []noteEvent{
				{Start: 0, End: 10, Pitch: 60, Bends: b(1)},
				{Start: 1, End: 2, Pitch: 61, Bends: b(2)},
				{Start: 5, End: 6, Pitch: 62, Bends: b(3)},
				{Start: 10, End: 11, Pitch: 63, Bends: b(4)},
			},
			want: []noteEvent{
				{Start: 0, End: 10, Pitch: 60},
				{Start: 1, End: 2, Pitch: 61},
				{Start: 5, End: 6, Pitch: 62},
				{Start: 10, End: 11, Pitch: 63, Bends: b(4)},
			},
		},
		{
			name: "ties fall through to end, pitch, amplitude, bends",
			in: []noteEvent{
				{Start: 0, End: 2, Pitch: 60, Amplitude: 0.5, Bends: b(1, 2)},
				{Start: 0, End: 2, Pitch: 60, Amplitude: 0.5, Bends: b(1)},
				{Start: 0, End: 2, Pitch: 60, Amplitude: 0.5, Bends: b(0, 5)},
				{Start: 0, End: 2, Pitch: 60, Amplitude: 0.25, Bends: b(9)},
				{Start: 0, End: 2, Pitch: 59, Amplitude: 0.75, Bends: b(9)},
				{Start: 0, End: 1, Pitch: 70, Amplitude: 0.75, Bends: b(9)},
			},
			want: []noteEvent{
				{Start: 0, End: 1, Pitch: 70, Amplitude: 0.75},
				{Start: 0, End: 2, Pitch: 59, Amplitude: 0.75},
				{Start: 0, End: 2, Pitch: 60, Amplitude: 0.25},
				{Start: 0, End: 2, Pitch: 60, Amplitude: 0.5},
				{Start: 0, End: 2, Pitch: 60, Amplitude: 0.5},
				{Start: 0, End: 2, Pitch: 60, Amplitude: 0.5},
			},
		},
		{
			// A NaN amplitude is neither less nor greater than anything in
			// Python, so the events keep their input order.
			name: "NaN amplitude stable",
			in: []noteEvent{
				{Start: 0, End: 1, Pitch: 60, Amplitude: 0.5, Bends: b(1)},
				{Start: 0, End: 1, Pitch: 60, Amplitude: nan, Bends: b(2)},
				{Start: 2, End: 3, Pitch: 60, Amplitude: nan, Bends: b(3)},
				{Start: 2, End: 3, Pitch: 60, Amplitude: 0.5, Bends: b(4)},
			},
			want: []noteEvent{
				{Start: 0, End: 1, Pitch: 60, Amplitude: 0.5},
				{Start: 0, End: 1, Pitch: 60, Amplitude: nan},
				{Start: 2, End: 3, Pitch: 60, Amplitude: nan},
				{Start: 2, End: 3, Pitch: 60, Amplitude: 0.5},
			},
		},
		{
			// Python compares None == None as equal; nil keeps input order.
			name: "nil bends stable",
			in: []noteEvent{
				{Start: 0, End: 1, Pitch: 60, Amplitude: 0.5},
				{Start: 0, End: 1, Pitch: 60, Amplitude: 0.5, Bends: b(3)},
				{Start: 0, End: 1, Pitch: 60, Amplitude: 0.5, Bends: b(1)},
			},
			want: []noteEvent{
				{Start: 0, End: 1, Pitch: 60, Amplitude: 0.5},
				{Start: 0, End: 1, Pitch: 60, Amplitude: 0.5},
				{Start: 0, End: 1, Pitch: 60, Amplitude: 0.5},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			in := slices.Clone(tc.in)
			got := dropOverlappingPitchBends(in)

			checkEvents(t, got, tc.want)
			// The caller's slice and bends are unchanged.
			checkEvents(t, in, tc.in)
		})
	}
}

func TestDropOverlappingPitchBendsOrderOfTies(t *testing.T) {
	t.Parallel()

	// Lexicographic bends: [0 5] < [1] < [1 2]; a prefix sorts first.
	in := []noteEvent{
		{Start: 0, End: 1, Pitch: 60, Bends: []int{1, 2}},
		{Start: 0, End: 1, Pitch: 60, Bends: []int{1}},
		{Start: 0, End: 1, Pitch: 60, Bends: []int{0, 5}},
	}

	slices.SortStableFunc(in, compareNoteEvents)

	want := [][]int{{0, 5}, {1}, {1, 2}}
	for i, e := range in {
		if !slices.Equal(e.Bends, want[i]) {
			t.Errorf("position %d: %v, want %v", i, e.Bends, want[i])
		}
	}
}

func TestCompareNoteEventsNaNAmplitude(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())

	ev := func(amp float32, bends ...int) noteEvent {
		return noteEvent{Start: 0, End: 1, Pitch: 60, Amplitude: amp, Bends: bends}
	}

	// Expectations are Python's sorted() on the same tuples. The bends tell
	// the events apart; a NaN amplitude stops the comparison before them.
	tests := []struct {
		name string
		in   []noteEvent
		want [][]int
	}{
		{name: "finite before NaN", in: []noteEvent{ev(0.5, 2), ev(nan, 1)}, want: [][]int{{2}, {1}}},
		{name: "NaN before finite", in: []noteEvent{ev(nan, 2), ev(0.5, 1)}, want: [][]int{{2}, {1}}},
		{name: "two NaNs skip bends", in: []noteEvent{ev(nan, 2), ev(nan, 1)}, want: [][]int{{2}, {1}}},
		{name: "equal finite compare bends", in: []noteEvent{ev(0.5, 2), ev(0.5, 1)}, want: [][]int{{1}, {2}}},
		{name: "finite order", in: []noteEvent{ev(0.5, 1), ev(0.25, 2)}, want: [][]int{{2}, {1}}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := slices.Clone(tc.in)
			slices.SortStableFunc(got, compareNoteEvents)

			for i, e := range got {
				if !slices.Equal(e.Bends, tc.want[i]) {
					t.Errorf("position %d: bends %v, want %v", i, e.Bends, tc.want[i])
				}
			}
		})
	}
}

func checkEvents(t *testing.T, got, want []noteEvent) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%d events, want %d", len(got), len(want))
	}

	for i := range want {
		g, w := got[i], want[i]
		if g.Start != w.Start || g.End != w.End || g.Pitch != w.Pitch ||
			math.Float32bits(g.Amplitude) != math.Float32bits(w.Amplitude) ||
			(g.Bends == nil) != (w.Bends == nil) || !slices.Equal(g.Bends, w.Bends) {
			t.Errorf("event %d: %+v, want %+v", i, g, w)
		}
	}
}

// numpyLinspace is np.linspace(start, stop, num) of NumPy 2.x for num >= 1:
// step = (stop-start)/(num-1), y = arange(num)*step + start with both
// operations rounded, and the last element set to stop.
func numpyLinspace(start, stop float64, num int) []float64 {
	y := make([]float64, num)
	if num == 1 {
		y[0] = start

		return y
	}

	step := (stop - start) / float64(num-1)
	for i := range y {
		y[i] = float64(float64(i)*step) + start
	}

	y[num-1] = stop

	return y
}

// midiBendTicks is np.round(np.array(bends) * 4096 / 3) clipped to the
// 14-bit pitch-wheel range, as note_events_to_midi computes it.
func midiBendTicks(b int) int {
	v := int(math.RoundToEven(float64(b*4096) / 3))

	return min(max(v, -8192), 8191)
}

// midiVelocity is int(np.round(127 * amplitude)) with amplitude an
// np.float32: under NEP 50 the product is a float32 multiplication.
func midiVelocity(amplitude float32) int {
	return int(math.RoundToEven(float64(127 * amplitude)))
}

func TestDropOverlappingPitchBendsMIDIFixtures(t *testing.T) {
	t.Parallel()

	for _, name := range bendClips {
		clip := fixture.MustOpen(t, name)

		for _, variant := range bendVariants {
			t.Run(name+"/"+variant, func(t *testing.T) {
				t.Parallel()

				ns, events := fixtureEvents(t, clip, variant)
				if ns.Options.MultiplePitchBends || len(ns.MIDI.Instruments) != 1 {
					t.Fatalf("want one instrument without multiple_pitch_bends")
				}

				inst := ns.MIDI.Instruments[0]
				sorted := dropOverlappingPitchBends(events)

				if len(sorted) != len(inst.Notes) {
					t.Fatalf("%d notes, want %d", len(sorted), len(inst.Notes))
				}

				var bends []fixture.PitchBend

				kept, f64Diff := 0, 0

				for i, e := range sorted {
					want := inst.Notes[i]
					if e.Start != want.Start || e.End != want.End || e.Pitch != want.Pitch {
						t.Errorf("note %d: %v-%v pitch %d, want %v-%v pitch %d",
							i, e.Start, e.End, e.Pitch, want.Start, want.End, want.Pitch)
					}

					if v := midiVelocity(e.Amplitude); v != want.Velocity {
						t.Errorf("note %d: velocity %d, want %d", i, v, want.Velocity)
					}

					if int(math.RoundToEven(127*float64(e.Amplitude))) != want.Velocity {
						f64Diff++
					}

					if len(e.Bends) == 0 {
						continue
					}

					kept++

					times := numpyLinspace(e.Start, e.End, len(e.Bends))
					for k, b := range e.Bends {
						bends = append(bends, fixture.PitchBend{Time: times[k], Pitch: midiBendTicks(b)})
					}
				}

				if !slices.Equal(bends, inst.PitchBends) {
					t.Errorf("%d pitch bends, want %d (or values differ)", len(bends), len(inst.PitchBends))
				}

				t.Logf("%d notes, %d keep bends, %d bend events; float64 velocity differs on %d notes",
					len(sorted), kept, len(bends), f64Diff)
			})
		}
	}
}

func TestMIDIVelocityFloat32(t *testing.T) {
	t.Parallel()

	// 127 * np.float32(0x1.83060cp-7) rounds to exactly 1.5 in float32, so
	// NumPy gives velocity 2; the float64 product is just below 1.5 and
	// would give 1. 60 float32 amplitudes in [0, 1] behave like this; none
	// occurs in the fixtures.
	a := float32(0x1.83060cp-7)

	if got := midiVelocity(a); got != 2 {
		t.Errorf("float32 velocity %d, want 2", got)
	}

	if got := int(math.RoundToEven(127 * float64(a))); got != 1 {
		t.Errorf("float64 velocity %d, want 1", got)
	}
}
