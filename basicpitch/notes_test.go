package basicpitch

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/cwbudde/algo-transcribe/internal/fixture"
)

// noteVariants are the note-creation variants of every fixture clip.
var noteVariants = []string{"default", "freq_range", "low_thresholds", "no_infer_onsets", "no_melodia"}

// fixturePosteriorgrams loads the unwrapped note and onset outputs of clip.
func fixturePosteriorgrams(tb testing.TB, clip *fixture.Clip) (frames, onsets []float32, nFrames int) {
	tb.Helper()

	frames, shape, err := clip.Float32("unwrapped_note")
	if err != nil {
		tb.Fatal(err)
	}

	onsets, oshape, err := clip.Float32("unwrapped_onset")
	if err != nil {
		tb.Fatal(err)
	}

	if len(shape) != 2 || shape[1] != noteBins || len(oshape) != 2 || oshape[0] != shape[0] || oshape[1] != noteBins {
		tb.Fatalf("shapes %v and %v, want (n, %d)", shape, oshape, noteBins)
	}

	return frames, onsets, shape[0]
}

// fixtureNoteParams converts the options of a fixture variant.
func fixtureNoteParams(o fixture.NoteOptions) noteParams {
	p := noteParams{
		onsetThresh:  o.OnsetThreshold,
		frameThresh:  o.FrameThreshold,
		minNoteLen:   o.MinNoteLenFrames,
		inferOnsets:  o.InferOnsets,
		melodiaTrick: o.MelodiaTrick,
		energyTol:    o.EnergyTol,
	}

	if o.MinimumFrequency != nil {
		p.minFreq = *o.MinimumFrequency
	}

	if o.MaximumFrequency != nil {
		p.maxFreq = *o.MaximumFrequency
	}

	return p
}

func TestDecodeNotesFixtures(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"synth", "vocal"} {
		for _, variant := range noteVariants {
			t.Run(name+"/"+variant, func(t *testing.T) {
				t.Parallel()

				clip := fixture.MustOpen(t, name)

				ns, err := clip.Notes(variant)
				if err != nil {
					t.Fatal(err)
				}

				frames, onsets, nFrames := fixturePosteriorgrams(t, clip)

				var d noteDecoder

				got, err := d.decode(frames, onsets, nFrames, fixtureNoteParams(ns.Options), nil)
				if err != nil {
					t.Fatal(err)
				}

				want := make([]frameNote, len(ns.Notes))
				for i, n := range ns.Notes {
					want[i] = frameNote{Start: n.StartFrame, End: n.EndFrame, Pitch: n.Pitch, Amplitude: float32(n.Amplitude)}
					if float64(want[i].Amplitude) != n.Amplitude {
						t.Fatalf("fixture note %d: amplitude %v is not a float32", i, n.Amplitude)
					}
				}

				requireNotes(t, got, want)
			})
		}
	}
}

// requireNotes fails unless got and want hold the same notes in the same
// order, with bit-identical amplitudes; any NaN matches any NaN.
func requireNotes(t *testing.T, got, want []frameNote) {
	t.Helper()

	if len(got) != len(want) {
		t.Errorf("got %d notes, want %d", len(got), len(want))
	}

	for i := range min(len(got), len(want)) {
		g, w := got[i], want[i]
		if g.Start != w.Start || g.End != w.End || g.Pitch != w.Pitch ||
			!sameFloat32(g.Amplitude, w.Amplitude) || g.Bends != nil {
			t.Errorf("note %d: got %+v, want %+v", i, g, w)
		}
	}
}

// sameFloat32 reports whether a and b have the same bits or are both NaN.
func sameFloat32(a, b float32) bool {
	return math.Float32bits(a) == math.Float32bits(b) || (a != a && b != b)
}

// cellRun sets frames [t0, t1) of bin f to v.
type cellRun struct {
	t0, t1, f int
	v         float32
}

// runMatrix builds a row-major (n, noteBins) matrix from runs.
func runMatrix(n int, runs []cellRun) []float32 {
	m := make([]float32, n*noteBins)

	for _, r := range runs {
		for t := r.t0; t < r.t1; t++ {
			m[t*noteBins+r.f] = r.v
		}
	}

	return m
}

// TestDecodeNotesEdgeCases checks corner cases against upstream. The expected
// notes are what upstream output_to_notes_polyphonic (numpy 2.5.3, scipy
// 1.18.1, librosa 1.0.0) returns for the same inputs.
func TestDecodeNotesEdgeCases(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())
	base := noteParams{onsetThresh: 0.5, frameThresh: 0.3, minNoteLen: 2, energyTol: 3}

	with := func(mod func(p *noteParams)) noteParams {
		p := base
		mod(&p)

		return p
	}
	melodia := with(func(p *noteParams) { p.melodiaTrick = true })
	inferMelodia := with(func(p *noteParams) { p.inferOnsets, p.melodiaTrick = true, true })
	freqTest := []cellRun{{2, 15, 10, 0.8}, {2, 15, 40, 0.8}, {2, 15, 80, 0.8}}
	freqTestOnsets := []cellRun{{3, 4, 10, 0.9}, {3, 4, 40, 0.9}, {3, 4, 80, 0.9}}

	tests := []struct {
		name           string
		n              int
		frames, onsets []cellRun
		p              noteParams
		want           []frameNote
	}{
		// Upstream raises on empty input; the port returns no notes.
		{name: "empty", n: 0, p: inferMelodia},
		// All-zero frames: the infer-onsets rescale is 0/0 = NaN everywhere,
		// so there are no onset peaks, and nothing exceeds frame_thresh.
		{name: "silent", n: 20, p: inferMelodia},
		// Constant frames: the rescale is NaN again, which hides the onset
		// peak at row 5, but the melodia pass still finds the note.
		{
			name: "constant frames", n: 20, p: inferMelodia,
			frames: []cellRun{{0, 20, 40, 0.8}}, onsets: []cellRun{{5, 6, 40, 0.9}},
			want: []frameNote{{Start: 0, End: 18, Pitch: 61, Amplitude: 0.8}},
		},
		{
			name: "constant frames without melodia", n: 20, p: with(func(p *noteParams) { p.inferOnsets = true }),
			frames: []cellRun{{0, 20, 40, 0.8}}, onsets: []cellRun{{5, 6, 40, 0.9}},
		},
		{
			name: "onset at row 1", n: 20, p: melodia,
			frames: []cellRun{{0, 10, 30, 0.7}}, onsets: []cellRun{{1, 2, 30, 0.9}},
			want: []frameNote{{Start: 1, End: 10, Pitch: 51, Amplitude: 0.7}},
		},
		// The melodia maximum is the first cell, row 0: the note starts at 0
		// and, with the i-1-k end offset, ends before the last loud frame.
		{
			name: "melodia from row 0", n: 20, p: melodia,
			frames: []cellRun{{0, 8, 30, 0.7}},
			want:   []frameNote{{Start: 0, End: 7, Pitch: 51, Amplitude: float32(0.6999999284744263)}},
		},
		// The backward walk stops before row 0, so the note starts at row 1
		// and row 0 is left for a later, zero-length melodia note.
		{
			name: "melodia backward walk to row 1", n: 20, p: melodia,
			frames: []cellRun{{0, 8, 30, 0.7}, {4, 5, 30, 0.9}},
			want:   []frameNote{{Start: 1, End: 7, Pitch: 51, Amplitude: float32(0.7333333492279053)}},
		},
		{
			name: "melodia zero-length note", n: 20, p: with(func(p *noteParams) { p.melodiaTrick, p.minNoteLen = true, -1 }),
			frames: []cellRun{{0, 8, 30, 0.7}, {4, 5, 30, 0.9}},
			want: []frameNote{
				{Start: 1, End: 7, Pitch: 51, Amplitude: float32(0.7333333492279053)},
				{Start: 0, End: 0, Pitch: 51, Amplitude: nan},
			},
		},
		// The onset walk stops at n-1, so the note ends there.
		{
			name: "onset note up to the last frame", n: 20, p: base,
			frames: []cellRun{{10, 20, 50, 0.7}}, onsets: []cellRun{{11, 12, 50, 0.9}},
			want: []frameNote{{Start: 11, End: 19, Pitch: 71, Amplitude: 0.7}},
		},
		{
			name: "melodia maximum in the last frame", n: 20, p: melodia,
			frames: []cellRun{{12, 20, 50, 0.7}, {19, 20, 50, 0.9}},
			want:   []frameNote{{Start: 12, End: 19, Pitch: 71, Amplitude: float32(0.6999999284744263)}},
		},
		// argrelmax with mode "clip" never reports the first or last row.
		{
			name: "onsets in the first and last row", n: 20, p: base,
			frames: []cellRun{{0, 20, 30, 0.7}}, onsets: []cellRun{{0, 1, 30, 0.9}, {19, 20, 30, 0.9}},
		},
		// peak_thresh_mat is 0 off the peaks, so onset_thresh 0 turns every
		// cell into an onset.
		{
			name: "onset threshold 0", n: 12, p: with(func(p *noteParams) { p.onsetThresh = 0 }),
			frames: []cellRun{{2, 9, 40, 0.7}},
			want: []frameNote{
				{Start: 6, End: 9, Pitch: 61, Amplitude: 0.7},
				{Start: 3, End: 6, Pitch: 61, Amplitude: 0.7},
				{Start: 0, End: 3, Pitch: 61, Amplitude: float32(0.23333333432674408)},
			},
		},
		{
			name: "negative onset threshold", n: 12, p: with(func(p *noteParams) { p.onsetThresh = -0.1 }),
			frames: []cellRun{{2, 9, 40, 0.7}, {3, 10, 60, 0.5}}, onsets: []cellRun{{4, 5, 60, 0.8}},
			want: []frameNote{
				{Start: 7, End: 10, Pitch: 81, Amplitude: 0.5},
				{Start: 6, End: 9, Pitch: 61, Amplitude: 0.7},
				{Start: 4, End: 7, Pitch: 81, Amplitude: 0.5},
				{Start: 3, End: 6, Pitch: 61, Amplitude: 0.7},
				{Start: 1, End: 4, Pitch: 81, Amplitude: float32(0.1666666716337204)},
				{Start: 0, End: 3, Pitch: 61, Amplitude: float32(0.23333333432674408)},
			},
		},
		{
			name: "one frame", n: 1, p: inferMelodia,
			frames: []cellRun{{0, 1, 40, 0.9}}, onsets: []cellRun{{0, 1, 40, 0.9}},
		},
		// The mean of an empty slice is NaN.
		{
			name: "one frame, zero-length note", n: 1, p: with(func(p *noteParams) { p.inferOnsets, p.melodiaTrick, p.minNoteLen = true, true, -1 }),
			frames: []cellRun{{0, 1, 40, 0.9}}, onsets: []cellRun{{0, 1, 40, 0.9}},
			want: []frameNote{{Start: 0, End: 0, Pitch: 61, Amplitude: nan}},
		},
		{
			name: "one frame, onset threshold 0", n: 1, p: with(func(p *noteParams) { p.onsetThresh, p.melodiaTrick, p.minNoteLen = 0, true, -1 }),
			frames: []cellRun{{0, 1, 40, 0.9}}, onsets: []cellRun{{0, 1, 40, 0.9}},
			want: []frameNote{{Start: 0, End: 0, Pitch: 61, Amplitude: nan}},
		},
		{
			name: "two frames", n: 2, p: with(func(p *noteParams) { p.inferOnsets, p.melodiaTrick, p.minNoteLen = true, true, 0 }),
			frames: []cellRun{{0, 2, 40, 0.9}, {1, 2, 41, 0.95}}, onsets: []cellRun{{0, 2, 40, 0.9}},
		},
		// The first maximum is (1, 41); its zero-length note leaves (0, 40)
		// and (1, 40), which tie, so row 0 comes first.
		{
			name: "two frames, zero-length notes", n: 2, p: with(func(p *noteParams) { p.melodiaTrick, p.minNoteLen = true, -1 }),
			frames: []cellRun{{0, 2, 40, 0.9}, {1, 2, 41, 0.95}}, onsets: []cellRun{{0, 2, 40, 0.9}},
			want: []frameNote{
				{Start: 1, End: 1, Pitch: 62, Amplitude: nan},
				{Start: 0, End: 0, Pitch: 61, Amplitude: nan},
				{Start: 1, End: 1, Pitch: 61, Amplitude: nan},
			},
		},
		// np.max(remaining_energy) is NaN, so the melodia loop never runs and
		// the note at bin 70 is not found.
		{
			name: "NaN frame stops melodia", n: 20, p: melodia,
			frames: []cellRun{{2, 12, 40, 0.8}, {2, 15, 70, 0.8}, {15, 16, 10, nan}}, onsets: []cellRun{{3, 4, 40, 0.9}},
			want: []frameNote{{Start: 3, End: 12, Pitch: 61, Amplitude: 0.8}},
		},
		// The onset note zeroes its neighbour bin, NaN included, so melodia
		// runs.
		{
			name: "NaN frame cleared by the onset pass", n: 20, p: melodia,
			frames: []cellRun{{2, 12, 40, 0.8}, {2, 15, 70, 0.8}, {5, 6, 41, nan}}, onsets: []cellRun{{3, 4, 40, 0.9}},
			want: []frameNote{
				{Start: 3, End: 12, Pitch: 61, Amplitude: 0.8},
				{Start: 2, End: 14, Pitch: 91, Amplitude: 0.8},
			},
		},
		// A NaN frame makes max(frame_diff) NaN, so every inferred onset is NaN.
		{
			name: "NaN frame with inferred onsets", n: 20, p: inferMelodia,
			frames: []cellRun{{2, 12, 40, 0.8}, {15, 16, 10, nan}}, onsets: []cellRun{{3, 4, 40, 0.9}},
		},
		// Equal maxima: the first in row-major order wins.
		{
			name: "melodia ties", n: 30, p: melodia,
			frames: []cellRun{{2, 8, 20, 0.6}, {2, 8, 60, 0.6}, {15, 25, 10, 0.6}},
			want: []frameNote{
				{Start: 2, End: 7, Pitch: 41, Amplitude: 0.6},
				{Start: 2, End: 7, Pitch: 81, Amplitude: 0.6},
				{Start: 15, End: 24, Pitch: 31, Amplitude: 0.6},
			},
		},
		{
			name: "inferred onsets", n: 30, p: inferMelodia,
			frames: []cellRun{{5, 20, 40, 0.8}, {10, 25, 45, 0.4}, {3, 9, 70, 0.2}}, onsets: []cellRun{{12, 13, 60, 0.6}},
			want: []frameNote{
				{Start: 5, End: 20, Pitch: 61, Amplitude: float32(0.8000000715255737)},
				{Start: 10, End: 24, Pitch: 66, Amplitude: float32(0.40000003576278687)},
			},
		},
		// 100 Hz and 1000 Hz are bins 22 and 62.
		{
			name: "frequency range", n: 20, p: with(func(p *noteParams) { p.melodiaTrick, p.minFreq, p.maxFreq = true, 100, 1000 }),
			frames: freqTest, onsets: freqTestOnsets,
			want: []frameNote{{Start: 3, End: 15, Pitch: 61, Amplitude: 0.8}},
		},
		// 10 Hz is bin -18; the slice [:, -18:] zeroes the top 18 bins.
		{
			name: "maximum frequency below A0", n: 20, p: with(func(p *noteParams) { p.melodiaTrick, p.maxFreq = true, 10 }),
			frames: freqTest, onsets: freqTestOnsets,
			want: []frameNote{
				{Start: 3, End: 15, Pitch: 61, Amplitude: 0.8},
				{Start: 3, End: 15, Pitch: 31, Amplitude: 0.8},
			},
		},
		// 1e-300 Hz is bin -12945: [:, :-12945] is empty, [:, -12945:] all.
		{
			name: "tiny minimum frequency", n: 20, p: with(func(p *noteParams) { p.melodiaTrick, p.minFreq = true, 1e-300 }),
			frames: []cellRun{{2, 15, 10, 0.8}, {2, 15, 80, 0.8}}, onsets: []cellRun{{3, 4, 10, 0.9}},
			want: []frameNote{
				{Start: 3, End: 15, Pitch: 31, Amplitude: 0.8},
				{Start: 2, End: 14, Pitch: 101, Amplitude: 0.8},
			},
		},
		{
			name: "tiny maximum frequency", n: 20, p: with(func(p *noteParams) { p.melodiaTrick, p.maxFreq = true, 1e-300 }),
			frames: []cellRun{{2, 15, 10, 0.8}, {2, 15, 80, 0.8}}, onsets: []cellRun{{3, 4, 10, 0.9}},
		},
		{
			name: "huge minimum frequency", n: 20, p: with(func(p *noteParams) { p.melodiaTrick, p.minFreq = true, 1e300 }),
			frames: []cellRun{{2, 15, 10, 0.8}, {2, 15, 80, 0.8}}, onsets: []cellRun{{3, 4, 10, 0.9}},
		},
		// Notes in the outermost bins zero only the neighbour that exists.
		{
			name: "edge bins", n: 20, p: melodia,
			frames: []cellRun{{2, 15, 0, 0.8}, {2, 15, 1, 0.5}, {2, 15, 87, 0.8}, {2, 15, 86, 0.5}},
			onsets: []cellRun{{3, 4, 0, 0.9}, {3, 4, 87, 0.9}},
			want: []frameNote{
				{Start: 3, End: 15, Pitch: 108, Amplitude: 0.8},
				{Start: 3, End: 15, Pitch: 21, Amplitude: 0.8},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var d noteDecoder

			got, err := d.decode(runMatrix(tt.n, tt.frames), runMatrix(tt.n, tt.onsets), tt.n, tt.p, nil)
			if err != nil {
				t.Fatal(err)
			}

			requireNotes(t, got, tt.want)
		})
	}
}

func TestNoteFreqIdx(t *testing.T) {
	t.Parallel()

	// int(np.round(librosa.hz_to_midi(hz) - 21)) under librosa 1.0.0.
	tests := []struct {
		hz   float64
		want int
	}{
		{100, 22},
		{1000, 62},
		{440, 48},
		{10, -18},
		{27.5, 0},
		{4186.009044809578, 87},
		{5e-324, -12945},
		{math.MaxFloat64, 12231},
	}

	for _, tt := range tests {
		if got := noteFreqIdx(tt.hz); got != tt.want {
			t.Errorf("noteFreqIdx(%v) = %d, want %d", tt.hz, got, tt.want)
		}
	}
}

func TestPySliceBound(t *testing.T) {
	t.Parallel()

	tests := []struct{ idx, want int }{
		{0, 0}, {22, 22}, {88, 88}, {100, 88}, {-1, 87}, {-18, 70}, {-88, 0}, {-89, 0}, {-12945, 0},
	}

	for _, tt := range tests {
		if got := pySliceBound(tt.idx, noteBins); got != tt.want {
			t.Errorf("pySliceBound(%d) = %d, want %d", tt.idx, got, tt.want)
		}
	}
}

// TestDecodeConstrainsInPlace checks that decode zeroes the columns outside
// the frequency range in the caller's slices, as constrain_frequency does.
func TestDecodeConstrainsInPlace(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		minFreq, maxFreq float64
		keep0, keep1     int // the bins [keep0, keep1) survive
	}{
		{"none", 0, 0, 0, noteBins},
		{"100 to 1000 Hz", 100, 1000, 22, 62},
		{"maximum below A0", 0, 10, 0, 70},
		{"minimum above C8", 5000, 0, 90, noteBins},
		{"empty range", 1000, 100, 62, 22},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			const n = 5

			frames := make([]float32, n*noteBins)
			onsets := make([]float32, n*noteBins)

			for i := range frames {
				frames[i], onsets[i] = 0.25, 0.125
			}

			var d noteDecoder

			_, err := d.decode(frames, onsets, n, noteParams{onsetThresh: 0.5, frameThresh: 0.3, minFreq: tt.minFreq, maxFreq: tt.maxFreq}, nil)
			if err != nil {
				t.Fatal(err)
			}

			for i := range frames {
				wantF, wantO := float32(0), float32(0)
				if f := i % noteBins; f >= tt.keep0 && f < tt.keep1 {
					wantF, wantO = 0.25, 0.125
				}

				if frames[i] != wantF || onsets[i] != wantO {
					t.Fatalf("cell %d: frames %v onsets %v, want %v %v", i, frames[i], onsets[i], wantF, wantO)
				}
			}
		})
	}
}

func TestDecodeNotesErrors(t *testing.T) {
	t.Parallel()

	ok := noteParams{onsetThresh: 0.5, frameThresh: 0.3, minNoteLen: 11, melodiaTrick: true, energyTol: noteEnergyTol}

	with := func(mod func(p *noteParams)) noteParams {
		p := ok
		mod(&p)

		return p
	}

	tests := []struct {
		name    string
		n, size int
		p       noteParams
	}{
		{"negative minimum frequency", 4, 4 * noteBins, with(func(p *noteParams) { p.minFreq = -100 })},
		{"NaN minimum frequency", 4, 4 * noteBins, with(func(p *noteParams) { p.minFreq = math.NaN() })},
		{"infinite minimum frequency", 4, 4 * noteBins, with(func(p *noteParams) { p.minFreq = math.Inf(1) })},
		{"negative maximum frequency", 4, 4 * noteBins, with(func(p *noteParams) { p.maxFreq = -1 })},
		{"NaN maximum frequency", 4, 4 * noteBins, with(func(p *noteParams) { p.maxFreq = math.NaN() })},
		{"-Inf maximum frequency", 4, 4 * noteBins, with(func(p *noteParams) { p.maxFreq = math.Inf(-1) })},
		{"negative frame threshold with melodia", 4, 4 * noteBins, with(func(p *noteParams) { p.frameThresh = -0.1 })},
		{"negative frame count", -1, 0, ok},
		{"short input", 4, 4*noteBins - 1, ok},
		{"long input", 4, 5 * noteBins, ok},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var d noteDecoder

			_, err := d.decode(make([]float32, tt.size), make([]float32, tt.size), tt.n, tt.p, nil)
			if err == nil {
				t.Fatal("got no error")
			}
		})
	}

	// Without melodia a negative frame threshold terminates upstream too.
	var d noteDecoder

	_, err := d.decode(make([]float32, 4*noteBins), make([]float32, 4*noteBins), 4,
		with(func(p *noteParams) { p.frameThresh, p.melodiaTrick = -0.1, false }), nil)
	if err != nil {
		t.Fatal(err)
	}
}

// defaultNoteParams are the note-creation defaults of docs/port-spec.md §5.
var defaultNoteParams = noteParams{
	onsetThresh: 0.5, frameThresh: 0.3, minNoteLen: 11,
	inferOnsets: true, melodiaTrick: true, energyTol: noteEnergyTol,
}

// syntheticPosteriorgrams builds deterministic note and onset outputs of
// nFrames frames: noise below the default thresholds with one sustained note
// per 10 frames on average. Two thirds of the notes have an onset peak at
// their start; the melodia pass has to find the others.
func syntheticPosteriorgrams(nFrames int) (frames, onsets []float32) {
	rng := rand.New(rand.NewPCG(1, 2))
	frames = make([]float32, nFrames*noteBins)
	onsets = make([]float32, nFrames*noteBins)

	for i := range frames {
		frames[i] = float32(math.Pow(rng.Float64(), 8) * 0.25)
		onsets[i] = float32(math.Pow(rng.Float64(), 8) * 0.25)
	}

	for range nFrames / 10 {
		f := rng.IntN(noteBins)
		t0 := rng.IntN(nFrames)
		t1 := min(nFrames, t0+10+rng.IntN(150))
		level := 0.4 + 0.5*rng.Float64()

		for t := t0; t < t1; t++ {
			frames[t*noteBins+f] = float32(level * (0.9 + 0.1*rng.Float64()))
		}

		if rng.IntN(3) > 0 {
			onsets[t0*noteBins+f] = float32(0.5 + 0.5*rng.Float64())
		}
	}

	return frames, onsets
}

// TestNoteDecoderReuse checks that a warm decoder gives the same notes again.
func TestNoteDecoderReuse(t *testing.T) {
	t.Parallel()

	frames, onsets, nFrames := fixturePosteriorgrams(t, fixture.MustOpen(t, "synth"))
	small, smallOnsets := syntheticPosteriorgrams(50)

	var d noteDecoder

	first, err := d.decode(frames, onsets, nFrames, defaultNoteParams, nil)
	if err != nil {
		t.Fatal(err)
	}

	first = slices.Clone(first)

	// A smaller input in between must not leave stale state behind.
	_, err = d.decode(small, smallOnsets, 50, defaultNoteParams, nil)
	if err != nil {
		t.Fatal(err)
	}

	dst := make([]frameNote, 0, 2*len(first))

	again, err := d.decode(frames, onsets, nFrames, defaultNoteParams, dst)
	if err != nil {
		t.Fatal(err)
	}

	requireNotes(t, again, first)
}

// TestNoteDecoderZeroAllocs checks that a warm decoder allocates nothing when
// dst has room. AllocsPerRun must not run in a parallel test.
func TestNoteDecoderZeroAllocs(t *testing.T) {
	frames, onsets, nFrames := fixturePosteriorgrams(t, fixture.MustOpen(t, "synth"))

	var d noteDecoder

	dst, err := d.decode(frames, onsets, nFrames, defaultNoteParams, nil)
	if err != nil {
		t.Fatal(err)
	}

	allocs := testing.AllocsPerRun(10, func() {
		dst, err = d.decode(frames, onsets, nFrames, defaultNoteParams, dst)
	})

	if err != nil {
		t.Fatal(err)
	}

	if allocs != 0 {
		t.Errorf("warm decode allocates %v times, want 0", allocs)
	}
}

func BenchmarkNoteDecode(b *testing.B) {
	frames, onsets, nFrames := fixturePosteriorgrams(b, fixture.MustOpen(b, "synth"))
	longFrames, longOnsets := syntheticPosteriorgrams(15500)

	for _, bc := range []struct {
		name           string
		frames, onsets []float32
		n              int
	}{
		{"synth", frames, onsets, nFrames},
		{"3min", longFrames, longOnsets, 15500},
	} {
		b.Run(bc.name, func(b *testing.B) {
			var d noteDecoder

			dst, err := d.decode(bc.frames, bc.onsets, bc.n, defaultNoteParams, nil)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				dst, err = d.decode(bc.frames, bc.onsets, bc.n, defaultNoteParams, dst)
				if err != nil {
					b.Fatal(err)
				}
			}

			b.ReportMetric(float64(len(dst)), "notes")
		})
	}
}
