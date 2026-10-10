package fixture

import (
	"bytes"
	"maps"
	"path/filepath"
	"slices"
	"testing"
)

// csvHeader is the first line of upstream save_note_events output. Python's
// csv.writer terminates lines with CRLF by default, and upstream keeps that.
const csvHeader = "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\r\n"

// wantClips holds the clips the generator must produce with their window and
// frame counts.
var wantClips = map[string]struct{ windows, frames int }{
	"synth": {4, 430},
	"vocal": {3, 344},
}

// wantVariants are the note-creation variants of every clip.
var wantVariants = []string{"default", "freq_range", "low_thresholds", "no_infer_onsets", "no_melodia"}

// wantArrays is the single table of expected arrays: dtype, shape (derived from
// the manifest counts) and axes. Adjust it here when the generator changes.
func wantArrays(m *Manifest) map[string]Array {
	nW, nF := m.NWindows, m.NFrames
	wtf := []string{"window", "time", "freq"}
	tf := []string{"time", "freq"}
	f32 := func(axes []string, shape ...int) Array { return Array{DType: DTypeFloat32, Shape: shape, Axes: axes} }

	return map[string]Array{
		"windows":           f32([]string{"window", "sample"}, nW, 43844),
		"cqt":               f32(wtf, nW, 172, 309),
		"normalized_log":    f32(wtf, nW, 172, 309),
		"cqt_bn":            f32(wtf, nW, 172, 309),
		HarmonicStack:       f32(harmonicStackAxes, nW, 172, 8, 264),
		"contour":           f32(wtf, nW, 172, 264),
		"note":              f32(wtf, nW, 172, 88),
		"onset":             f32(wtf, nW, 172, 88),
		"unwrapped_contour": f32(tf, nF, 264),
		"unwrapped_note":    f32(tf, nF, 88),
		"unwrapped_onset":   f32(tf, nF, 88),
		"frame_times":       {DType: DTypeFloat64, Shape: []int{nF}, Axes: []string{"time"}},
	}
}

func TestRealFixtures(t *testing.T) {
	t.Parallel()

	names, err := Names()
	if err != nil {
		t.Fatal(err)
	}

	if got, want := names, slices.Sorted(maps.Keys(wantClips)); !slices.Equal(got, want) {
		t.Fatalf("Names() = %v, want %v", got, want)
	}

	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			c := MustOpen(t, name)
			checkManifest(t, c)
			checkAudio(t, c)
			checkArrays(t, c)
			checkHarmonicStackNHWC(t, c)

			for _, v := range slices.Sorted(maps.Keys(c.Manifest.Variants)) {
				checkVariant(t, c, v)
			}
		})
	}
}

func checkManifest(t *testing.T, c *Clip) {
	t.Helper()

	m := &c.Manifest
	if m.Clip != filepath.Base(c.Dir) {
		t.Errorf("clip = %q, directory %q", m.Clip, c.Dir)
	}

	if got := m.Upstream["basic_pitch"]; got != "0.4.0" {
		t.Errorf("upstream basic_pitch = %q, want 0.4.0", got)
	}

	want := Constants{
		SampleRate: SampleRate, FFTHop: 256, WindowSamples: 43844, OverlapSamples: 7680,
		HopSamples: 36164, FramesPerWindow: 172, OverlapFrames: 30,
	}
	if m.Constants != want {
		t.Errorf("constants = %+v, want %+v", m.Constants, want)
	}

	counts := wantClips[m.Clip]
	if m.NWindows != counts.windows || m.NFrames != counts.frames || m.NSamples <= 0 {
		t.Errorf("n_windows %d, n_frames %d, n_samples %d; want %d windows and %d frames",
			m.NWindows, m.NFrames, m.NSamples, counts.windows, counts.frames)
	}

	if got := slices.Sorted(maps.Keys(m.Variants)); !slices.Equal(got, wantVariants) {
		t.Errorf("variants = %v, want %v", got, wantVariants)
	}
}

func checkAudio(t *testing.T, c *Clip) {
	t.Helper()

	audio, err := c.Audio()
	if err != nil {
		t.Fatal(err)
	}

	if len(audio) != c.Manifest.NSamples {
		t.Errorf("len(Audio()) = %d, n_samples = %d", len(audio), c.Manifest.NSamples)
	}
}

func checkArrays(t *testing.T, c *Clip) {
	t.Helper()

	m := &c.Manifest
	want := wantArrays(m)

	for _, name := range slices.Sorted(maps.Keys(want)) {
		if _, ok := m.Arrays[name]; !ok {
			t.Errorf("manifest lacks array %q", name)
		}
	}

	for _, name := range slices.Sorted(maps.Keys(m.Arrays)) {
		w, ok := want[name]
		if !ok {
			t.Errorf("unexpected array %q (add it to wantArrays)", name)

			continue
		}

		if got := m.Arrays[name].Axes; !slices.Equal(got, w.Axes) {
			t.Errorf("array %q axes = %q, want %q", name, got, w.Axes)
		}

		var (
			shape []int
			err   error
			n     int
		)

		switch w.DType {
		case DTypeFloat32:
			var data []float32

			data, shape, err = c.Float32(name)
			n = len(data)
		default:
			var data []float64

			data, shape, err = c.Float64(name)
			n = len(data)
		}

		if err != nil {
			t.Errorf("array %q: %v", name, err)

			continue
		}

		if !slices.Equal(shape, w.Shape) {
			t.Errorf("array %q shape = %v, want %v", name, shape, w.Shape)
		}

		if count, _ := elements(shape); n != count {
			t.Errorf("array %q has %d elements, shape %v", name, n, shape)
		}

		if name == "frame_times" && n != m.NFrames {
			t.Errorf("len(frame_times) = %d, n_frames = %d", n, m.NFrames)
		}
	}
}

// checkHarmonicStackNHWC spot-checks the transposed last window against the
// stored layout.
func checkHarmonicStackNHWC(t *testing.T, c *Clip) {
	t.Helper()

	w := c.Manifest.NWindows - 1

	stored, shape, err := c.Window(HarmonicStack, w)
	if err != nil {
		t.Fatal(err)
	}

	nhwc, nhwcShape, err := c.HarmonicStackNHWC(w)
	if err != nil {
		t.Fatal(err)
	}

	nT, nH, nF := shape[0], shape[1], shape[2]
	if !slices.Equal(nhwcShape, []int{nT, nF, nH}) || nhwcShape[1] != 264 || nhwcShape[2] != 8 {
		t.Fatalf("NHWC shape = %v, stored window shape %v", nhwcShape, shape)
	}

	for _, p := range [][3]int{{0, 0, 0}, {nT - 1, nF - 1, nH - 1}, {85, 131, 3}, {17, 200, 7}} {
		tt, f, h := p[0], p[1], p[2]
		if got, want := nhwc[(tt*nF+f)*nH+h], stored[(tt*nH+h)*nF+f]; got != want {
			t.Errorf("NHWC(%d, %d, %d) = %v, stored %v", tt, f, h, got, want)
		}
	}
}

func checkVariant(t *testing.T, c *Clip, variant string) {
	t.Helper()

	ns, err := c.Notes(variant)
	if err != nil {
		t.Fatal(err)
	}

	if ns.Variant != variant {
		t.Errorf("%s: notes JSON names variant %q", variant, ns.Variant)
	}

	if len(ns.Notes) == 0 {
		t.Errorf("%s: no notes", variant)
	}

	if len(ns.NoteEvents) != len(ns.Notes) {
		t.Fatalf("%s: %d notes, %d note events", variant, len(ns.Notes), len(ns.NoteEvents))
	}

	for i, n := range ns.Notes {
		if n.Pitch < 21 || n.Pitch > 108 {
			t.Errorf("%s: note %d pitch %d outside [21, 108]", variant, i, n.Pitch)
		}

		if n.StartFrame < 0 || n.EndFrame <= n.StartFrame || n.EndFrame > c.Manifest.NFrames {
			t.Errorf("%s: note %d frames [%d, %d) outside [0, %d)", variant, i, n.StartFrame, n.EndFrame, c.Manifest.NFrames)
		}

		e := ns.NoteEvents[i]
		if e.Pitch != n.Pitch || e.Amplitude != n.Amplitude {
			t.Errorf("%s: note event %d = %+v, note %+v", variant, i, e, n)
		}

		if got, want := len(e.PitchBends), n.EndFrame-n.StartFrame; e.PitchBends == nil || got != want {
			t.Errorf("%s: note event %d has %d pitch bends, want %d", variant, i, got, want)
		}
	}

	for _, md := range []MIDIData{ns.MIDI, ns.MIDIMultiplePitchBends} {
		for _, inst := range md.Instruments {
			for _, pb := range inst.PitchBends {
				if pb.Pitch < -8192 || pb.Pitch > 8191 {
					t.Errorf("%s: MIDI pitch bend %d outside [-8192, 8191]", variant, pb.Pitch)
				}
			}
		}
	}

	csv, err := c.CSV(variant)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.HasPrefix(csv, []byte(csvHeader)) {
		t.Errorf("%s: CSV starts with %q", variant, csv[:min(len(csv), len(csvHeader))])
	}

	for _, multiple := range []bool{false, true} {
		mid, err := c.MIDI(variant, multiple)
		if err != nil {
			t.Fatal(err)
		}

		if !bytes.HasPrefix(mid, []byte("MThd")) {
			t.Errorf("%s (multiple=%v): MIDI does not start with MThd", variant, multiple)
		}
	}
}
