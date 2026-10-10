package fixture

import (
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// Values of the synthetic arrays written by writeTinyClip.
var (
	tinyA32 = []float32{0, 1, -2.5, 3.25, float32(math.Pi), -0}
	tinyB32 = []float32{1, 2, 3, 4, 5, 6, 7, 8}
	tinyC64 = []float64{0.1, -1e-300, math.MaxFloat64}
	tinyPCM = []int16{0, 1, -1, 32767, -32768}
	// tinyHS is a harmonic stack with stored shape (window 2, time 2,
	// harmonic 3, freq 4) whose element values are their own flat index.
	tinyHS = func() []float32 {
		v := make([]float32, 2*2*3*4)
		for i := range v {
			v[i] = float32(i)
		}

		return v
	}()
)

const tinyNotes = `{
  "variant": "default",
  "options": {"onset_threshold": 0.5, "frame_threshold": 0.3, "minimum_note_length_ms": 127.7,
    "min_note_len_frames": 11, "infer_onsets": true, "melodia_trick": true,
    "minimum_frequency": null, "maximum_frequency": 1000.0, "include_pitch_bends": true,
    "multiple_pitch_bends": false, "midi_tempo": 120, "energy_tol": 11},
  "notes": [{"start_frame": 12, "end_frame": 16, "pitch": 60, "amplitude": 0.53125}],
  "note_events": [{"start_s": 0.13, "end_s": 0.46, "pitch": 60, "amplitude": 0.53125, "pitch_bends": [0, 0, -1, 0]}],
  "midi": {"instruments": [{"program": 4, "is_drum": false,
    "notes": [{"start": 0.13, "end": 0.46, "pitch": 60, "velocity": 67}],
    "pitch_bends": [{"time": 0.13, "pitch": -8192}]}]},
  "midi_multiple_pitch_bends": {"instruments": []}
}`

const tinyCSV = "start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\n0.13,0.46,60,67,0,0,-1,0\n"

func le32(v []float32) []byte {
	b := make([]byte, 0, 4*len(v))
	for _, x := range v {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(x))
	}

	return b
}

func le64(v []float64) []byte {
	b := make([]byte, 0, 8*len(v))
	for _, x := range v {
		b = binary.LittleEndian.AppendUint64(b, math.Float64bits(x))
	}

	return b
}

func sha(b []byte) string {
	s := sha256.Sum256(b)

	return hex.EncodeToString(s[:])
}

func gz(t *testing.T, b []byte) []byte {
	t.Helper()

	var buf bytes.Buffer

	zw := gzip.NewWriter(&buf)

	_, err := zw.Write(b)
	if err != nil {
		t.Fatal(err)
	}

	err = zw.Close()
	if err != nil {
		t.Fatal(err)
	}

	return buf.Bytes()
}

func writeFile(t *testing.T, dir, name string, b []byte) {
	t.Helper()

	err := os.WriteFile(filepath.Join(dir, name), b, 0o600)
	if err != nil {
		t.Fatal(err)
	}
}

// tinyManifest returns a manifest for writeTinyClip as a generic JSON object,
// spelled with the on-disk keys so the test does not depend on the struct tags.
func tinyManifest() map[string]any {
	a32, b32, c64, hs := le32(tinyA32), le32(tinyB32), le64(tinyC64), le32(tinyHS)
	window, wf := []string{"window", "time", "freq"}, []string{"window", "freq"}

	return map[string]any{
		"format_version": 1,
		"clip":           "tiny",
		"description":    "synthetic loader test",
		"source":         map[string]any{"kind": "test"},
		"upstream":       map[string]any{"basic_pitch": "0.4.0"},
		"model":          map[string]any{"path": "nmp.onnx", "sha256": strings.Repeat("0", 64)},
		"constants": map[string]any{
			"sample_rate": 22050, "fft_hop": 256, "window_samples": 43844, "overlap_samples": 7680,
			"hop_samples": 36164, "frames_per_window": 172, "overlap_frames": 30,
		},
		"n_samples": len(tinyPCM),
		"n_windows": 2,
		"n_frames":  3,
		"wav":       "audio.wav",
		"arrays": map[string]any{
			"a32": map[string]any{"file": "a.f32", "dtype": "float32", "shape": []int{2, 3}, "axes": wf, "sha256": sha(a32)},
			"b32": map[string]any{
				"file": "b.f32.gz", "dtype": "float32", "shape": []int{2, 2, 2}, "axes": window, "sha256": sha(b32),
			},
			"c64": map[string]any{"file": "c.f64", "dtype": "float64", "shape": []int{3}, "axes": []string{"time"}, "sha256": sha(c64)},
			HarmonicStack: map[string]any{
				"file": "hs.f32", "dtype": "float32", "shape": []int{2, 2, 3, 4},
				"axes": harmonicStackAxes, "sha256": sha(hs),
			},
		},
		"variants": map[string]any{
			"default": map[string]any{
				"options": map[string]any{"onset_threshold": 0.5},
				"notes":   "notes.json", "csv": "notes.csv", "midi": "notes.mid", "midi_multiple_pitch_bends": "multi.mid",
			},
		},
		"checks": map[string]any{"peak": 1.5},
	}
}

func writeManifest(t *testing.T, dir string, m map[string]any) {
	t.Helper()

	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}

	writeFile(t, dir, ManifestFile, b)
}

// writeTinyClip writes a complete synthetic clip into a temporary directory.
// edit, when non-nil, modifies the manifest before it is written.
func writeTinyClip(t *testing.T, edit func(m map[string]any)) string {
	t.Helper()

	dir := t.TempDir()
	writeFile(t, dir, "a.f32", le32(tinyA32))
	writeFile(t, dir, "b.f32.gz", gz(t, le32(tinyB32)))
	writeFile(t, dir, "c.f64", le64(tinyC64))
	writeFile(t, dir, "notes.json", []byte(tinyNotes))
	writeFile(t, dir, "notes.csv", []byte(tinyCSV))
	writeFile(t, dir, "notes.mid", []byte("MThd single"))
	writeFile(t, dir, "multi.mid", []byte("MThd multi"))
	writeFile(t, dir, "hs.f32", le32(tinyHS))
	writeFile(t, dir, "audio.wav", wavBytes(tinyPCM, nil))

	m := tinyManifest()
	if edit != nil {
		edit(m)
	}

	writeManifest(t, dir, m)

	return dir
}

// wavBytes builds a 16-bit mono PCM WAV at SampleRate. extra chunks are
// inserted between "fmt " and "data".
func wavBytes(pcm []int16, extra []byte) []byte {
	return wavWith(1, 1, SampleRate, 16, pcm, extra)
}

func wavWith(format, channels uint16, rate uint32, bits uint16, pcm []int16, extra []byte) []byte {
	var body []byte

	body = append(body, "WAVEfmt "...)
	body = binary.LittleEndian.AppendUint32(body, 16)
	body = binary.LittleEndian.AppendUint16(body, format)
	body = binary.LittleEndian.AppendUint16(body, channels)
	body = binary.LittleEndian.AppendUint32(body, rate)
	body = binary.LittleEndian.AppendUint32(body, rate*uint32(channels)*uint32(bits)/8)
	body = binary.LittleEndian.AppendUint16(body, channels*bits/8)
	body = binary.LittleEndian.AppendUint16(body, bits)
	body = append(body, extra...)
	body = append(body, "data"...)

	body = binary.LittleEndian.AppendUint32(body, uint32(2*len(pcm)))
	for _, v := range pcm {
		body = binary.LittleEndian.AppendUint16(body, uint16(v))
	}

	out := append([]byte("RIFF"), binary.LittleEndian.AppendUint32(nil, uint32(len(body)))...)

	return append(out, body...)
}

func array(m map[string]any, name string) map[string]any {
	arrays, _ := m["arrays"].(map[string]any)
	a, _ := arrays[name].(map[string]any)

	return a
}

func TestLoadTinyClip(t *testing.T) {
	t.Parallel()

	c, err := Load(writeTinyClip(t, nil))
	if err != nil {
		t.Fatal(err)
	}

	m := c.Manifest
	if m.Clip != "tiny" || m.NWindows != 2 || m.NFrames != 3 || m.NSamples != len(tinyPCM) || m.WAV != "audio.wav" {
		t.Errorf("manifest scalars = %+v", m)
	}

	if m.Constants.HopSamples != 36164 || m.Constants.FramesPerWindow != 172 {
		t.Errorf("constants = %+v", m.Constants)
	}

	if m.Upstream["basic_pitch"] != "0.4.0" {
		t.Errorf("upstream = %v", m.Upstream)
	}

	if string(m.Checks) != `{"peak":1.5}` {
		t.Errorf("checks = %s", m.Checks)
	}

	tests := []struct {
		name  string
		want  []float32
		shape []int
	}{
		{"a32", tinyA32, []int{2, 3}},
		{"b32", tinyB32, []int{2, 2, 2}},
	}
	for _, tc := range tests {
		got, shape, err := c.Float32(tc.name)
		if err != nil {
			t.Fatalf("Float32(%q): %v", tc.name, err)
		}

		if !slices.Equal(shape, tc.shape) {
			t.Errorf("Float32(%q) shape = %v, want %v", tc.name, shape, tc.shape)
		}

		if !slices.Equal(le32(got), le32(tc.want)) {
			t.Errorf("Float32(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}

	c64, shape, err := c.Float64("c64")
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(shape, []int{3}) || !slices.Equal(le64(c64), le64(tinyC64)) {
		t.Errorf("Float64(c64) = %v %v", c64, shape)
	}

	win, shape, err := c.Window("b32", 1)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(win, tinyB32[4:]) || !slices.Equal(shape, []int{2, 2}) {
		t.Errorf("Window(b32, 1) = %v %v", win, shape)
	}
}

func TestTinyClipFiles(t *testing.T) {
	t.Parallel()

	c, err := Load(writeTinyClip(t, nil))
	if err != nil {
		t.Fatal(err)
	}

	ns, err := c.Notes("default")
	if err != nil {
		t.Fatal(err)
	}

	if ns.Variant != "default" || len(ns.Notes) != 1 || ns.Notes[0] != (Note{12, 16, 60, 0.53125}) {
		t.Errorf("notes = %+v", ns)
	}

	if ns.Options.MinimumFrequency != nil || ns.Options.MaximumFrequency == nil || *ns.Options.MaximumFrequency != 1000 {
		t.Errorf("frequency options = %v %v", ns.Options.MinimumFrequency, ns.Options.MaximumFrequency)
	}

	if !slices.Equal(ns.NoteEvents[0].PitchBends, []int{0, 0, -1, 0}) {
		t.Errorf("event bends = %v", ns.NoteEvents[0].PitchBends)
	}

	if bend := ns.MIDI.Instruments[0].PitchBends[0]; bend.Pitch != -8192 || ns.MIDI.Instruments[0].Program != 4 {
		t.Errorf("midi = %+v", ns.MIDI)
	}

	files := []struct {
		name string
		read func() ([]byte, error)
		want string
	}{
		{"csv", func() ([]byte, error) { return c.CSV("default") }, tinyCSV},
		{"midi", func() ([]byte, error) { return c.MIDI("default", false) }, "MThd single"},
		{"midi multiple", func() ([]byte, error) { return c.MIDI("default", true) }, "MThd multi"},
		{"wav", c.WAV, string(wavBytes(tinyPCM, nil))},
	}
	for _, f := range files {
		got, err := f.read()
		if err != nil {
			t.Fatalf("%s: %v", f.name, err)
		}

		if string(got) != f.want {
			t.Errorf("%s = %q, want %q", f.name, got, f.want)
		}
	}
}

func TestLoadErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		edit func(m map[string]any)
		want string
	}{
		{"format version", func(m map[string]any) { m["format_version"] = 2 }, "format_version 2"},
		{"unknown field", func(m map[string]any) { m["extra"] = true }, `unknown field "extra"`},
		{"unknown dtype", func(m map[string]any) { array(m, "a32")["dtype"] = "int16" }, `unknown dtype "int16"`},
		{"dtype extension", func(m map[string]any) { array(m, "c64")["dtype"] = "float32" }, "needs a .f32 file"},
		{"empty shape", func(m map[string]any) { array(m, "a32")["shape"] = []int{} }, "empty shape"},
		{"missing axes", func(m map[string]any) { delete(array(m, "a32"), "axes") }, "do not match shape"},
		{"short axes", func(m map[string]any) { array(m, "b32")["axes"] = []string{"window"} }, "do not match shape"},
		{"zero dimension", func(m map[string]any) { array(m, "a32")["shape"] = []int{2, 0} }, "non-positive dimension"},
		{"bad sha", func(m map[string]any) { array(m, "a32")["sha256"] = "ABC" }, "64 lowercase hex"},
		{"escaping file", func(m map[string]any) { array(m, "a32")["file"] = "../a.f32" }, "not a local path"},
		{"escaping wav", func(m map[string]any) { m["wav"] = "/etc/passwd" }, "not a local path"},
		{"escaping notes", func(m map[string]any) {
			m["variants"].(map[string]any)["default"].(map[string]any)["notes"] = "../n.json"
		}, "not a local path"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, err := Load(writeTinyClip(t, tc.edit))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("Load error = %v, want it to contain %q", err, tc.want)
			}
		})
	}
}

func TestReadErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		edit    func(m map[string]any)
		corrupt func(t *testing.T, dir string)
		read    func(c *Clip) error
		want    string
	}{
		{
			name: "unknown array",
			read: func(c *Clip) error { _, _, err := c.Float32("nope"); return err },
			want: `unknown array "nope"`,
		},
		{
			name: "dtype mismatch float32",
			read: func(c *Clip) error { _, _, err := c.Float32("c64"); return err },
			want: "dtype float64, not float32",
		},
		{
			name: "dtype mismatch float64",
			read: func(c *Clip) error { _, _, err := c.Float64("a32"); return err },
			want: "dtype float32, not float64",
		},
		{
			name: "size mismatch",
			edit: func(m map[string]any) { array(m, "a32")["shape"] = []int{2, 4} },
			read: func(c *Clip) error { _, _, err := c.Float32("a32"); return err },
			want: "size 24 bytes, shape [2 4] needs 32",
		},
		{
			name: "sha mismatch",
			edit: func(m map[string]any) { array(m, "a32")["sha256"] = strings.Repeat("ab", 32) },
			read: func(c *Clip) error { _, _, err := c.Float32("a32"); return err },
			want: "sha256",
		},
		{
			name: "sha mismatch gzip",
			corrupt: func(t *testing.T, dir string) {
				t.Helper()
				writeFile(t, dir, "b.f32.gz", gz(t, le32([]float32{1, 2, 3, 4, 5, 6, 7, 9})))
			},
			read: func(c *Clip) error { _, _, err := c.Float32("b32"); return err },
			want: "sha256",
		},
		{
			name: "corrupt gzip",
			corrupt: func(t *testing.T, dir string) {
				t.Helper()
				writeFile(t, dir, "b.f32.gz", le32(tinyB32))
			},
			read: func(c *Clip) error { _, _, err := c.Float32("b32"); return err },
			want: "gzip",
		},
		{
			name: "missing file",
			corrupt: func(t *testing.T, dir string) {
				t.Helper()

				err := os.Remove(filepath.Join(dir, "c.f64"))
				if err != nil {
					t.Fatal(err)
				}
			},
			read: func(c *Clip) error { _, _, err := c.Float64("c64"); return err },
			want: "read c.f64",
		},
		{
			name: "window of 1-D array",
			read: func(c *Clip) error { _, _, err := c.Window("c64", 0); return err },
			want: "dtype float64",
		},
		{
			name: "window out of range",
			read: func(c *Clip) error { _, _, err := c.Window("a32", 2); return err },
			want: "window 2 out of range [0, 2)",
		},
		{
			name: "harmonic stack axes",
			edit: func(m map[string]any) {
				array(m, HarmonicStack)["axes"] = []string{"window", "time", "freq", "harmonic"}
			},
			read: func(c *Clip) error { _, _, err := c.HarmonicStackNHWC(0); return err },
			want: "has axes",
		},
		{
			name: "harmonic stack window",
			read: func(c *Clip) error { _, _, err := c.HarmonicStackNHWC(2); return err },
			want: "window 2 out of range",
		},
		{
			name: "audio",
			corrupt: func(t *testing.T, dir string) {
				t.Helper()
				writeFile(t, dir, "audio.wav", wavWith(1, 2, SampleRate, 16, tinyPCM, nil))
			},
			read: func(c *Clip) error { _, err := c.Audio(); return err },
			want: "decode audio.wav: format tag 1, 2 channels",
		},
		{
			name: "missing variant notes",
			read: func(c *Clip) error { _, err := c.Notes("nope"); return err },
			want: `unknown variant "nope"`,
		},
		{
			name: "missing variant csv",
			read: func(c *Clip) error { _, err := c.CSV("nope"); return err },
			want: `unknown variant "nope"`,
		},
		{
			name: "missing variant midi",
			read: func(c *Clip) error { _, err := c.MIDI("nope", true); return err },
			want: `unknown variant "nope"`,
		},
		{
			name: "notes unknown field",
			corrupt: func(t *testing.T, dir string) {
				t.Helper()
				writeFile(t, dir, "notes.json", []byte(`{"variant": "default", "bogus": 1}`))
			},
			read: func(c *Clip) error { _, err := c.Notes("default"); return err },
			want: `unknown field "bogus"`,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			dir := writeTinyClip(t, tc.edit)
			if tc.corrupt != nil {
				tc.corrupt(t, dir)
			}

			c, err := Load(dir)
			if err != nil {
				t.Fatal(err)
			}

			err = tc.read(c)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want it to contain %q", err, tc.want)
			}
		})
	}
}

func TestOpenRejectsNonLocalName(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"../synth", "/tmp", ""} {
		_, err := Open(name)
		if err == nil {
			t.Errorf("Open(%q) succeeded", name)
		}
	}
}

func TestRoot(t *testing.T) {
	t.Parallel()

	root, err := Root()
	if err != nil {
		t.Fatal(err)
	}

	if filepath.Base(root) != "fixtures" || filepath.Base(filepath.Dir(root)) != "testdata" {
		t.Fatalf("Root() = %s", root)
	}

	_, err = os.Stat(filepath.Join(root, "..", "..", "go.mod"))
	if err != nil {
		t.Fatalf("Root() = %s is not below the module root: %v", root, err)
	}
}

func TestHarmonicStackNHWC(t *testing.T) {
	t.Parallel()

	c, err := Load(writeTinyClip(t, nil))
	if err != nil {
		t.Fatal(err)
	}

	const nT, nH, nF = 2, 3, 4

	for w := range 2 {
		got, shape, err := c.HarmonicStackNHWC(w)
		if err != nil {
			t.Fatal(err)
		}

		if !slices.Equal(shape, []int{nT, nF, nH}) {
			t.Fatalf("shape = %v, want [%d %d %d]", shape, nT, nF, nH)
		}

		for tt := range nT {
			for f := range nF {
				for h := range nH {
					want := tinyHS[((w*nT+tt)*nH+h)*nF+f]
					if v := got[(tt*nF+f)*nH+h]; v != want {
						t.Errorf("window %d (t %d, f %d, h %d) = %v, want %v", w, tt, f, h, v, want)
					}
				}
			}
		}
	}
}

func TestDecodeWAV(t *testing.T) {
	t.Parallel()

	want := []float32{0, 1.0 / 32768, -1.0 / 32768, 32767.0 / 32768, -1}
	// An odd-sized unknown chunk exercises the RIFF pad byte.
	list := append([]byte("LIST\x03\x00\x00\x00abc"), 0)

	tests := []struct {
		name string
		wav  []byte
		want string // error substring; empty for success
	}{
		{"plain", wavBytes(tinyPCM, nil), ""},
		{"unknown chunk", wavBytes(tinyPCM, list), ""},
		{"not riff", []byte("RIFX\x00\x00\x00\x00WAVE"), "not a RIFF/WAVE"},
		{"float format", wavWith(3, 1, SampleRate, 16, tinyPCM, nil), "format tag 3"},
		{"stereo", wavWith(1, 2, SampleRate, 16, tinyPCM, nil), "2 channels"},
		{"rate", wavWith(1, 1, 44100, 16, tinyPCM, nil), "44100 Hz"},
		{"bits", wavWith(1, 1, SampleRate, 24, tinyPCM, nil), "24 bits"},
		{"truncated", wavBytes(tinyPCM, nil)[:50], "exceeds"},
		{"no data", wavBytes(tinyPCM, nil)[:36], `missing "data"`},
		{"no fmt", []byte("RIFF\x0c\x00\x00\x00WAVEdata\x00\x00\x00\x00"), `missing "fmt "`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := decodeWAV(tc.wav)
			if tc.want != "" {
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("error = %v, want it to contain %q", err, tc.want)
				}

				return
			}

			if err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(got, want) {
				t.Errorf("samples = %v, want %v", got, want)
			}
		})
	}
}

func TestDecodeStrictTrailingData(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		raw  string
		ok   bool
	}{
		{"single object", `{"a":1}`, true},
		{"trailing whitespace", "{\"a\":1}\n \t", true},
		{"trailing bracket", `{"a":1}]`, false},
		{"trailing brace", `{"a":1}}`, false},
		{"second value", `{"a":1}{"a":2}`, false},
		{"trailing garbage", `{"a":1}x`, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var v struct{ A int }

			err := decodeStrict([]byte(tc.raw), &v)
			if (err == nil) != tc.ok {
				t.Fatalf("decodeStrict(%q) error = %v, want ok = %v", tc.raw, err, tc.ok)
			}
		})
	}
}
