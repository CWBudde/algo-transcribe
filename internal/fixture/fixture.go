// Package fixture loads the golden fixtures that scripts/fixtures/generate.py
// produces from the upstream basic-pitch Python reference.
//
// Every clip lives in its own directory under testdata/fixtures/<clip>/ and is
// described by a manifest.json. Arrays are raw little-endian float32 (.f32) or
// float64 (.f64) files in C order without a header, optionally gzip-compressed
// (.gz suffix). Each array is checked against the element count implied by
// its shape and against the SHA-256 of its uncompressed bytes before it is
// returned. Note sets are plain JSON; CSV and MIDI files are returned as the
// raw bytes upstream wrote.
//
// The package is meant for tests. It never runs Python and builds for js/wasm.
package fixture

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// FormatVersion is the manifest format version this package understands.
const FormatVersion = 1

// ManifestFile is the name of the manifest inside every clip directory.
const ManifestFile = "manifest.json"

// Element types of fixture arrays, as written in [Array.DType].
const (
	DTypeFloat32 = "float32"
	DTypeFloat64 = "float64"
)

// Manifest describes one fixture clip. It mirrors manifest.json.
type Manifest struct {
	FormatVersion int                `json:"format_version"`
	Clip          string             `json:"clip"`
	Description   string             `json:"description"`
	Source        json.RawMessage    `json:"source"`
	Upstream      map[string]string  `json:"upstream"`
	Model         Model              `json:"model"`
	Constants     Constants          `json:"constants"`
	NSamples      int                `json:"n_samples"`
	NWindows      int                `json:"n_windows"`
	NFrames       int                `json:"n_frames"`
	WAV           string             `json:"wav"`
	Arrays        map[string]Array   `json:"arrays"`
	Variants      map[string]Variant `json:"variants"`
	Checks        json.RawMessage    `json:"checks"`
}

// Model identifies the ONNX model the fixtures were produced with.
type Model struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
}

// Constants are the upstream basic-pitch constants recorded by the generator.
type Constants struct {
	SampleRate      int `json:"sample_rate"`
	FFTHop          int `json:"fft_hop"`
	WindowSamples   int `json:"window_samples"`
	OverlapSamples  int `json:"overlap_samples"`
	HopSamples      int `json:"hop_samples"`
	FramesPerWindow int `json:"frames_per_window"`
	OverlapFrames   int `json:"overlap_frames"`
}

// Array describes one array file of a clip.
type Array struct {
	File  string `json:"file"`
	DType string `json:"dtype"`
	Shape []int  `json:"shape"`
	// Axes names every dimension of Shape, for example "window", "time",
	// "harmonic" or "freq".
	Axes   []string `json:"axes"`
	SHA256 string   `json:"sha256"`
}

// Variant names the output files of one note-creation option set.
type Variant struct {
	Options                json.RawMessage `json:"options"`
	Notes                  string          `json:"notes"`
	CSV                    string          `json:"csv"`
	MIDI                   string          `json:"midi"`
	MIDIMultiplePitchBends string          `json:"midi_multiple_pitch_bends"`
}

// Clip is a loaded, validated fixture manifest. Array and note data are read
// lazily from Dir.
type Clip struct {
	Dir      string
	Manifest Manifest
}

// Root returns the fixture root <module>/testdata/fixtures. It walks up from
// the working directory to the first directory that contains go.mod, so it
// works from the test binary of any package in the module, natively and under
// js/wasm in Node.
func Root() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("fixture: working directory: %w", err)
	}

	start := dir

	for {
		_, statErr := os.Stat(filepath.Join(dir, "go.mod"))
		if statErr == nil {
			return filepath.Join(dir, "testdata", "fixtures"), nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("fixture: no go.mod above %s: %w", start, os.ErrNotExist)
		}

		dir = parent
	}
}

// Names returns the sorted names of the clip directories under [Root] that
// contain a manifest.
func Names() ([]string, error) {
	root, err := Root()
	if err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(root)
	if err != nil {
		return nil, fmt.Errorf("fixture: list clips: %w", err)
	}

	var names []string

	for _, e := range entries {
		if !e.IsDir() {
			continue
		}

		_, statErr := os.Stat(filepath.Join(root, e.Name(), ManifestFile))
		if statErr == nil {
			names = append(names, e.Name())
		}
	}

	return names, nil // os.ReadDir sorts by name.
}

// Open loads the clip called name from [Root].
func Open(name string) (*Clip, error) {
	if !filepath.IsLocal(name) {
		return nil, fmt.Errorf("fixture: clip name %q is not a local path", name)
	}

	root, err := Root()
	if err != nil {
		return nil, err
	}

	return Load(filepath.Join(root, name))
}

// Load parses and validates the manifest in dir. Unknown manifest fields are
// an error, so a generator that drifts from this loader is noticed at once.
func Load(dir string) (*Clip, error) {
	raw, err := os.ReadFile(filepath.Join(dir, ManifestFile))
	if err != nil {
		return nil, fmt.Errorf("fixture: read manifest: %w", err)
	}

	var m Manifest

	err = decodeStrict(raw, &m)
	if err != nil {
		return nil, fmt.Errorf("fixture: decode manifest %s: %w", dir, err)
	}

	err = m.validate()
	if err != nil {
		return nil, fmt.Errorf("fixture: manifest %s: %w", dir, err)
	}

	return &Clip{Dir: dir, Manifest: m}, nil
}

func (m *Manifest) validate() error {
	if m.FormatVersion != FormatVersion {
		return fmt.Errorf("format_version %d, want %d", m.FormatVersion, FormatVersion)
	}

	if m.WAV != "" && !filepath.IsLocal(m.WAV) {
		return fmt.Errorf("wav %q is not a local path", m.WAV)
	}

	for name, a := range m.Arrays {
		err := a.validate()
		if err != nil {
			return fmt.Errorf("array %q: %w", name, err)
		}
	}

	for name, v := range m.Variants {
		for _, f := range []string{v.Notes, v.CSV, v.MIDI, v.MIDIMultiplePitchBends} {
			if !filepath.IsLocal(f) {
				return fmt.Errorf("variant %q: file %q is not a local path", name, f)
			}
		}
	}

	return nil
}

func (a *Array) validate() error {
	if !filepath.IsLocal(a.File) {
		return fmt.Errorf("file %q is not a local path", a.File)
	}

	ext := filepath.Ext(strings.TrimSuffix(a.File, ".gz"))

	switch a.DType {
	case DTypeFloat32:
		if ext != ".f32" {
			return fmt.Errorf("dtype %s needs a .f32 file, got %q", a.DType, a.File)
		}
	case DTypeFloat64:
		if ext != ".f64" {
			return fmt.Errorf("dtype %s needs a .f64 file, got %q", a.DType, a.File)
		}
	default:
		return fmt.Errorf("unknown dtype %q", a.DType)
	}

	if len(a.Shape) == 0 {
		return errors.New("empty shape")
	}

	for _, d := range a.Shape {
		if d <= 0 {
			return fmt.Errorf("shape %v has a non-positive dimension", a.Shape)
		}
	}

	if len(a.Axes) != len(a.Shape) {
		return fmt.Errorf("axes %q do not match shape %v", a.Axes, a.Shape)
	}

	sum, err := hex.DecodeString(a.SHA256)
	if err != nil || len(sum) != 32 || strings.ToLower(a.SHA256) != a.SHA256 {
		return fmt.Errorf("sha256 %q is not 64 lowercase hex digits", a.SHA256)
	}

	return nil
}

// readFile reads a file of the clip by its manifest-relative name.
func (c *Clip) readFile(name string) ([]byte, error) {
	if name == "" || !filepath.IsLocal(name) {
		return nil, fmt.Errorf("fixture: file %q is not a local path", name)
	}

	b, err := os.ReadFile(filepath.Join(c.Dir, filepath.FromSlash(name)))
	if err != nil {
		return nil, fmt.Errorf("fixture: read %s: %w", name, err)
	}

	return b, nil
}

func decodeStrict(raw []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()

	err := dec.Decode(v)
	if err != nil {
		return err
	}

	if dec.More() {
		return errors.New("trailing data after JSON value")
	}

	return nil
}
