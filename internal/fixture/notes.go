package fixture

import "fmt"

// NoteSet is the notes JSON of one variant: the upstream note-creation results
// for one option set.
type NoteSet struct {
	Variant string      `json:"variant"`
	Options NoteOptions `json:"options"`
	// Notes is the raw output_to_notes_polyphonic result in emission order.
	Notes []Note `json:"notes"`
	// NoteEvents are the seconds-based events, parallel to Notes. Their
	// PitchBends hold one bend per frame of the note.
	NoteEvents             []NoteEvent `json:"note_events"`
	MIDI                   MIDIData    `json:"midi"`
	MIDIMultiplePitchBends MIDIData    `json:"midi_multiple_pitch_bends"`
}

// NoteOptions are the note-creation options a variant was produced with.
type NoteOptions struct {
	OnsetThreshold      float64 `json:"onset_threshold"`
	FrameThreshold      float64 `json:"frame_threshold"`
	MinimumNoteLengthMS float64 `json:"minimum_note_length_ms"`
	MinNoteLenFrames    int     `json:"min_note_len_frames"`
	InferOnsets         bool    `json:"infer_onsets"`
	MelodiaTrick        bool    `json:"melodia_trick"`
	// MinimumFrequency and MaximumFrequency are nil when upstream got None.
	MinimumFrequency   *float64 `json:"minimum_frequency"`
	MaximumFrequency   *float64 `json:"maximum_frequency"`
	IncludePitchBends  bool     `json:"include_pitch_bends"`
	MultiplePitchBends bool     `json:"multiple_pitch_bends"`
	MIDITempo          float64  `json:"midi_tempo"`
	EnergyTol          int      `json:"energy_tol"`
}

// Note is one frame-based note. Amplitude is a float32 value stored exactly
// as float64.
type Note struct {
	StartFrame int     `json:"start_frame"`
	EndFrame   int     `json:"end_frame"`
	Pitch      int     `json:"pitch"`
	Amplitude  float64 `json:"amplitude"`
}

// NoteEvent is one seconds-based note event. PitchBends is nil when upstream
// had None.
type NoteEvent struct {
	StartS     float64 `json:"start_s"`
	EndS       float64 `json:"end_s"`
	Pitch      int     `json:"pitch"`
	Amplitude  float64 `json:"amplitude"`
	PitchBends []int   `json:"pitch_bends"`
}

// MIDIData is the pretty_midi object upstream would write, as JSON.
type MIDIData struct {
	Instruments []Instrument `json:"instruments"`
}

// Instrument is one pretty_midi instrument.
type Instrument struct {
	Program    int         `json:"program"`
	IsDrum     bool        `json:"is_drum"`
	Notes      []MIDINote  `json:"notes"`
	PitchBends []PitchBend `json:"pitch_bends"`
}

// MIDINote is one pretty_midi note.
type MIDINote struct {
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Pitch    int     `json:"pitch"`
	Velocity int     `json:"velocity"`
}

// PitchBend is one pretty_midi pitch bend; Pitch is in [-8192, 8191].
type PitchBend struct {
	Time  float64 `json:"time"`
	Pitch int     `json:"pitch"`
}

// Notes reads the notes JSON of variant. Unknown fields are an error.
func (c *Clip) Notes(variant string) (*NoteSet, error) {
	v, err := c.variant(variant)
	if err != nil {
		return nil, err
	}

	raw, err := c.readFile(v.Notes)
	if err != nil {
		return nil, err
	}

	var ns NoteSet

	err = decodeStrict(raw, &ns)
	if err != nil {
		return nil, fmt.Errorf("fixture: decode %s: %w", v.Notes, err)
	}

	return &ns, nil
}

// CSV returns the CSV bytes upstream wrote for variant.
func (c *Clip) CSV(variant string) ([]byte, error) {
	v, err := c.variant(variant)
	if err != nil {
		return nil, err
	}

	return c.readFile(v.CSV)
}

// MIDI returns the Standard MIDI File bytes upstream wrote for variant, with
// multiple_pitch_bends=True when multiple is set.
func (c *Clip) MIDI(variant string, multiple bool) ([]byte, error) {
	v, err := c.variant(variant)
	if err != nil {
		return nil, err
	}

	if multiple {
		return c.readFile(v.MIDIMultiplePitchBends)
	}

	return c.readFile(v.MIDI)
}

// WAV returns the bytes of the clip's input WAV file.
func (c *Clip) WAV() ([]byte, error) {
	return c.readFile(c.Manifest.WAV)
}

func (c *Clip) variant(name string) (Variant, error) {
	v, ok := c.Manifest.Variants[name]
	if !ok {
		return Variant{}, fmt.Errorf("fixture: unknown variant %q in clip %q", name, c.Manifest.Clip)
	}

	return v, nil
}
