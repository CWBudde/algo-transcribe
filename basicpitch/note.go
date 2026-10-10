package basicpitch

// Constants of upstream note_creation.py and constants.py that note creation
// uses (docs/port-spec.md §1 and §5).
const (
	// noteMIDIOffset is MIDI_OFFSET: note bin 0 is MIDI 21 (A0).
	noteMIDIOffset = 21
	// noteMaxFreqIdx is MAX_FREQ_IDX, the highest note bin.
	noteMaxFreqIdx = 87
	// noteBins is the number of note and onset bins per frame.
	noteBins = 88
	// noteContourBins is N_FREQ_BINS_CONTOURS, the contour bins per frame.
	noteContourBins = 264
	// noteEnergyTol is the default energy_tol of output_to_notes_polyphonic.
	noteEnergyTol = 11
)

// frameNote is one note as output_to_notes_polyphonic emits it, in model
// frames, optionally with the pitch bends get_pitch_bends adds.
type frameNote struct {
	// Start and End are frame indices; End is exclusive for the amplitude
	// and the bends.
	Start, End int
	// Pitch is the MIDI pitch, the note bin plus noteMIDIOffset.
	Pitch int
	// Amplitude is NumPy's float32 mean of the frames in [Start, End).
	Amplitude float32
	// Bends holds one bend per frame in [Start, End), in 1/3 semitones.
	// nil stands for upstream's None.
	Bends []int
}

// noteEvent is one note event in seconds, the tuple model_output_to_notes
// returns: (start_s, end_s, pitch, amplitude, bends).
type noteEvent struct {
	Start, End float64
	Pitch      int
	Amplitude  float32
	// Bends is nil for upstream's None.
	Bends []int
}

