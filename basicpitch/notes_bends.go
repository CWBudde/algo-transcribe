package basicpitch

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

// Pitch-bend constants of upstream get_pitch_bends (docs/port-spec.md §5).
const (
	// pbTolerance is n_bins_tolerance: the bend window reaches this many
	// contour bins (1/3 semitone each) below and above the note.
	pbTolerance = 25
	// pbWindowLen is window_length = 2*n_bins_tolerance + 1.
	pbWindowLen = 2*pbTolerance + 1
	// pbMaxMIDI is the highest MIDI pitch of a note bin.
	pbMaxMIDI = noteMIDIOffset + noteMaxFreqIdx
)

// Time-conversion constants of upstream model_frames_to_time and
// constants.py (docs/port-spec.md §5).
const (
	// timeSampleRate is AUDIO_SAMPLE_RATE.
	timeSampleRate = 22050
	// timeFFTHop is FFT_HOP, the hop between model frames in samples.
	timeFFTHop = 256
	// timeAnnotFrames is ANNOT_N_FRAMES = ANNOTATIONS_FPS * 2 = 86 * 2.
	timeAnnotFrames = 172
	// timeAudioNSamples is AUDIO_N_SAMPLES = 22050*2 - FFT_HOP.
	timeAudioNSamples = 2*timeSampleRate - timeFFTHop
	// timeAlignmentOffset is the "magic number" upstream adds to the
	// window offset so that the frames align.
	timeAlignmentOffset = 0.0018
)

// pbGaussianHalf is the first half, index 0 to 25 (the centre, 1.0), of
// scipy.signal.windows.gaussian(51, std=5) = exp(-n²/50) for n = -25..25,
// as SciPy 1.18.1 computes it. The window is exactly symmetric.
//
// The values are hard-coded rather than computed with math.Exp: Go matches
// SciPy bit for bit natively on arm64, but the pure-Go exp used under
// js/wasm and on other architectures may be off by an ulp, which could move
// an argmax between two near-equal products.
var pbGaussianHalf = [pbTolerance + 1]float64{
	0x1.f42ed3f68e690p-19, 0x1.4d2dcae279bfdp-17, 0x1.aa774310a5b40p-16,
	0x1.063bf418bbb1cp-14, 0x1.35d9cc01eb2d5p-13, 0x1.5fc21041027adp-12,
	0x1.7facdba659766p-11, 0x1.92144ad2e5a28p-10, 0x1.94d817547e93ep-9,
	0x1.87a506e4ea251p-8, 0x1.6c0504695c417p-7, 0x1.45139542cf62fp-6,
	0x1.16eab0131bceep-5, 0x1.cbdb2151527fap-5, 0x1.6c3912ee53ea2p-4,
	0x1.152aaa3bf81ccp-3, 0x1.954be9c3e16c7p-3, 0x1.1cb5cf620ef1fp-2,
	0x1.80518d7d9201ap-2, 0x1.f26f2ef268c13p-2, 0x1.368b2fc6f960ap-1,
	0x1.73c9ce7c81bcep-1, 0x1.aba88982ab390p-1, 0x1.d8a2b4ac44685p-1,
	0x1.f5dc99badec5bp-1, 0x1.0000000000000p+0,
}

// pbGaussian is the full 51-point window, mirrored from pbGaussianHalf.
var pbGaussian = func() (w [pbWindowLen]float64) {
	for k := range w {
		w[k] = pbGaussianHalf[min(k, pbWindowLen-1-k)]
	}

	return w
}()

// timeWindowOffset is upstream's window_offset,
//
//	(FFT_HOP / AUDIO_SAMPLE_RATE) * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / FFT_HOP) + 0.0018,
//
// with every operation rounded to float64 in Python's order. Go evaluates
// constant expressions exactly, so the steps go through typed variables,
// and the product is converted explicitly so that it cannot fuse with the
// addition into an FMA. The result is 0x1.525d69b7026c5p-7.
var timeWindowOffset = func() float64 {
	hop, sr := float64(timeFFTHop), float64(timeSampleRate)
	frames, samples := float64(timeAnnotFrames), float64(timeAudioNSamples)

	hopSeconds := hop / sr
	overhang := frames - samples/hop

	return float64(hopSeconds*overhang) + timeAlignmentOffset
}()

// pitchContourBin is int(np.round(midi_pitch_to_contour_bin(pitch))):
// round(36 * log2(midi_to_hz(pitch) / 27.5)), half to even, with librosa's
// midi_to_hz(p) = 440 * 2**((p-69)/12). It equals 3*(pitch-21) for every
// MIDI pitch; the upstream formula is kept so that the port reads like it.
func pitchContourBin(pitch int) int {
	hz := 440.0 * math.Pow(2, (float64(pitch)-69)/12)

	return int(math.RoundToEven(12.0 * 3 * math.Log2(hz/27.5)))
}

// pitchBendWindow returns the bounds get_pitch_bends uses for a note at
// pitch: the contour columns [fs, fe), the Gaussian slice [gs, ge) and
// pb_shift. For the reachable pitches 21..108 (contour bins 0..261) the two
// slices always have the same length fe-fs = ge-gs, so the NumPy product
// never fails to broadcast. The window is clipped at the bottom for pitches
// below 30 (bin < 25) and at the top for pitches above 100 (bin > 238);
// pb_shift is min(bin, 25), so a bend of 0 always means the note's own bin.
func pitchBendWindow(pitch int) (fs, fe, gs, ge, shift int) {
	fi := pitchContourBin(pitch)
	fs = max(fi-pbTolerance, 0)
	fe = min(noteContourBins, fi+pbTolerance+1)
	gs = max(0, pbTolerance-fi)
	ge = pbWindowLen - max(0, fi-(noteContourBins-pbTolerance-1))
	shift = pbTolerance - max(0, pbTolerance-fi)

	return fs, fe, gs, ge, shift
}

// getPitchBends ports get_pitch_bends: it sets the Bends of every note to
// one bend per frame in [Start, End), in 1/3 semitones. contours is the
// row-major (nFrames, 264) contour posteriorgram.
//
// All bends share one backing buffer, sliced with a capped capacity so that
// appending to one note's bends never overwrites the next. A note with
// Start == End gets an empty, non-nil slice, like upstream's empty list.
// Pitches outside 21..108 are an error: upstream's window arithmetic is
// only sound for the 88 note bins the model produces.
func getPitchBends(contours []float32, nFrames int, notes []frameNote) error {
	if nFrames < 0 || len(contours) != nFrames*noteContourBins {
		return fmt.Errorf("basicpitch: pitch bends: contour has %d values, want %d frames of %d",
			len(contours), nFrames, noteContourBins)
	}

	total := 0

	for i, n := range notes {
		if n.Start < 0 || n.End < n.Start || n.End > nFrames {
			return fmt.Errorf("basicpitch: pitch bends: note %d spans frames [%d, %d) outside [0, %d)",
				i, n.Start, n.End, nFrames)
		}

		if n.Pitch < noteMIDIOffset || n.Pitch > pbMaxMIDI {
			return fmt.Errorf("basicpitch: pitch bends: note %d has pitch %d outside [%d, %d]",
				i, n.Pitch, noteMIDIOffset, pbMaxMIDI)
		}

		total += n.End - n.Start
	}

	buf := make([]int, total)

	for i := range notes {
		n := &notes[i]
		k := n.End - n.Start
		n.Bends = buf[:k:k]
		buf = buf[k:]

		noteBends(n.Bends, contours, n.Start, n.Pitch)
	}

	return nil
}

// noteBends writes the bends of one note, starting at frame start, into
// dst, one per element. The caller has checked the frame range and that
// pitch is in 21..108. It does not allocate, so a resumable caller can
// spread the notes of a clip over several steps.
//
// Per frame it takes the argmax of contour[t, fs:fe] * gaussian[gs:ge]
// minus pb_shift. The product of a float32 contour value and the float64
// window is float64, as in NumPy. The argmax follows np.argmax: the first
// maximum wins, and the first NaN wins over everything.
func noteBends(dst []int, contours []float32, start, pitch int) {
	fs, fe, gs, ge, shift := pitchBendWindow(pitch)
	win := pbGaussian[gs:ge]

	for t := range dst {
		row := contours[(start+t)*noteContourBins+fs : (start+t)*noteContourBins+fe]

		best, bestV := 0, float64(row[0])*win[0]
		if !math.IsNaN(bestV) {
			for k := 1; k < len(row); k++ {
				v := float64(row[k]) * win[k]
				if math.IsNaN(v) {
					best = k

					break
				}

				if v > bestV {
					best, bestV = k, v
				}
			}
		}

		dst[t] = best - shift
	}
}

// modelFrameTime is element frame of model_frames_to_time:
//
//	frame*256/22050 - window_offset * floor(frame/172)
//
// librosa's frames_to_time multiplies in integers and divides in float64.
// The floor uses ANNOT_N_FRAMES = 172 although stitched windows contribute
// 142 frames each (docs/port-spec.md §5 and §7); the upstream quirk is kept
// verbatim. The product is converted explicitly so that it cannot fuse
// with the subtraction into an FMA, which NumPy does not do either.
func modelFrameTime(frame int) float64 {
	original := float64(frame*timeFFTHop) / timeSampleRate
	window := math.Floor(float64(frame) / timeAnnotFrames)

	return original - float64(timeWindowOffset*window)
}

// modelFramesToTime ports model_frames_to_time: the start time in seconds
// of each of n model frames. It reproduces upstream bit for bit.
func modelFramesToTime(n int) []float64 {
	times := make([]float64, max(n, 0))
	for i := range times {
		times[i] = modelFrameTime(i)
	}

	return times
}

// frameNotesToEvents converts frame notes to the seconds-based events of
// model_output_to_notes, (times_s[start], times_s[end], pitch, amplitude,
// bends), in the same order. times_s covers nFrames frames, so both frame
// indices must lie in [0, nFrames); upstream would raise IndexError
// otherwise. The events share the notes' Bends slices.
func frameNotesToEvents(notes []frameNote, nFrames int) ([]noteEvent, error) {
	events := make([]noteEvent, len(notes))

	for i, n := range notes {
		if n.Start < 0 || n.Start >= nFrames || n.End < 0 || n.End >= nFrames {
			return nil, fmt.Errorf("basicpitch: note times: note %d spans frames [%d, %d) outside [0, %d)",
				i, n.Start, n.End, nFrames)
		}

		events[i] = noteEvent{
			Start:     modelFrameTime(n.Start),
			End:       modelFrameTime(n.End),
			Pitch:     n.Pitch,
			Amplitude: n.Amplitude,
			Bends:     n.Bends,
		}
	}

	return events, nil
}

// compareNoteEvents orders events like Python compares the tuples
// (start_s, end_s, pitch, amplitude, bends): field by field, and the bends
// lists lexicographically, a shorter prefix first.
//
// Python raises TypeError when it has to compare None with a list; two
// Nones compare equal. Mixed bends cannot occur, since include_pitch_bends
// gives every note a list or every note None. compareNoteEvents treats a
// nil Bends as equal to anything, so such ties keep their input order.
// Times are finite, so NaN ordering does not arise.
func compareNoteEvents(a, b noteEvent) int {
	if c := cmp.Compare(a.Start, b.Start); c != 0 {
		return c
	}

	if c := cmp.Compare(a.End, b.End); c != 0 {
		return c
	}

	if c := cmp.Compare(a.Pitch, b.Pitch); c != 0 {
		return c
	}

	if c := cmp.Compare(a.Amplitude, b.Amplitude); c != 0 {
		return c
	}

	if a.Bends == nil || b.Bends == nil {
		return 0
	}

	return slices.Compare(a.Bends, b.Bends)
}

// dropOverlappingPitchBends ports drop_overlapping_pitch_bends: it returns
// the events sorted like Python's sorted() (stable, see compareNoteEvents)
// with the bends of every two notes that overlap in time removed. Note j
// overlaps an earlier note i when it starts before i ends; a note starting
// exactly at the end of i does not.
//
// events is not modified. Like the tuples upstream builds, the result
// shares the bend slices of events that keep their bends; it only replaces
// Bends with nil and never writes into them.
func dropOverlappingPitchBends(events []noteEvent) []noteEvent {
	out := slices.Clone(events)
	slices.SortStableFunc(out, compareNoteEvents)

	for i := 0; i < len(out)-1; i++ {
		for j := i + 1; j < len(out); j++ {
			if out[j].Start >= out[i].End {
				break
			}

			out[i].Bends = nil
			out[j].Bends = nil
		}
	}

	return out
}
