package basicpitch

import (
	"errors"
	"fmt"
	"math"
)

// This file ports constrain_frequency, get_infered_onsets and
// output_to_notes_polyphonic of upstream note_creation.py (basic-pitch
// v0.4.0, docs/port-spec.md §5). The output must match the fixtures bit for
// bit, so every step keeps NumPy's evaluation order, dtype and NaN
// semantics; each comment names the upstream expression it reproduces.

// noteParams are the arguments of output_to_notes_polyphonic.
type noteParams struct {
	// onsetThresh is onset_thresh: the minimum peak height of an onset.
	onsetThresh float64
	// frameThresh is frame_thresh: the minimum frame activation that keeps a
	// note sounding.
	frameThresh float64
	// minNoteLen is min_note_len in frames; notes of at most this length are
	// dropped.
	minNoteLen int
	// inferOnsets is infer_onsets: add onsets from rises of the frames.
	inferOnsets bool
	// minFreq and maxFreq are min_freq and max_freq in Hz; 0 stands for
	// upstream's None. Negative, NaN and infinite values are an error, as
	// upstream raises on them (int(round(nan)) is a ValueError and
	// int(round(±inf)) an OverflowError). A literal 0 raises upstream too
	// (log2(0) = -inf); here it means None.
	minFreq, maxFreq float64
	// melodiaTrick is melodia_trick: the second, onset-free pass.
	melodiaTrick bool
	// energyTol is energy_tol: how many frames below frameThresh end a note.
	energyTol int
}

// noteDecoder runs output_to_notes_polyphonic. It keeps its work buffers, so
// decoding again with the same number of frames allocates nothing. The zero
// value is ready to use; a noteDecoder is not safe for concurrent use.
type noteDecoder struct {
	// onsets is the float64 onset matrix (n, noteBins) after constraining
	// and, with inferOnsets, after get_infered_onsets.
	onsets []float64
	// energy is remaining_energy, a float64 copy of the constrained frames.
	energy []float64
	// rowMax and rowArg cache max(energy[t, :]) and the first column that
	// holds it, for the melodia pass.
	rowMax []float64
	rowArg []uint8
	// tree is a tournament tree over rowMax; see melodiaPass.
	tree []int32
}

// decode decodes the note and onset posteriorgrams of nFrames frames into
// notes, appending them to dst[:0] in upstream's emission order. frames and
// onsets are row-major (nFrames, noteBins), the model's unwrapped note and
// onset outputs. Bends of the returned notes are nil.
//
// Like upstream's constrain_frequency, decode zeroes the columns outside
// [p.minFreq, p.maxFreq] of frames and onsets in place, so the caller's
// slices change (docs/port-spec.md §7). The amplitudes are taken from the
// constrained frames, as upstream.
//
// Deliberate deviations from upstream:
//   - nFrames == 0 yields no notes and no error. Upstream raises a
//     ValueError (np.max of an empty array) when infer_onsets or
//     melodia_trick is on.
//   - With melodiaTrick, a negative frameThresh is an error. Upstream never
//     terminates then: a zeroed cell stays above the threshold, so its
//     melodia loop finds a maximum forever (unless a NaN ends it).
func (d *noteDecoder) decode(frames, onsets []float32, nFrames int, p noteParams, dst []frameNote) ([]frameNote, error) {
	dst = dst[:0]

	err := p.validate()
	if err != nil {
		return dst, err
	}

	if nFrames < 0 {
		return dst, fmt.Errorf("basicpitch: notes: negative frame count %d", nFrames)
	}

	n := nFrames * noteBins
	if len(frames) != n || len(onsets) != n {
		return dst, fmt.Errorf("basicpitch: notes: got %d frame and %d onset values, want %d (%d frames x %d bins)",
			len(frames), len(onsets), n, nFrames, noteBins)
	}

	if nFrames == 0 {
		return dst, nil
	}

	constrainFrequency(onsets, frames, p.maxFreq, p.minFreq)

	d.onsets = growNoteBuf(d.onsets, n)
	if p.inferOnsets {
		inferOnsets(d.onsets, onsets, frames, nFrames)
	} else {
		// Upstream keeps float32 onsets here; comparing them as float64 is
		// exact, so one representation serves both cases.
		for i, v := range onsets {
			d.onsets[i] = float64(v)
		}
	}

	// remaining_energy = np.zeros(frames.shape); remaining_energy[:, :] = frames
	d.energy = growNoteBuf(d.energy, n)
	for i, v := range frames {
		d.energy[i] = float64(v)
	}

	dst = d.onsetPass(frames, nFrames, p, dst)

	if p.melodiaTrick {
		dst = d.melodiaPass(frames, nFrames, p, dst)
	}

	return dst, nil
}

func (p *noteParams) validate() error {
	for _, f := range [...]struct {
		name string
		hz   float64
	}{{"minimum", p.minFreq}, {"maximum", p.maxFreq}} {
		if f.hz < 0 || math.IsNaN(f.hz) || math.IsInf(f.hz, 0) {
			return fmt.Errorf("basicpitch: notes: %s frequency %v Hz is neither 0 (none) nor positive and finite", f.name, f.hz)
		}
	}

	if p.melodiaTrick && p.frameThresh < 0 {
		return errors.New("basicpitch: notes: a negative frame threshold never ends the melodia pass")
	}

	return nil
}

// noteFreqIdx is int(np.round(librosa.hz_to_midi(hz) - MIDI_OFFSET)), with
// hz_to_midi(f) = 12*(log2(f) - log2(440)) + 69 and np.round rounding half
// to even. Go's math.Log2 is not guaranteed to match libm's log2 to the last
// bit; that can only matter for a frequency within an ulp of a half-semitone
// boundary.
func noteFreqIdx(hz float64) int {
	// The float64 conversion keeps Go from fusing the product and the
	// addition into an FMA, which NumPy never does.
	midi := float64(12*(math.Log2(hz)-math.Log2(440.0))) + 69

	return int(math.RoundToEven(midi - noteMIDIOffset))
}

// pySliceBound resolves a Python slice bound idx on an axis of length n:
// negative values count from the end, and the result is clamped to [0, n].
func pySliceBound(idx, n int) int {
	if idx < 0 {
		idx += n
	}

	return min(max(idx, 0), n)
}

// constrainFrequency is constrain_frequency: it zeroes the columns
// [:, max_idx:] and then [:, :min_idx] of onsets and frames in place. A
// frequency of 0 stands for None. The bounds follow Python slice semantics,
// so a maximum below A0 gives a negative index that zeroes columns counted
// from the top (max_freq = 10 Hz gives -18 and zeroes the top 18 bins).
func constrainFrequency(onsets, frames []float32, maxFreq, minFreq float64) {
	zero := func(c0, c1 int) {
		if c0 >= c1 {
			return
		}

		for r := 0; r < len(frames); r += noteBins {
			clear(onsets[r+c0 : r+c1])
			clear(frames[r+c0 : r+c1])
		}
	}

	if maxFreq != 0 {
		zero(pySliceBound(noteFreqIdx(maxFreq), noteBins), noteBins)
	}

	if minFreq != 0 {
		zero(0, pySliceBound(noteFreqIdx(minFreq), noteBins))
	}
}

// inferOnsets is get_infered_onsets with n_diff = 2, writing the result into
// dst. Upstream prepends np.zeros (float64) to the float32 frames, so the
// differences, the rescale and the result are float64:
//
//	diff_n[t] = frames[t] - frames[t-n]   (frames[t] - 0 for t < n)
//	frame_diff = max(min(diff_1, diff_2), 0), rows 0 and 1 set to 0
//	frame_diff = max(onsets) * frame_diff / max(frame_diff)
//	result = np.max([onsets, frame_diff], axis=0)
//
// np.min, np.max and np.maximum propagate NaN, and so does every step here.
// When frame_diff is all zero (silent or constant frames) the rescale is
// 0/0 and the whole result is NaN, which leaves no onset peaks at all.
func inferOnsets(dst []float64, onsets, frames []float32, nFrames int) {
	// frame_diff[:n_diff, :] = 0 covers row 0 even when nFrames < 2, and
	// np.max(frame_diff) starts from those zero rows.
	clear(dst[:min(2, nFrames)*noteBins])

	maxDiff := 0.0

	for i := 2 * noteBins; i < len(frames); i++ {
		cur := float64(frames[i])
		d1 := cur - float64(frames[i-noteBins])
		d2 := cur - float64(frames[i-2*noteBins])

		fd := nanMin(d1, d2)
		if fd < 0 { // frame_diff[frame_diff < 0] = 0 leaves NaN alone.
			fd = 0
		}

		dst[i] = fd
		maxDiff = nanMax(maxDiff, fd)
	}

	maxOnset := float64(onsets[0])
	for _, v := range onsets {
		maxOnset = nanMax(maxOnset, float64(v))
	}

	for i, v := range onsets {
		// (a * b) / c element by element, as NumPy evaluates it.
		dst[i] = nanMax(float64(v), maxOnset*dst[i]/maxDiff)
	}
}

// nanMin is np.minimum for two values: NaN if either is NaN.
func nanMin(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}

	return min(a, b)
}

// nanMax is np.maximum for two values: NaN if either is NaN.
func nanMax(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}

	return max(a, b)
}

// onsetPass is the onset loop of output_to_notes_polyphonic.
//
// Upstream builds peak_thresh_mat, a float64 zero matrix holding the onset
// values at the peaks of scipy.signal.argrelmax(onsets, axis=0), collects
// np.where(peak_thresh_mat >= onset_thresh) row-major and walks that list
// backwards. Scanning the cells in reverse row-major order visits the same
// cells in the same order without a list. A peak is strictly greater than
// both time neighbours with mode "clip", so the first and last rows never
// are; NaN is never a peak. Non-peak cells hold 0, so with onset_thresh <= 0
// every one of them is an onset as well, as upstream.
func (d *noteDecoder) onsetPass(frames []float32, nFrames int, p noteParams, dst []frameNote) []frameNote {
	on, re := d.onsets, d.energy

	for t := nFrames - 1; t >= 0; t-- {
		for f := noteBins - 1; f >= 0; f-- {
			v := 0.0

			c := t*noteBins + f
			if t > 0 && t < nFrames-1 && on[c] > on[c-noteBins] && on[c] > on[c+noteBins] {
				v = on[c]
			}

			if !(v >= p.onsetThresh) {
				continue
			}

			// If we're too close to the end of the audio, continue.
			if t >= nFrames-1 {
				continue
			}

			// Find where the energy at this bin stays below frame_thresh
			// for energy_tol frames.
			i, k := t+1, 0
			for i < nFrames-1 && k < p.energyTol {
				if re[i*noteBins+f] < p.frameThresh {
					k++
				} else {
					k = 0
				}

				i++
			}

			i -= k // Go back to the frame above the threshold.

			if i-t <= p.minNoteLen {
				continue
			}

			for r := t; r < i; r++ {
				zeroNeighbourhood(re[r*noteBins:(r+1)*noteBins], f)
			}

			dst = append(dst, frameNote{
				Start:     t,
				End:       i,
				Pitch:     f + noteMIDIOffset,
				Amplitude: columnMean(frames, t, i, f),
			})
		}
	}

	return dst
}

// zeroNeighbourhood zeroes bin f of one remaining_energy row and its
// neighbours f-1 and f+1 inside [0, MAX_FREQ_IDX].
func zeroNeighbourhood(row []float64, f int) {
	row[f] = 0
	if f < noteMaxFreqIdx {
		row[f+1] = 0
	}

	if f > 0 {
		row[f-1] = 0
	}
}

// melodiaPass is the melodia_trick loop of output_to_notes_polyphonic:
//
//	while np.max(remaining_energy) > frame_thresh:
//	    i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), shape)
//
// followed by a forward and a backward walk from i_mid that zero the bin and
// its neighbours as they go.
//
// Scanning the whole matrix for every note is O(nFrames * noteBins) per
// iteration, far too slow for a song. Instead rowMax caches each row's
// maximum and rowArg its first column, and a tournament tree over rowMax
// yields the first row holding the global maximum; only the rows a walk
// touched are recomputed. This is exactly np.argmax's first occurrence over
// the flattened array: in row-major order the first cell holding the global
// maximum M lies in the first row whose maximum is M, at the first column of
// that row holding M. The tree prefers the left child on ties, so its root
// is that row. -0 and +0 compare equal in both orders. Each walk costs
// O(rows touched * log nFrames) for the cache instead of a full scan.
//
// np.max propagates NaN and NaN > frame_thresh is false, so a NaN left in
// remaining_energy when the loop starts ends it at once. NaN cannot appear
// later (the loop only writes zeros), so the cache only holds numbers.
func (d *noteDecoder) melodiaPass(frames []float32, nFrames int, p noteParams, dst []frameNote) []frameNote {
	re := d.energy
	d.rowMax = growNoteBuf(d.rowMax, nFrames)
	d.rowArg = growNoteBuf(d.rowArg, nFrames)

	for t := range nFrames {
		if !d.updateRow(t) {
			return dst
		}
	}

	d.buildTree(nFrames)

	for {
		r := d.tree[1]
		if !(d.rowMax[r] > p.frameThresh) {
			return dst
		}

		iMid, f := int(r), int(d.rowArg[r])
		re[iMid*noteBins+f] = 0

		// Forward pass. Compare first, then zero, then advance, as upstream.
		i, k := iMid+1, 0
		for i < nFrames-1 && k < p.energyTol {
			if re[i*noteBins+f] < p.frameThresh {
				k++
			} else {
				k = 0
			}

			zeroNeighbourhood(re[i*noteBins:(i+1)*noteBins], f)
			i++
		}

		iEnd := i - 1 - k // Go back to the frame above the threshold.
		hi := max(iMid, i-1)

		// Backward pass. It stops before row 0, which it never visits.
		i, k = iMid-1, 0
		for i > 0 && k < p.energyTol {
			if re[i*noteBins+f] < p.frameThresh {
				k++
			} else {
				k = 0
			}

			zeroNeighbourhood(re[i*noteBins:(i+1)*noteBins], f)
			i--
		}

		iStart := i + 1 + k // Go back to the frame above the threshold.
		lo := min(iMid, i+1)

		for t := lo; t <= hi; t++ {
			d.refreshRow(t, f)
		}

		if iEnd-iStart <= p.minNoteLen {
			continue // The note is too short.
		}

		dst = append(dst, frameNote{
			Start:     iStart,
			End:       iEnd,
			Pitch:     f + noteMIDIOffset,
			Amplitude: columnMean(frames, iStart, iEnd, f),
		})
	}
}

// updateRow recomputes rowMax[t] and rowArg[t]. It reports false if the row
// holds a NaN.
func (d *noteDecoder) updateRow(t int) bool {
	row := d.energy[t*noteBins : (t+1)*noteBins]
	best, arg := row[0], 0

	for f, v := range row {
		if math.IsNaN(v) {
			return false
		}

		if v > best {
			best, arg = v, f
		}
	}

	d.rowMax[t], d.rowArg[t] = best, uint8(arg)

	return true
}

// refreshRow updates the cache of row t after a walk zeroed bins f-1, f and
// f+1 of it. If the row maximum is positive and none of those bins held its
// first occurrence, the new zeros lie below the maximum and change neither
// it nor its column, so the row is not rescanned. The tree only depends on
// the row maxima, so it is only fixed when the maximum changed.
func (d *noteDecoder) refreshRow(t, f int) {
	old, arg := d.rowMax[t], int(d.rowArg[t])
	if old > 0 && (arg < f-1 || arg > f+1) {
		return
	}

	d.updateRow(t) // NaN cannot appear during the melodia loop.

	if d.rowMax[t] != old {
		d.fixTree(t)
	}
}

// buildTree builds the tournament tree over rowMax[:nFrames]. Leaves start
// at index size, a power of two; tree[1] is the root (for a single row the
// root is that row's leaf). Every node holds the
// first row with the largest rowMax below it, or -1 for padding.
func (d *noteDecoder) buildTree(nFrames int) {
	size := 1
	for size < nFrames {
		size *= 2
	}

	d.tree = growNoteBuf(d.tree, 2*size)
	for j := range size {
		d.tree[size+j] = -1
		if j < nFrames {
			d.tree[size+j] = int32(j)
		}
	}

	for j := size - 1; j >= 1; j-- {
		d.tree[j] = d.winner(d.tree[2*j], d.tree[2*j+1])
	}
}

// fixTree updates the ancestors of row t after rowMax[t] changed.
func (d *noteDecoder) fixTree(t int) {
	for j := (len(d.tree)/2 + t) / 2; j >= 1; j /= 2 {
		d.tree[j] = d.winner(d.tree[2*j], d.tree[2*j+1])
	}
}

// winner returns the row with the larger rowMax, preferring a on ties.
func (d *noteDecoder) winner(a, b int32) int32 {
	switch {
	case b < 0:
		return a
	case a < 0:
		return b
	case d.rowMax[b] > d.rowMax[a]:
		return b
	default:
		return a
	}
}

// columnMean is np.mean(frames[start:end, f]) for a float32 matrix: NumPy's
// float32 pairwise sum of the column, then float32(sum / np.intp(count)).
// The count is an np.intp, so under NEP 50 the division runs in float64 and
// is rounded back to float32. An empty slice gives NaN (0/0), as upstream.
func columnMean(frames []float32, start, end, f int) float32 {
	n := end - start

	// The reduction starts from the identity 0, which only matters for -0.
	var sum float32
	if n > 0 {
		sum += pairwiseSum32(frames[start*noteBins+f:], n, noteBins)
	}

	return float32(float64(sum) / float64(n))
}

// pairwiseBlock is PW_BLOCKSIZE of NumPy's pairwise summation.
const pairwiseBlock = 128

// pairwiseSum32 is FLOAT_pairwise_sum of numpy/_core/src/umath/
// loops_utils.h.src: the float32 sum of n values of a at the given stride,
// in exactly NumPy's order. Below 8 values it adds them in sequence; up to
// pairwiseBlock it uses 8 interleaved accumulators combined as a tree; above
// it splits at a multiple of 8 near the middle and recurses.
//
// NumPy's reduction hands the whole column to one inner-loop call, since a
// float32 column needs no casting buffer. Columns longer than the iterator's
// buffer (8192 values, 95 s of frames) might be chunked; no fixture note is
// that long, so that case is not reproduced.
func pairwiseSum32(a []float32, n, stride int) float32 {
	if n < 8 {
		var res float32
		for i := range n {
			res += a[i*stride]
		}

		return res
	}

	if n <= pairwiseBlock {
		var r [8]float32
		for j := range r {
			r[j] = a[j*stride]
		}

		i := 8
		for ; i < n-n%8; i += 8 {
			for j := range r {
				r[j] += a[(i+j)*stride]
			}
		}

		res := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			res += a[i*stride]
		}

		return res
	}

	n2 := n / 2
	n2 -= n2 % 8

	return pairwiseSum32(a, n2, stride) + pairwiseSum32(a[n2*stride:], n-n2, stride)
}

// growNoteBuf returns s resized to n elements, reusing its storage when the
// capacity allows. The contents are unspecified.
func growNoteBuf[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}

	return s[:n]
}
