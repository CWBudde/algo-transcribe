# basic-pitch port specification

This document pins what the Go port of
[spotify/basic-pitch](https://github.com/spotify/basic-pitch) has to reproduce.
The reference is **basic-pitch v0.4.0** (`pyproject.toml` version `0.4.0`), and
the model is `basic_pitch/saved_models/icassp_2022/nmp.onnx`.

**The upstream source wins.** The facts below were extracted from the source
and from parsing the ONNX file. Before you implement a section, read the cited
upstream file. If it disagrees with this document, follow upstream, fix this
document in the same PR, and say so in the PR body.

Upstream files, all at `https://raw.githubusercontent.com/spotify/basic-pitch/main/`:

- `basic_pitch/constants.py`: sample rate, hop and frame constants
- `basic_pitch/inference.py`: windowing, `unwrap_output`, `predict`
- `basic_pitch/models.py`: network graph, `get_cqt`
- `basic_pitch/nn.py`: `HarmonicStacking`, `FlattenAudioCh`, `FlattenFreqCh`
- `basic_pitch/layers/signal.py`: `NormalizedLog`, `GlobalNormalizedLog`
- `basic_pitch/layers/nnaudio.py`: `CQT2010v2` (ported from nnAudio)
- `basic_pitch/note_creation.py`: note events, pitch bends, MIDI, CSV

## 1. Constants

| Name                         | Value   | Note                                                          |
| ---------------------------- | ------- | ------------------------------------------------------------- |
| `AUDIO_SAMPLE_RATE`          | 22050   | Mono input. Upstream resamples with `librosa.load` (soxr_hq). |
| `FFT_HOP`                    | 256     |                                                               |
| `ANNOTATIONS_FPS`            | 86      | `22050 // 256`. Integer; the true rate is 86.13.              |
| `AUDIO_WINDOW_LENGTH`        | 2 s     |                                                               |
| `ANNOT_N_FRAMES`             | 172     | `86 * 2`                                                      |
| `AUDIO_N_SAMPLES`            | 43844   | `22050*2 - 256`                                               |
| `ANNOTATIONS_BASE_FREQUENCY` | 27.5 Hz | A0                                                            |
| `ANNOTATIONS_N_SEMITONES`    | 88      |                                                               |
| `NOTES_BINS_PER_SEMITONE`    | 1       | 88 note bins                                                  |
| `CONTOURS_BINS_PER_SEMITONE` | 3       | 264 contour bins                                              |
| `MIDI_OFFSET`                | 21      | note bin 0 is MIDI 21                                         |
| overlapping frames           | 30      | so `overlap_len = 30*256 = 7680`                              |
| `hop_size`                   | 36164   | `43844 - 7680`                                                |

## 2. Windowing and stitching (`inference.py`)

1. Prepend `overlap_len/2 = 3840` zeros to the audio.
2. Cut windows: `for i in range(0, len_padded, 36164)`, take
   `audio[i : i+43844]` and zero-pad the last window at the end.
3. Run each window through the model. Each window yields 172 frames.
4. `unwrap_output`:
   - Drop `n_olap = 15` frames from the start and the end of every window,
     leaving 142 frames.
   - Concatenate the windows.
   - Trim to `int(n_expected_windows * 142)` frames, where
     `n_expected_windows = original_len / 36164` is a float division.
   - The trim length uses the **original** (unpadded) length.

## 3. Front end (in-graph, `models.py` `get_cqt`)

The saved model takes raw audio. The CQT is part of the graph.

### 3.1 CQT: nnAudio `CQT2010v2` as ported in `layers/nnaudio.py`

**Parameters**

- `sr` 22050, `hop_length` 256, `fmin` 27.5.
- `bins_per_octave` 36.
- `n_bins = 3 * min(ceil(12*log2(8)) + 88, floor(12*log2(11025/27.5)))`
  `= 3 * min(124, 103) = 309`.
- `filter_scale` 1, `basis_norm` 1 (L1), periodic Hann window.
- `pad_mode` reflect, output magnitude, `normalization_type` "librosa",
  `match_torch_exactly=True`.

**Derived values**

- `Q = 1/(2^(1/36) - 1) ≈ 51.44`.
- `n_octaves = ceil(309/36) = 9`.
- `fmax_t = 7040 * 2^(20/36) ≈ 10346.6 Hz`, `fmin_t = 7040 * 2^(-15/36) ≈ 5272.9 Hz`.
- The early-downsampling factor evaluates to 1, so early downsampling is a no-op.

**Top-octave kernels.** There are 36 complex kernels at
`f_k = fmin_t * 2^(k/36)`.

- Kernel length is `l = ceil(Q*22050/f_k)`. It is centred in `n_fft = 256` at
  `start = ceil(128 - l/2) - (l % 2)`.
- Each kernel is scipy periodic `hann(l) * exp(i*2*pi*f*n/22050)` for
  `n ∈ [floor(-l/2), l/2)`. It is divided by `l`, then L1-normalised.

**Each octave**

1. Reflect-pad 128 on each side.
2. `real = conv1d(x, real_kernel)` and `imag = -conv1d(x, imag_kernel)`, both
   VALID, with stride equal to the current hop.
3. The hop halves per octave: 256, 128, …, 1.
4. Between octaves:
   - Zero-pad 127 on each side.
   - Convolve with a 256-tap low-pass: `firwin2(256, [0, 0.5/1.001, 0.5*1.001, 1], [1, 1, 0, 0])`
     (Hamming window, default `nfreqs`).
   - Decimate with stride 2.
5. The signal lengths run 43844 → 21922 → 10961 → 5480 → 2740 → 1370 → 685 →
   342 → 171. Every octave yields 172 frames.

**Assembly**

- Concatenate the octaves lowest-first, giving 324 bins, and keep the last 309.
- "librosa" normalization: multiply bin `k` by `sqrt(ceil(Q*22050/f))` with
  `f = 27.5 * 2^(k/36)`.
- Magnitude is `sqrt(re² + im²)`. The output is (172, 309).

**ONNX cross-check.** The ONNX file stores the real and imaginary kernels as
[36,1,1,256], the low-pass as [1,1,1,256] and the sqrt-length factors as
[309,1,1]. Generated kernels must match them.

### 3.2 NormalizedLog (`layers/signal.py`)

- `p = 10*log10(mag² + 1e-10)`.
- Subtract the minimum over the **whole window**, then divide by the maximum of
  the result. Division by zero yields 0 (`div_no_nan`).
- The scaling is per 2 s window, so results depend on window boundaries.

### 3.3 BatchNorm on the CQT (1 channel, inference)

Folded, this is `y = x * 2.480741 + (-0.876918)`. Read the exact values from
the ONNX Mul/Add constants instead of hard-coding them.

### 3.4 Harmonic stacking (`nn.py` `HarmonicStacking`)

- Harmonics are `[0.5, 1, 2, 3, 4, 5, 6, 7]`, and the shifts are
  `round(36*log2(h)) = [-36, 0, 36, 57, 72, 84, 93, 101]` bins.
- A positive shift takes `x[:, s:]` and zero-pads at the top; a negative shift
  zero-pads at the bottom.
- Stack the shifted copies as channels, then truncate frequency to 264.
- The result is (172, 264, 8).

## 4. CNN (`models.py`)

Layout is time = H and freq = W, NHWC. All convolutions use TF `padding="same"`:

- Stride-1 odd kernels pad `(k-1)/2` on each side.
- With stride 3 on frequency (264 → 88), TF "same" pads `total = max((out-1)*s + k - in, 0)`,
  with `pad_before = total/2` (floor) and the remainder after.

| #   | Layer            | Input                                | Kernel (t×f), stride | Channels | Then              | Output                |
| --- | ---------------- | ------------------------------------ | -------------------- | -------- | ----------------- | --------------------- |
| 1   | conv2d_1         | harmonic stack                       | 3×39, 1              | 8→8      | BN (folded), ReLU | (172,264,8)           |
| 2   | contours-reduced | 1                                    | 5×5, 1               | 8→1      | sigmoid           | **contour** (172,264) |
| 3   | conv2d_2         | contour (post-sigmoid)               | 7×7, (1,3)           | 1→32     | ReLU, no BN       | (172,88,32)           |
| 4   | conv2d_3         | 3                                    | 7×3, 1               | 32→1     | sigmoid           | **note** (172,88)     |
| 5   | conv2d_4         | harmonic stack                       | 5×5, (1,3)           | 8→32     | BN (folded), ReLU | (172,88,32)           |
| 6   | concat           | [note (post-sigmoid), 5] on channels |                      | 33       |                   | (172,88,33)           |
| 7   | conv2d_5         | 6                                    | 3×3, 1               | 33→1     | sigmoid           | **onset** (172,88)    |

- `models.py` has a first contour conv (32 filters, 5×5) commented out
  ("unintentionally skipped when training"). It is **not** part of the model.
- There are 16,782 trainable parameters.
- ONNX Conv weights are OIHW: [8,8,3,39], [1,8,5,5], [32,1,7,7], [1,32,7,3],
  [32,8,5,5], [1,33,3,3].
- ONNX input is `serving_default_input_2:0`, shape (batch, 43844, 1).
- ONNX outputs: `StatefulPartitionedCall:0` is contour, `:1` is note and `:2`
  is onset.
- Map initializers to layers through the Conv nodes' inputs, not by name order.

## 5. Note creation (`note_creation.py`)

### Defaults (`inference.predict`)

- `onset_threshold` 0.5, `frame_threshold` 0.3.
- `minimum_note_length` 127.70 ms, converted as `min_note_len = round(ms/1000 * 22050/256) = 11` frames.
- `infer_onsets` True, `melodia_trick` True, `energy_tol` 11.
- `include_pitch_bends` True, `multiple_pitch_bends` False, `midi_tempo` 120.
- `minimum_frequency` and `maximum_frequency` are None.

### `constrain_frequency`

- `idx = round(hz_to_midi(f) - 21)`.
- Zero `[:, :min_idx]` and `[:, max_idx:]` of **both** onsets and frames.
  Upstream does this in place.

### `get_infered_onsets` (`n_diff = 2`)

1. For n = 1, 2: `diff_n = frames - shift_down(frames, n)`, with the shift
   zero-filled.
2. Take the element-wise min over n, clamp at ≥ 0, and zero the first 2 rows.
3. Rescale by `max(onsets) / max(diff)`. NaN/zero behaviour must match NumPy.
4. `onsets = max(onsets, diff)`.

### `output_to_notes_polyphonic`

**Onset pass**

1. Peaks come from `scipy.signal.argrelmax(onsets, axis=0)`: strict `>` against
   both time neighbours, mode "clip". Edge rows are never peaks.
2. Keep peaks with `onset >= onset_thresh`. Collect them row-major (time, then
   freq), then **reverse** the list.
3. For each peak:
   - Skip it if `start >= n_frames - 1`.
   - Walk `i` from `start+1` while `i < n_frames-1 && k < energy_tol`. `k`
     counts consecutive frames with `remaining_energy[i, f] < frame_thresh` and
     resets otherwise.
   - Set `i -= k`.
   - Skip it if `i - start <= min_note_len`.
   - Zero `remaining_energy[start:i, f]`, and also at `f±1` within
     `[0, MAX_FREQ_IDX = 87]`.
   - `amplitude = mean(frames[start:i, f])`, taken from the original frames.
   - Emit `(start, i, f+21, amplitude)`.

**Melodia pass** (when `melodia_trick`)

1. Loop while `max(remaining_energy) > frame_thresh`.
2. Take the argmax over the flattened array (first occurrence wins) and zero it.
3. Forward walk, zeroing `f` and `f±1` as it goes. The end is `i_end = i - 1 - k`.
   This is **not** the `i - k` of the onset pass.
4. Backward walk while `i > 0`. The start is `i_start = i + 1 + k`.
5. Skip the note if `i_end - i_start <= min_note_len`. Otherwise emit it with
   `amplitude = mean(frames[i_start:i_end, f])`.

Notes come out in emission order, unsorted. Preserve that order.

### Pitch bends (`get_pitch_bends`)

- Window: `gaussian(51, std=5)`.
- For each note:
  - `freq_idx = round(36*log2(midi_to_hz(pitch)/27.5))`, the contour bin.
  - Take a window of ±25 bins, clipped to `[0, 264)`, with the Gaussian slice
    offset to match.
  - For each frame: `argmax(contour[t, window] * gaussian_slice) - pb_shift`,
    in units of 1/3 semitone.

### Time conversion (`model_frames_to_time`)

```
t = n*256/22050 - window_offset * floor(n / 172)
window_offset = (256/22050) * (172 - 43844/256) + 0.0018   # MAGIC_ALIGNMENT_OFFSET
```

The floor uses 172 even though stitched windows are 142 frames. Port it
verbatim.

### MIDI (`note_events_to_midi`, pretty_midi)

- `PrettyMIDI(initial_tempo=120)`, instrument program "Electric Piano 1" (4).
- `velocity = round(127 * amplitude)`.
- Without `multiple_pitch_bends`:
  - Use one instrument.
  - `drop_overlapping_pitch_bends` sorts the notes and removes the bends of any
    notes that overlap in time.
- With `multiple_pitch_bends`: one instrument per pitch.
- Bend events go at `linspace(start, end, len(bends))`, with value
  `round(b * 4096 / 3)` clipped to `[-8192, 8191]` (a ±2 semitone range).
- Times are in seconds. The Go writer uses `cwbudde/midi` (PPQ 480) at 120 BPM
  and must reproduce the same **times**, not the same ticks.

### CSV (`note_events_to_csv`)

The columns are `start_time_s, end_time_s, pitch_midi, velocity`, followed by
the pitch bends. Match the upstream header and number formatting.

## 6. Parity policy

**Fixtures.** Golden data comes from `scripts/fixtures` (pinned `uv` project,
basic-pitch 0.4.0, onnxruntime) and is committed under `testdata/`. Go tests
never run Python.

**Tolerances**

| Stage                                           | Tolerance                                         |
| ----------------------------------------------- | ------------------------------------------------- |
| CQT                                             | ≤ 1e-4 relative                                   |
| Harmonic stack                                  | ≤ 1e-4 absolute                                   |
| Model outputs, given the fixture harmonic stack | ≤ 1e-4 absolute                                   |
| Note creation, given the fixture model outputs  | exact (same notes, same order, same bends)        |
| End to end at 22050 Hz input                    | identical pitches, onsets/offsets within ±1 frame |

**Resampling.** Other sample rates go through algo-dsp `resample`, which is not
soxr. They are checked by note-level tolerance only.

## 7. Known upstream quirks to keep

- Normalisation is per window (see 3.2).
- The output length rule `int(len/36164*142)` is the current upstream rule; the
  older rule, used by basic-pitch-ts fixtures, gives 1032 frames instead of 1039
  for 264,821 samples.
- The onset and melodia passes use different end offsets (`i-k` versus `i-1-k`).
- The 172-frame floor in the time conversion (see section 5).
- `constrain_frequency` mutates its inputs in place.
