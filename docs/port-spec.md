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
   - Trim to `int(floor(original_len * (86 / 22050)))` frames (v0.4.0,
     `ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE` with the integer FPS 86).
   - The trim length uses the **original** (unpadded) length.
   - Upstream `main` changed this after v0.4.0 to `int(original_len / 36164 * 142)`
     (a float division). The pinned reference is v0.4.0, and the fixtures follow
     it: the synth clip (110,250 samples) has 430 frames, not 432, and the vocal
     clip (88,200 samples) 344, not 346. The manifests record both counts.

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
  `n` in `np.r_[-l//2 : l//2]`, that is `n ∈ [floor(-l/2), floor(l/2))`: `l`
  values, starting one sample earlier than symmetric for odd `l`. It is divided
  by `l`, then L1-normalised.

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
[309,1,1]. Generated kernels must match them. `basicpitch/weights.bin` carries
them as `cqt/kernel_real` and `cqt/kernel_imag` [36,256], `cqt/lowpass` [256]
and `cqt/sqrt_lengths` [309].

What the ONNX graph shows (checked when the weights were extracted, PLAN 1.2):

- The kernels are float32 and match the formulas above to 1e-9 absolute, the
  low-pass to 7e-9 and the sqrt-length factors to 6e-8 relative. No constant
  folding has altered them.
- The periodic Hann window is exactly 0 at its first sample. So each kernel
  row is 0 at `start`, and its non-zero support is `[start+1, start+l)`.
- The low-pass taps are exactly symmetric (`h[n] == h[255-n]` bit for bit) and
  sum to 0.99977926, not 1.
- The imaginary kernel stores `+imag(exp(...))`. The graph applies the sign as
  a `Neg` node after each imaginary `Conv`, so the real kernel is identified as
  the one whose `Conv` output is not negated.
- The graph has 32 `Conv` nodes: 18 CQT convolutions (real and imaginary for
  each of the 9 octaves, strides 256 down to 1), 8 low-pass convolutions
  (stride 2) and the 6 model layers of section 4. The CQT and low-pass
  convolutions have all-zero bias initializers.
- The reflect padding of 128 is a `Pad` with `mode=reflect`. Before every
  low-pass the signal is zero-padded by 127 on each side.
- The sqrt-length factors multiply the real and imaginary parts before the
  magnitude (`ReduceSum` of squares, then `Sqrt`). Because the factors are
  positive, this is equivalent to scaling the magnitude.

### 3.2 NormalizedLog (`layers/signal.py`)

- `p = 10*log10(mag² + 1e-10)`. The ONNX graph computes this as
  `10 * (ln(mag² + 1e-10) * 0.43429446)` in float32, and the min and max reduce
  over the time and frequency axes together.
- Subtract the minimum over the **whole window**, then divide by the maximum of
  the result. Division by zero yields 0 (`div_no_nan`).
- The scaling is per 2 s window, so results depend on window boundaries.

### 3.3 BatchNorm on the CQT (1 channel, inference)

Folded, this is `y = x * 2.480741 + (-0.8769183)`. The ONNX graph holds it as
a `Mul` and an `Add` with one-element float32 constants. The embedded weights
carry them as `batch_normalization/scale` and `batch_normalization/offset`;
read them from there instead of hard-coding them.

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
- For 264 → 88 this is `total = k - 3`. The two stride-3 layers therefore pad
  **symmetrically**: 2/2 for conv2d_2 (k=7) and 1/1 for conv2d_4 (k=5). The
  floor rule would only matter for an even kernel (k=4 pads 0/1, k=6 pads
  1/2). `nn.Conv2D` implements the general rule and tests the asymmetric
  cases.

| #   | Layer            | Input                                | Kernel (t×f), stride | Channels | Then              | Output                |
| --- | ---------------- | ------------------------------------ | -------------------- | -------- | ----------------- | --------------------- |
| 1   | conv2d_1         | harmonic stack                       | 3×39, 1              | 8→8      | BN (folded), ReLU | (172,264,8)           |
| 2   | contours-reduced | 1                                    | 5×5, 1               | 8→1      | sigmoid           | **contour** (172,264) |
| 3   | conv2d_2         | contour (post-sigmoid)               | 7×7, (1,3)           | 1→32     | ReLU, no BN       | (172,88,32)           |
| 4   | conv2d_3         | 3                                    | 7×3, 1               | 32→1     | sigmoid           | **note** (172,88)     |
| 5   | conv2d_4         | harmonic stack                       | 5×5, (1,3)           | 8→32     | BN (folded), ReLU | (172,88,32)           |
| 6   | concat           | [note (post-sigmoid), 5] on channels |                      | 33       |                   | (172,88,33)           |
| 7   | conv2d_5         | 6                                    | 3×3, 1               | 33→1     | sigmoid           | **onset** (172,88)    |

- `models.py` has a first contour conv (32 filters, 5×5) that is **not** part
  of the model. In v0.4.0 it is still in the source, but its output is
  overwritten (the next conv reads the harmonic stack, not its result), so it
  never reaches the graph; upstream `main` later commented it out
  ("unintentionally skipped when training").
- There are 16,782 trainable parameters.
- ONNX Conv weights are OIHW: [8,8,3,39], [1,8,5,5], [32,1,7,7], [1,32,7,3],
  [32,8,5,5], [1,33,3,3].
- ONNX input is `serving_default_input_2:0`, shape (batch, 43844, 1).
- ONNX outputs: `StatefulPartitionedCall:0` is contour, `:1` is note and `:2`
  is onset. The graph lists them in the order `:2`, `:1`, `:0`.
- Map initializers to layers through the Conv nodes' inputs, not by name order.

**What the ONNX graph adds** (checked when the weights were extracted, PLAN 1.2):

- The BatchNorms after conv2d_1 and conv2d_4 are already folded into those
  layers' ONNX kernels and biases. tf2onnx emits each layer as a single `Conv`
  followed directly by `Relu`, with no BatchNorm parameters left in the graph.
  Use the ONNX kernel and bias as they are; do not fold again.
- As a result, the six layers hold 16,700 parameters (kernels and biases):
  7,496 + 201 + 1,600 + 673 + 6,432 + 298. The 16,782 Keras trainable
  parameters are these plus the BatchNorm gamma and beta: 2 for the CQT
  BatchNorm, which survives as the scale and offset of section 3.3, and 80 for
  the 8- and 32-channel BatchNorms, which are folded away.
- The explicit `pads` attributes, `[t_begin, f_begin, t_end, f_end]`, are
  conv2d_1 `[1,19,1,19]`, contours-reduced `[2,2,2,2]`, conv2d_2 `[3,2,3,2]`,
  conv2d_3 `[3,1,3,1]`, conv2d_4 `[2,1,2,1]` and conv2d_5 `[1,1,1,1]`. These
  are the TF "same" values above. All dilations are 1 and all groups are 1.
- In the concat that feeds conv2d_5, the note (post-sigmoid) is channel 0 and
  the 32 conv2d_4 channels are 1..32.
- conv2d_1 and conv2d_4 read the same tensor: the harmonic stack, transposed to
  NCHW. conv2d_2 reads the contour graph output (`StatefulPartitionedCall:0`)
  reshaped back to NCHW.
- Initializer names are unreliable. tf2onnx merges the names of the ops that
  share a constant. For example, conv2d_4's bias is named after `re_lu_3`,
  `batch_normalization_3`, conv2d_2 and conv2d_4, and the zero bias of the
  low-pass convolutions is called `model_1/conv2d_5/Conv2D`. The kernels are
  `const_fold_opt__NNN`.
- `basicpitch/weights.bin` stores the kernels in HWIO
  (`hwio[h][w][i][o] = oihw[o][i][h][w]`) as `<layer>/kernel` and the biases as
  `<layer>/bias`.

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
  Upstream does this in place, `max_freq` first.
- The bounds are Python slice bounds: a negative index counts from the end
  and everything is clamped to `[0, 88]`. A `max_freq` below A0 therefore
  zeroes bins counted from the **top** (10 Hz gives index -18 and zeroes the
  top 18 bins).
- `0`, negative, NaN and infinite frequencies raise upstream (`int()` of
  ±inf or NaN). The Go port uses 0 for None and rejects the others.

### `get_infered_onsets` (`n_diff = 2`)

1. For n = 1, 2: `diff_n = frames - shift_down(frames, n)`, with the shift
   zero-filled.
2. Take the element-wise min over n, clamp at ≥ 0, and zero the first 2 rows.
3. Rescale as `max(onsets) * diff / max(diff)`, in that order. NaN/zero
   behaviour must match NumPy.
4. `onsets = max(onsets, diff)` (`np.max` over the stacked pair, so NaN
   propagates).

If `max(diff)` is 0 (silent or constant frames, or fewer than 3 frames), the
rescale is `0/0`. Every inferred onset is then NaN, there are no peaks and the
onset pass emits nothing; the melodia pass still runs on the frames.

The zero rows that upstream prepends are NumPy's default float64, so the
differences, the rescale and the returned onsets are **float64**, although the
model outputs are float32. With `infer_onsets` the peak picking and the
threshold comparison therefore run in float64. `remaining_energy` is a float64
copy of the frames as well, while `amplitude` is the float32 mean of the
float32 frames.

### `output_to_notes_polyphonic`

**Onset pass**

1. Peaks come from `scipy.signal.argrelmax(onsets, axis=0)`: strict `>` against
   both time neighbours, mode "clip". Edge rows are never peaks.
2. Keep peaks with `onset >= onset_thresh`. Collect them row-major (time, then
   freq), then **reverse** the list. Upstream thresholds a matrix that is 0
   everywhere except at the peaks, so with `onset_thresh <= 0` every non-peak
   cell counts as an onset too.
3. For each peak:
   - Skip it if `start >= n_frames - 1`.
   - Walk `i` from `start+1` while `i < n_frames-1 && k < energy_tol`. `k`
     counts consecutive frames with `remaining_energy[i, f] < frame_thresh` and
     resets otherwise.
   - Set `i -= k`.
   - Skip it if `i - start <= min_note_len`.
   - Zero `remaining_energy[start:i, f]`, and also at `f±1` within
     `[0, MAX_FREQ_IDX = 87]`.
   - `amplitude = mean(frames[start:i, f])`, taken from the (constrained)
     float32 frames. `np.mean` of float32 sums with NumPy's float32
     **pairwise** summation (8 interleaved accumulators, blocks of 128) and
     divides by an `np.intp` count, i.e. in float64, rounded back to float32.
     A float64 or a sequential float32 sum changes dozens of fixture
     amplitudes.
   - Emit `(start, i, f+21, amplitude)`.

**Melodia pass** (when `melodia_trick`)

1. Loop while `max(remaining_energy) > frame_thresh`.
2. Take the argmax over the flattened array (first occurrence wins) and zero it.
3. Forward walk, zeroing `f` and `f±1` as it goes. The end is `i_end = i - 1 - k`.
   This is **not** the `i - k` of the onset pass.
4. Backward walk while `i > 0`. The start is `i_start = i + 1 + k`.
5. Skip the note if `i_end - i_start <= min_note_len`. Otherwise emit it with
   `amplitude = mean(frames[i_start:i_end, f])`.

Edge behaviour of the melodia pass:

- `np.max` propagates NaN and `NaN > frame_thresh` is false, so a NaN left in
  `remaining_energy` after the onset pass ends the melodia pass before it
  starts.
- With `frame_thresh < 0` the loop never terminates (a zeroed cell stays
  above the threshold). The Go port returns an error instead.
- With `min_note_len < 0` it emits zero-length notes (`i_start == i_end`)
  with a NaN amplitude (mean of an empty slice). The Go port reproduces this.
- On an empty input upstream raises (`np.max` of an empty array) whenever
  `infer_onsets` or `melodia_trick` is on. The Go port returns no notes.
- The Go port does not rescan the matrix per iteration. It caches each row's
  maximum and first argmax column in a tournament tree, which gives exactly
  NumPy's first occurrence in the flattened array.

Notes come out in emission order, unsorted. Preserve that order.

### Pitch bends (`get_pitch_bends`)

- Window: `gaussian(51, std=5)`. The Go port hard-codes SciPy 1.18.1's
  values: native `math.Exp` matches them, but other platforms are not
  guaranteed to.
- For each note:
  - `freq_idx = round(36*log2(midi_to_hz(pitch)/27.5))`, the contour bin.
    It rounds to `3*(pitch-21)` for every pitch.
  - Take a window of ±25 bins, clipped to `[0, 264)`, with the Gaussian slice
    offset to match. It is clipped below pitch 30 and above pitch 100, and
    `pb_shift = min(freq_idx, 25)`, so a bend of 0 is always the note's own
    bin. For pitches 21..108 both slices have the same length; outside that
    range NumPy would fail to broadcast.
  - For each frame: `argmax(contour[t, window] * gaussian_slice) - pb_shift`,
    in units of 1/3 semitone. The product is float64, and the first maximum
    (or the first NaN) wins.

### Time conversion (`model_frames_to_time`)

```
t = n*256/22050 - window_offset * floor(n / 172)
window_offset = (256/22050) * (172 - 43844/256) + 0.0018   # MAGIC_ALIGNMENT_OFFSET
```

The floor uses 172 even though stitched windows are 142 frames. Port it
verbatim.

`frames_to_time` multiplies in integers and divides in float64. Python rounds
every step of `window_offset` to float64 (`0x1.525d69b7026c5p-7`). In Go, an
untyped constant expression is evaluated exactly and lands one ulp higher, and
on arm64 `a - b*c` may be fused into an FMA. The port computes it in typed
float64 steps and converts every product explicitly.

### MIDI (`note_events_to_midi`, pretty_midi)

- `PrettyMIDI(initial_tempo=120)`, instrument program "Electric Piano 1" (4).
- `velocity = round(127 * amplitude)`. The amplitude is an `np.float32`, so
  under NEP 50 this is a **float32** multiplication, then half-to-even
  rounding. 60 float32 amplitudes in `[0, 1]` round differently in float64
  (for example `127 * 0x1.83060cp-7` is exactly 1.5 in float32, which gives
  2), though no fixture note hits one.
- Without `multiple_pitch_bends`:
  - Use one instrument.
  - `drop_overlapping_pitch_bends` sorts the notes (Python tuple order:
    start, end, pitch, amplitude, then the bends lists; a NaN amplitude is
    neither less nor greater, so such ties keep their order without comparing
    the bends) and removes the bends of any notes that overlap in time. A note that starts exactly where
    another ends does not overlap it.
- With `multiple_pitch_bends`: one instrument per pitch.
- Bend events go at `linspace(start, end, len(bends))`, with value
  `round(b * 4096 / 3)` clipped to `[-8192, 8191]` (a ±2 semitone range).
- Times are in seconds. The Go writer uses `cwbudde/midi` (PPQ 480) at 120 BPM
  and must reproduce the same **times**, not the same ticks.

### CSV (`inference.save_note_events`)

The header is `start_time_s,end_time_s,pitch_midi,velocity,pitch_bend`. Each row
holds the start and end time, the pitch, `round(127 * amplitude)` and then every
pitch bend as its own column. Times use Python's shortest round-trip `repr`
of the float64 value (for example `3.3913972789115645`), written by
`csv.writer` with `\r\n` line ends. Match it byte for byte.

## 6. Parity policy

**Fixtures.** Golden data comes from `scripts/fixtures` (pinned `uv` project,
basic-pitch 0.4.0, onnxruntime) and is committed under `testdata/fixtures/`.
Load it with `internal/fixture`. The format is described in
`testdata/fixtures/README.md`. Go tests never run Python.

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
- The output length rule of v0.4.0 is `floor(len * 86 / 22050)` (see 2). Upstream
  `main` later changed it to `int(len/36164*142)`. For 264,821 samples the v0.4.0
  rule (also used by the basic-pitch-ts fixtures) gives 1032 frames and the
  `main` rule 1039. Port the v0.4.0 rule.
- With `infer_onsets`, the onsets become float64 (see 5).
- The onset and melodia passes use different end offsets (`i-k` versus `i-1-k`).
- The 172-frame floor in the time conversion (see section 5).
- `constrain_frequency` mutates its inputs in place, with Python slice
  semantics for the bounds (see section 5).
- `infer_onsets` turns silent or constant frames into all-NaN onsets (see
  section 5).
- `velocity` is computed in float32 (see section 5).
