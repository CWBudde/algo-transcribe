# Golden fixtures

Reference data from upstream basic-pitch 0.4.0 (Python, ONNX model through
onnxruntime). The generator is
[`scripts/fixtures/generate.py`](../../scripts/fixtures/README.md); it rewrites
this directory. Do not edit these files by hand. Load them in Go with
`internal/fixture`.

## Clips

| Clip    | Samples | Windows | Frames | Content                                                                                                     |
| ------- | ------- | ------- | ------ | ----------------------------------------------------------------------------------------------------------- |
| `synth` | 110,250 | 4       | 430    | Deterministic synthetic polyphony (chords, decays, a pitch glide, silence). The fourth window is all zeros. |
| `vocal` | 88,200  | 3       | 344    | First 4 s of `vocal-da-80bpm.22050.wav` from basic-pitch-ts.                                                |

Both clips are mono, 16-bit, 22050 Hz, so no resampling is involved.

### Credit

`vocal/audio.wav` is an excerpt of `test_data/vocal-da-80bpm.22050.wav` from
[spotify/basic-pitch-ts](https://github.com/spotify/basic-pitch-ts) (commit
`2d498f82b61c71898edf0e8dd661b99076676c8b`, SHA-256
`b4d142de94b039cdd7ad90e2fabb3351aa294b82207ddd3b153ad1d3f5e285fa`),
Copyright 2022 Spotify AB, licensed under the Apache License 2.0. All derived
data (model activations, notes, MIDI, CSV) comes from basic-pitch and its model
weights, Copyright 2022 Spotify AB, Apache License 2.0. See `NOTICE`.

## Format

Each clip directory has a `manifest.json` with these fields:

- `upstream`: the package versions that produced the data.
- `model`: the SHA-256 of `nmp.onnx`.
- `constants`: the windowing constants.
- `n_samples`, `n_windows` and `n_frames`.
- `arrays`: one entry per array.
- `variants`: one entry per note-creation option set.
- `checks`: the generator's cross-check results (max abs differences, the
  BatchNorm constants, the indices of all-zero windows, both frame-count rules,
  note counts).

**Arrays** are raw little-endian, row-major files without a header: `.f32` holds
float32 and `.f64` float64. Files over 64 KiB are gzip-compressed (`.gz`). Each
manifest entry gives `file`, `dtype`, `shape`, `axes` and the `sha256` of the
uncompressed bytes.

| Array               | Shape (axes)                                      | What                                                                    |
| ------------------- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| `windows`           | nW × 43844 (window, sample)                       | model input per window, from `get_audio_input`                          |
| `cqt`               | nW × 172 × 309 (window, time, freq)               | CQT magnitude incl. librosa normalisation                               |
| `normalized_log`    | nW × 172 × 309                                    | NormalizedLog output                                                    |
| `cqt_bn`            | nW × 172 × 309                                    | CQT after the BatchNorm                                                 |
| `harmonic_stack`    | nW × 172 × 8 × 264 (window, time, harmonic, freq) | harmonic stack; the model's layout is (time, freq, harmonic), see below |
| `contour`           | nW × 172 × 264                                    | model output per window                                                 |
| `note`              | nW × 172 × 88                                     | model output per window                                                 |
| `onset`             | nW × 172 × 88                                     | model output per window                                                 |
| `unwrapped_contour` | nF × 264 (time, freq)                             | after `unwrap_output`                                                   |
| `unwrapped_note`    | nF × 88                                           | after `unwrap_output`                                                   |
| `unwrapped_onset`   | nF × 88                                           | after `unwrap_output`                                                   |
| `frame_times`       | nF float64 (time)                                 | `model_frames_to_time(nF)`                                              |

The harmonic stack is stored as (time, harmonic, freq) instead of the model's
NHWC (time, freq, harmonic). In that order each harmonic row is a shifted copy
of the BatchNorm row beside it, and gzip shrinks the two files from 3.2 MB to
1.1 MB. `fixture.Clip.HarmonicStackNHWC` returns a window in model layout.

The clip input is `audio.wav`, not a float array. It decodes to the model input
samples exactly as `int16 / 32768`.

**Variants** are `default`, `no_melodia` (`melodia_trick=False`),
`no_infer_onsets` (`infer_onsets=False`), `freq_range`
(`minimum_frequency=100, maximum_frequency=1000`) and `low_thresholds`
(`onset_threshold=0.3, frame_threshold=0.2`). Each variant has four files:

- `notes_<variant>.json` holds these keys:
  - `options`: the options, including `min_note_len_frames`.
  - `notes`: the raw `output_to_notes_polyphonic` events in emission order
    (`start_frame`, `end_frame`, `pitch`, `amplitude`).
  - `note_events`: upstream's seconds-based tuples with the per-frame pitch
    bends from `get_pitch_bends` (`start_s`, `end_s`, `pitch`, `amplitude`,
    `pitch_bends`).
  - `midi`: the pretty_midi instruments, notes and pitch bends of
    `note_events_to_midi`.
  - `midi_multiple_pitch_bends`: the same with `multiple_pitch_bends=True`.

  Amplitudes are float32 values written as exact decimal float64.

- `notes_<variant>.csv`: byte for byte what `inference.save_note_events` writes
  (`\r\n` line ends; `.gitattributes` keeps git from converting them).
- `notes_<variant>.mid`: pretty_midi's MIDI file.
- `notes_<variant>_multi.mid`: pretty_midi's MIDI file with
  `multiple_pitch_bends=True`.

## Size

The directory is about 8.0 MB, within the ~8 MB budget of the plan (first set
at ~5 MB, then revised because that needs dropping parity inputs). The raw
float data for the 7 windows is 19.4 MB. gzip gets the per-window arrays to
65–70 % of their raw size and the unwrapped ones to about 80 %. The reordered
harmonic stack comes down to 10 %. The task lists every array for parity tests,
so all of them are kept.
The two biggest cuts that remain would drop data that later tests compare
against:

- `cqt_bn` and `unwrapped_*` are bit-exact functions of other arrays. Dropping
  them saves about 2 MB.
- A vocal excerpt of ≤ 3.1 s has 2 windows instead of 3, which saves about 1 MB.
