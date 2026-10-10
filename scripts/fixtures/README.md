# Golden fixtures generator

`generate.py` produces the reference data in `testdata/fixtures/` that the Go
port is tested against. It runs the **upstream** basic-pitch 0.4.0 Python code
and the shipped ONNX model. Go tests only read the committed output and never
run Python (docs/port-spec.md §6).

## Regenerate

```bash
just fixtures            # = cd scripts/fixtures && uv run generate.py
```

The only requirement is [uv](https://docs.astral.sh/uv/). It installs the
pinned Python (3.12) and the locked packages from `uv.lock` into
`scripts/fixtures/.venv/`. On first run the script downloads the vocal clip
into `scripts/fixtures/.cache/` and checks its SHA-256. Both directories are
git-ignored. A run takes a few seconds and rewrites `testdata/fixtures/`
completely.

The output is deterministic. Running the script twice on the same machine gives
byte-identical files: no random numbers, gzip with `mtime=0`. onnxruntime on a
different CPU may change the last bits of the float outputs. If a regeneration
changes data files, find out why before you commit them.

Regenerate only when the inputs or the format change. Note the reason in the
commit message.

## Pins

`pyproject.toml` pins basic-pitch 0.4.0, onnxruntime, onnx, numpy, scipy,
librosa and pretty_midi exactly, and `uv.lock` pins everything else. Some pins
are deliberate:

- **Python 3.12, scipy 1.18.** scipy 1.15 (the last release for Python 3.10)
  ships macOS wheels whose PROPACK extension no longer loads on current macOS
  (`__DATA/__thread_bss ... offset field is not zero`).
- **No TensorFlow, CoreML or TFLite.** basic-pitch 0.4.0 declares a
  platform-dependent backend: tensorflow on Linux, coremltools on macOS and
  tflite-runtime on old Pythons. The fixtures come from `nmp.onnx` through
  onnxruntime only. `override-dependencies` removes the other backends, so every
  platform runs the same code path. TensorFlow < 2.15.1 has no wheels for this
  Python on macOS anyway.
- **setuptools 80.** resampy 0.4.2, imported by `basic_pitch.note_creation`,
  still imports `pkg_resources`, which setuptools 81 removed.

## What the script does

For each clip (`synth`, `vocal`):

1. Writes the clip as a 16-bit mono 22050 Hz WAV (`audio.wav`), so no
   resampling is involved. It asserts that `librosa.load` decodes the WAV to
   exactly `int16 / 32768`.
   - `synth`: a deterministic 5 s clip (`synth_clip`) with overlapping chords of
     harmonic tones, decaying envelopes, a slow D4→F4 glide over a bass note and
     digital silence. The fourth window is all zeros, which exercises
     NormalizedLog's `div_no_nan`.
   - `vocal`: the first 4 s of `test_data/vocal-da-80bpm.22050.wav` from
     [spotify/basic-pitch-ts](https://github.com/spotify/basic-pitch-ts),
     pinned to a commit and checked by SHA-256.
2. Windows the WAV with upstream `inference.get_audio_input` and checks each
   window against manual padding.
3. Runs every window through upstream's `inference.Model` (onnxruntime, CPU) for
   contour, note and onset. A second session on a copy of the graph with extra
   outputs exposes the CQT magnitude, the NormalizedLog output, the CQT after
   BatchNorm and the harmonic stack. The script asserts the following:
   - the extended graph reproduces the model outputs bit for bit;
   - NumPy's NormalizedLog matches to < 1e-6;
   - NumPy's `x * scale + offset` with the ONNX constants matches BatchNorm
     exactly;
   - NumPy's harmonic stacking matches exactly.
4. Stitches with upstream `inference.unwrap_output` and asserts equality with
   upstream `inference.run_inference` on the WAV.
5. For every option variant (`default`, `no_melodia`, `no_infer_onsets`,
   `freq_range` = 100–1000 Hz, `low_thresholds` = onset 0.3 / frame 0.2), it
   runs `output_to_notes_polyphonic`, `get_pitch_bends`, `model_frames_to_time`
   and `note_events_to_midi`. It asserts that the result equals
   `note_creation.model_output_to_notes` and, for every variant except
   `no_infer_onsets` (which `predict` cannot express), also `inference.predict`
   end to end. It writes the CSV with upstream `save_note_events` and the MIDI
   with pretty_midi.

The output format is described in
[`testdata/fixtures/README.md`](../../testdata/fixtures/README.md). The Go
loader is `internal/fixture`.
