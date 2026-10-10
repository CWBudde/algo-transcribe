"""Generate the golden fixtures for algo-transcribe from upstream basic-pitch.

Run via `just fixtures` (or `uv run generate.py` in this directory). The output
goes to `testdata/fixtures/<clip>/` at the repository root: one `manifest.json`
per clip, little-endian float32/float64 arrays (gzip-compressed when large),
the 16-bit WAV that was fed to upstream, and per option variant the note
events as JSON, plus the CSV and MIDI files upstream writes.

Everything runs through the upstream basic-pitch 0.4.0 code paths:
`get_audio_input` windows the WAV, the shipped `nmp.onnx` runs in onnxruntime
(through upstream's `Model` wrapper), `unwrap_output` stitches the windows, and
`note_creation` produces notes, pitch bends, MIDI and CSV. Intermediate front-end
tensors come from a copy of the ONNX graph with extra graph outputs; the script
asserts that this copy reproduces the unmodified model outputs, and recomputes
NormalizedLog, BatchNorm and harmonic stacking in NumPy as a cross-check.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import logging
import pathlib
import platform
import shutil
import sys
import urllib.request
import warnings
from importlib import metadata

import numpy as np
import scipy.io.wavfile

logging.getLogger().setLevel(logging.ERROR)  # silence basic-pitch's "X is not installed" warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")  # resampy 0.4.2

import basic_pitch  # noqa: E402
import librosa  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
from basic_pitch import inference, note_creation  # noqa: E402
from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP  # noqa: E402
from onnx import numpy_helper  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = REPO / "testdata" / "fixtures"
CACHE = HERE / ".cache"

MODEL_PATH = pathlib.Path(basic_pitch.__file__).parent / "saved_models" / "icassp_2022" / "nmp.onnx"

VOCAL_COMMIT = "2d498f82b61c71898edf0e8dd661b99076676c8b"
VOCAL_PATH = "test_data/vocal-da-80bpm.22050.wav"
VOCAL_URL = f"https://raw.githubusercontent.com/spotify/basic-pitch-ts/{VOCAL_COMMIT}/{VOCAL_PATH}"
VOCAL_SHA256 = "b4d142de94b039cdd7ad90e2fabb3351aa294b82207ddd3b153ad1d3f5e285fa"
VOCAL_SECONDS = 4

# The windowing constants of inference.run_inference.
N_OVERLAPPING_FRAMES = 30
OVERLAP_LEN = N_OVERLAPPING_FRAMES * FFT_HOP
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN
N_FRAMES_PER_WINDOW = 172

# Arrays larger than this many bytes are written gzip-compressed.
GZIP_THRESHOLD = 64 * 1024

# Intermediate tensors of nmp.onnx (tf2onnx names), located by the prefix of
# the node output name. For the BatchNorm (a Mul then an Add) the last match,
# the Add, is the one we want.
INTERMEDIATES = {
    "cqt": ("model_1/cq_t2010v2_1/transpose_70;", (N_FRAMES_PER_WINDOW, 309)),
    "normalized_log": ("model_1/normalized_log_1/Reshape_2;", (N_FRAMES_PER_WINDOW, 309)),
    "cqt_bn": ("model_1/batch_normalization/FusedBatchNormV3;", (N_FRAMES_PER_WINDOW, 309, 1)),
    "harmonic_stack": ("model_1/harmonic-stacking-diff/strided_slice_7;", (N_FRAMES_PER_WINDOW, 264, 8)),
}

HARMONIC_SHIFTS = [int(np.round(36 * np.log2(h))) for h in [0.5, 1, 2, 3, 4, 5, 6, 7]]

DEFAULT_OPTIONS = {
    "onset_threshold": 0.5,
    "frame_threshold": 0.3,
    "minimum_note_length_ms": 127.70,
    "infer_onsets": True,
    "melodia_trick": True,
    "minimum_frequency": None,
    "maximum_frequency": None,
    "include_pitch_bends": True,
    "multiple_pitch_bends": False,
    "midi_tempo": 120,
    "energy_tol": 11,
}

VARIANTS = {
    "default": {},
    "no_melodia": {"melodia_trick": False},
    "no_infer_onsets": {"infer_onsets": False},
    "freq_range": {"minimum_frequency": 100.0, "maximum_frequency": 1000.0},
    "low_thresholds": {"onset_threshold": 0.3, "frame_threshold": 0.2},
}


# --------------------------------------------------------------------------
# Inputs


def synth_clip() -> np.ndarray:
    """A deterministic 5 s polyphonic test signal as int16 samples.

    0.00-0.25 s silence; 0.25-1.75 s C major triad (C4 E4 G4); 1.25-2.75 s
    A minor voicing (A3 C5 E5) overlapping the first chord; 2.75-3.25 s
    digital silence; 3.25-4.65 s a G2 bass under a slow glide from D4 to F4;
    4.65-5.00 s digital silence. Every tone has six harmonics with 1/h
    amplitudes, a 10 ms attack, an exponential decay and a 30 ms release. No
    random numbers are involved, and the result is quantised to the int16 grid
    so the committed WAV decodes to exactly the samples the model saw.

    The fourth model window starts at sample 104652 (4.746 s), after the last
    tone has ended, so it is all zeros and exercises NormalizedLog's
    division by a zero maximum (div_no_nan).
    """
    sr = AUDIO_SAMPLE_RATE
    n = 5 * sr
    t = np.arange(n, dtype=np.float64) / sr
    y = np.zeros(n, dtype=np.float64)

    def tone(midi_from: float, midi_to: float, start: float, end: float, gain: float, tau: float) -> None:
        attack, release = 0.010, 0.030
        local = t - start
        env = np.exp(-np.clip(local, 0, None) / tau)
        env *= np.clip(local / attack, 0, 1) * np.clip((end - t) / release, 0, 1)
        env[(t < start) | (t >= end)] = 0
        # Linear glide in pitch (MIDI units) over the note, integrated to phase.
        frac = np.clip(local / (end - start), 0, 1)
        f0 = 440.0 * 2.0 ** ((midi_from + (midi_to - midi_from) * frac - 69.0) / 12.0)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        for h in range(1, 7):
            y[:] += gain * env * np.sin(h * phase) / h

    for m in (60, 64, 67):
        tone(m, m, 0.25, 1.75, 1.0, 0.9)
    for m in (57, 72, 76):
        tone(m, m, 1.25, 2.75, 0.8, 0.7)
    tone(43, 43, 3.25, 4.65, 0.9, 1.2)
    tone(62, 65, 3.25, 4.65, 1.0, 2.0)

    y *= 0.5 / np.max(np.abs(y))
    return np.round(y * 32767).astype(np.int16)


def vocal_clip() -> tuple[np.ndarray, dict]:
    """The first 4 s of basic-pitch-ts' vocal test file, as int16 samples."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "vocal-da-80bpm.22050.wav"
    if not path.exists() or sha256(path.read_bytes()) != VOCAL_SHA256:
        print(f"downloading {VOCAL_URL}")
        with urllib.request.urlopen(VOCAL_URL) as resp:  # noqa: S310 - pinned https URL
            data = resp.read()
        if sha256(data) != VOCAL_SHA256:
            sys.exit(f"vocal clip: SHA-256 mismatch for {VOCAL_URL}")
        path.write_bytes(data)
    sr, pcm = scipy.io.wavfile.read(path)
    assert sr == AUDIO_SAMPLE_RATE and pcm.dtype == np.int16 and pcm.ndim == 1, (sr, pcm.dtype, pcm.shape)

    # The excerpt must be exactly what librosa makes of the full file.
    full, _ = librosa.load(str(path), sr=AUDIO_SAMPLE_RATE, mono=True)
    n = VOCAL_SECONDS * sr
    assert np.array_equal(full[:n], pcm[:n].astype(np.float32) / 32768), "librosa decode differs from int16/32768"
    source = {
        "repository": "https://github.com/spotify/basic-pitch-ts",
        "commit": VOCAL_COMMIT,
        "path": VOCAL_PATH,
        "sha256": VOCAL_SHA256,
        "license": "Apache-2.0, Copyright 2022 Spotify AB",
        "excerpt": f"samples [0, {n}) of {len(pcm)}",
    }
    return pcm[:n], source


# --------------------------------------------------------------------------
# Helpers


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def to_jsonable(x):
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def json_text(obj, indent: str = "") -> str:
    """JSON with one line per element once a container is too long for a line."""
    flat = json.dumps(obj, separators=(", ", ": "))
    if len(flat) + len(indent) <= 100 or not isinstance(obj, (dict, list)) or not obj:
        return flat
    inner = indent + "  "
    if isinstance(obj, dict):
        items = [f"{inner}{json.dumps(k)}: {json_text(v, inner)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"
    return "[\n" + ",\n".join(inner + json_text(v, inner) for v in obj) + "\n" + indent + "]"


def write_json(path: pathlib.Path, obj) -> None:
    path.write_text(json_text(to_jsonable(obj)) + "\n")


class ArrayWriter:
    """Writes raw little-endian arrays and collects their manifest entries."""

    EXT = {"float32": ".f32", "float64": ".f64"}

    def __init__(self, directory: pathlib.Path):
        self.dir = directory
        self.entries: dict[str, dict] = {}

    def add(self, name: str, arr: np.ndarray, axes: list[str]) -> None:
        dtype = str(arr.dtype)
        assert dtype in self.EXT, (name, dtype)  # never convert silently
        assert len(axes) == arr.ndim, (name, axes, arr.shape)
        raw = np.ascontiguousarray(arr).astype(arr.dtype.newbyteorder("<"), copy=False).tobytes()
        fname = name + self.EXT[dtype]
        payload = raw
        if len(raw) > GZIP_THRESHOLD:
            fname += ".gz"
            payload = gzip.compress(raw, compresslevel=9, mtime=0)
        (self.dir / fname).write_bytes(payload)
        self.entries[name] = {
            "file": fname,
            "dtype": dtype,
            "shape": list(arr.shape),
            "axes": axes,
            "sha256": sha256(raw),
        }


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0


# --------------------------------------------------------------------------
# Model


def intermediate_session() -> tuple[ort.InferenceSession, dict[str, str]]:
    """nmp.onnx with the front-end tensors of INTERMEDIATES as extra outputs."""
    model = onnx.load(str(MODEL_PATH))
    names: dict[str, str] = {}
    for key, (prefix, _) in INTERMEDIATES.items():
        hits = [o for node in model.graph.node for o in node.output if o.startswith(prefix)]
        if not hits:
            sys.exit(f"intermediate {key}: no node output starts with {prefix!r}")
        names[key] = hits[-1]
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(names[key]))
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess, names


def bn_constants() -> tuple[np.float32, np.float32]:
    """The folded CQT BatchNorm scale and offset (Mul and Add initializers)."""
    model = onnx.load(str(MODEL_PATH))
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    prefix = INTERMEDIATES["cqt_bn"][0]
    consts = {}
    for node in model.graph.node:
        if node.output[0].startswith(prefix) and node.op_type in ("Mul", "Add"):
            (c,) = [inits[x] for x in node.input if x in inits]
            consts[node.op_type] = np.float32(c.reshape(()))
    return consts["Mul"], consts["Add"]


def numpy_normalized_log(cqt: np.ndarray) -> np.ndarray:
    """layers/signal.py NormalizedLog, in float32 as the graph computes it."""
    power = cqt * cqt + np.float32(1e-10)
    log_power = np.log(power) * np.float32(0.43429446) * np.float32(10)
    log_power = log_power - np.min(log_power)
    peak = np.max(log_power)
    return np.zeros_like(log_power) if peak == 0 else log_power / peak


def numpy_harmonic_stack(bn: np.ndarray) -> np.ndarray:
    """nn.py HarmonicStacking on a (172, 309) window, truncated to 264 bins."""
    channels = []
    for s in HARMONIC_SHIFTS:
        if s == 0:
            shifted = bn
        elif s > 0:
            shifted = np.pad(bn[:, s:], [(0, 0), (0, s)])
        else:
            shifted = np.pad(bn[:, :s], [(0, 0), (-s, 0)])
        channels.append(shifted)
    return np.stack(channels, axis=-1)[:, :264, :]


# --------------------------------------------------------------------------
# Note creation


def notes_for_variant(unwrapped: dict[str, np.ndarray], opts: dict) -> dict:
    """Run note_creation step by step and record every stage."""
    min_note_len = int(np.round(opts["minimum_note_length_ms"] / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    # output_to_notes_polyphonic (constrain_frequency) mutates its inputs.
    frames = unwrapped["note"].copy()
    onsets = unwrapped["onset"].copy()
    contours = unwrapped["contour"].copy()
    notes = note_creation.output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=opts["onset_threshold"],
        frame_thresh=opts["frame_threshold"],
        min_note_len=min_note_len,
        infer_onsets=opts["infer_onsets"],
        max_freq=opts["maximum_frequency"],
        min_freq=opts["minimum_frequency"],
        melodia_trick=opts["melodia_trick"],
        energy_tol=opts["energy_tol"],
    )
    with_bends = note_creation.get_pitch_bends(contours, notes)
    times = note_creation.model_frames_to_time(contours.shape[0])
    events = [(times[s], times[e], p, a, b) for s, e, p, a, b in with_bends]

    # The same through upstream's one-call entry point must agree exactly.
    midi, events_upstream = note_creation.model_output_to_notes(
        {k: v.copy() for k, v in unwrapped.items()},
        onset_thresh=opts["onset_threshold"],
        frame_thresh=opts["frame_threshold"],
        infer_onsets=opts["infer_onsets"],
        min_note_len=min_note_len,
        min_freq=opts["minimum_frequency"],
        max_freq=opts["maximum_frequency"],
        include_pitch_bends=opts["include_pitch_bends"],
        multiple_pitch_bends=opts["multiple_pitch_bends"],
        melodia_trick=opts["melodia_trick"],
        midi_tempo=opts["midi_tempo"],
    )
    assert repr(to_jsonable(events)) == repr(to_jsonable(events_upstream)), "model_output_to_notes disagrees"
    midi_multi = note_creation.note_events_to_midi(events, multiple_pitch_bends=True, midi_tempo=opts["midi_tempo"])

    return {
        "min_note_len": min_note_len,
        "notes": notes,
        "with_bends": with_bends,
        "events": events,
        "midi": midi,
        "midi_multi": midi_multi,
    }


def midi_to_json(midi) -> dict:
    return {
        "instruments": [
            {
                "program": inst.program,
                "is_drum": inst.is_drum,
                "notes": [
                    {"start": n.start, "end": n.end, "pitch": n.pitch, "velocity": n.velocity} for n in inst.notes
                ],
                "pitch_bends": [{"time": b.time, "pitch": b.pitch} for b in inst.pitch_bends],
            }
            for inst in midi.instruments
        ]
    }


def midi_bytes(midi) -> bytes:
    buf = io.BytesIO()
    midi.write(buf)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Clip generation


def run_windows(model: inference.Model, windows: list[np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    """Model outputs via upstream's Model, front-end tensors via the extended graph."""
    sess, inter_names = intermediate_session()
    output_names = ["StatefulPartitionedCall:0", "StatefulPartitionedCall:1", "StatefulPartitionedCall:2"]
    bn_scale, bn_offset = bn_constants()
    per_window: dict[str, list[np.ndarray]] = {k: [] for k in ["contour", "note", "onset", *INTERMEDIATES]}
    checks = {"model_vs_extended_graph": 0.0, "normalized_log_numpy": 0.0, "cqt_bn_numpy": 0.0}
    hs_exact = True
    for window in windows:
        pred = model.predict(window)
        res = sess.run(output_names + list(inter_names.values()), {"serving_default_input_2:0": window})
        ext = dict(zip(["contour", "note", "onset", *inter_names], res, strict=True))
        for k in ("contour", "note", "onset"):
            checks["model_vs_extended_graph"] = max(checks["model_vs_extended_graph"], max_abs(pred[k], ext[k]))
            per_window[k].append(pred[k][0])
        for k, (_, shape) in INTERMEDIATES.items():
            v = ext[k][0]
            assert v.shape == shape and v.dtype == np.float32, (k, v.shape, v.dtype)
            per_window[k].append(v)
        cqt, nl, bn, hs = (per_window[k][-1] for k in ("cqt", "normalized_log", "cqt_bn", "harmonic_stack"))
        checks["normalized_log_numpy"] = max(checks["normalized_log_numpy"], max_abs(nl, numpy_normalized_log(cqt)))
        checks["cqt_bn_numpy"] = max(checks["cqt_bn_numpy"], max_abs(bn[:, :, 0], nl * bn_scale + bn_offset))
        hs_exact &= bool(np.array_equal(hs, numpy_harmonic_stack(bn[:, :, 0])))
    assert checks["model_vs_extended_graph"] == 0.0, checks
    assert checks["normalized_log_numpy"] < 1e-6 and checks["cqt_bn_numpy"] < 1e-6, checks
    assert hs_exact, "harmonic stack differs from the shifted BatchNorm output"
    checks["harmonic_stack_equals_numpy_exactly"] = hs_exact
    checks["cqt_bn_scale"] = bn_scale
    checks["cqt_bn_offset"] = bn_offset

    stacked = {k: np.stack(v) for k, v in per_window.items()}
    stacked["cqt_bn"] = stacked["cqt_bn"][..., 0]
    return stacked, checks


def generate_clip(name: str, description: str, pcm: np.ndarray, source: dict, model_sha: str) -> None:
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    wav_path = out / "audio.wav"
    scipy.io.wavfile.write(wav_path, AUDIO_SAMPLE_RATE, pcm)

    # Upstream windowing, straight from the WAV file.
    model = inference.Model(MODEL_PATH)
    assert model.model_type == inference.Model.MODEL_TYPES.ONNX, model.model_type
    inputs = list(inference.get_audio_input(wav_path, OVERLAP_LEN, HOP_SIZE))
    windows = [window for window, _, _ in inputs]  # each (1, 43844, 1)
    original_length = inputs[0][2]
    audio, _ = librosa.load(str(wav_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    assert original_length == len(pcm) == len(audio)
    assert np.array_equal(audio, pcm.astype(np.float32) / 32768)
    padded = np.concatenate([np.zeros(OVERLAP_LEN // 2, np.float32), audio])
    for w, window in enumerate(windows):
        expect = np.zeros(AUDIO_N_SAMPLES, np.float32)
        chunk = padded[w * HOP_SIZE : w * HOP_SIZE + AUDIO_N_SAMPLES]
        expect[: len(chunk)] = chunk
        assert np.array_equal(window[0, :, 0], expect), f"window {w} differs from manual padding"

    stacked, checks = run_windows(model, windows)
    checks["silent_windows"] = [w for w, window in enumerate(windows) if not window.any()]
    arrays = ArrayWriter(out)
    # The input samples are not stored separately: audio.wav decodes to them
    # exactly (int16 / 32768, asserted above).
    arrays.add("windows", np.stack([w[0, :, 0] for w in windows]), ["window", "sample"])
    for k in ("cqt", "normalized_log", "cqt_bn"):
        arrays.add(k, stacked[k], ["window", "time", "freq"])
    # The model's harmonic stack is (time, freq, harmonic). It is stored as
    # (time, harmonic, freq): every harmonic row is then a shifted copy of the
    # BatchNorm row next to it, which gzip shrinks from 3.2 MB to 1.1 MB.
    arrays.add(
        "harmonic_stack",
        stacked["harmonic_stack"].transpose(0, 1, 3, 2),
        ["window", "time", "harmonic", "freq"],
    )
    for k in ("contour", "note", "onset"):
        arrays.add(k, stacked[k], ["window", "time", "freq"])

    # Stitching, cross-checked against upstream's run_inference on the WAV.
    unwrapped = {
        k: inference.unwrap_output(stacked[k], original_length, N_OVERLAPPING_FRAMES)
        for k in ("contour", "note", "onset")
    }
    reference = inference.run_inference(wav_path, model)
    for k in unwrapped:
        assert np.array_equal(unwrapped[k], reference[k]), f"run_inference disagrees on {k}"
    n_frames = unwrapped["note"].shape[0]
    # v0.4.0 trims to floor(len * 86 / 22050); upstream main later switched to
    # int(len / 36164 * 142). Record both so the difference stays visible.
    checks["n_frames_v0_4_0_rule"] = int(np.floor(original_length * (86 / AUDIO_SAMPLE_RATE)))
    checks["n_frames_main_rule"] = int(original_length / HOP_SIZE * (N_FRAMES_PER_WINDOW - N_OVERLAPPING_FRAMES))
    assert n_frames == checks["n_frames_v0_4_0_rule"], (n_frames, checks)
    for k in ("contour", "note", "onset"):
        arrays.add("unwrapped_" + k, unwrapped[k], ["time", "freq"])
    arrays.add("frame_times", note_creation.model_frames_to_time(n_frames), ["time"])

    # Note creation per option variant.
    variants = {}
    for vname, overrides in VARIANTS.items():
        opts = {**DEFAULT_OPTIONS, **overrides}
        r = notes_for_variant(unwrapped, opts)
        opts_out = {**opts, "min_note_len_frames": r["min_note_len"]}
        write_json(
            out / f"notes_{vname}.json",
            {
                "variant": vname,
                "options": opts_out,
                "notes": [{"start_frame": s, "end_frame": e, "pitch": p, "amplitude": a} for s, e, p, a in r["notes"]],
                "note_events": [
                    {"start_s": s, "end_s": e, "pitch": p, "amplitude": a, "pitch_bends": b}
                    for s, e, p, a, b in r["events"]
                ],
                "midi": midi_to_json(r["midi"]),
                "midi_multiple_pitch_bends": midi_to_json(r["midi_multi"]),
            },
        )
        inference.save_note_events(r["events"], out / f"notes_{vname}.csv")
        (out / f"notes_{vname}.mid").write_bytes(midi_bytes(r["midi"]))
        (out / f"notes_{vname}_multi.mid").write_bytes(midi_bytes(r["midi_multi"]))
        variants[vname] = {
            "options": opts_out,
            "notes": f"notes_{vname}.json",
            "csv": f"notes_{vname}.csv",
            "midi": f"notes_{vname}.mid",
            "midi_multiple_pitch_bends": f"notes_{vname}_multi.mid",
        }
        checks[f"n_notes_{vname}"] = len(r["notes"])

        # inference.predict has no infer_onsets parameter; every other variant
        # must match it end to end.
        if opts["infer_onsets"]:
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, predicted = inference.predict(
                    wav_path,
                    model,
                    onset_threshold=opts["onset_threshold"],
                    frame_threshold=opts["frame_threshold"],
                    minimum_note_length=opts["minimum_note_length_ms"],
                    minimum_frequency=opts["minimum_frequency"],
                    maximum_frequency=opts["maximum_frequency"],
                    multiple_pitch_bends=opts["multiple_pitch_bends"],
                    melodia_trick=opts["melodia_trick"],
                    midi_tempo=opts["midi_tempo"],
                )
            assert repr(to_jsonable(predicted)) == repr(to_jsonable(r["events"])), f"predict disagrees ({vname})"

    manifest = {
        "format_version": 1,
        "clip": name,
        "description": description,
        "source": source,
        "upstream": {
            "basic_pitch": metadata.version("basic-pitch"),
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "scipy": metadata.version("scipy"),
            "librosa": librosa.__version__,
            "pretty_midi": metadata.version("pretty_midi"),
            "onnx": onnx.__version__,
            "python": platform.python_version(),
        },
        "model": {"path": "basic_pitch/saved_models/icassp_2022/nmp.onnx", "sha256": model_sha},
        "constants": {
            "sample_rate": AUDIO_SAMPLE_RATE,
            "fft_hop": FFT_HOP,
            "window_samples": AUDIO_N_SAMPLES,
            "overlap_samples": OVERLAP_LEN,
            "hop_samples": HOP_SIZE,
            "frames_per_window": N_FRAMES_PER_WINDOW,
            "overlap_frames": N_OVERLAPPING_FRAMES,
        },
        "n_samples": int(original_length),
        "n_windows": len(windows),
        "n_frames": n_frames,
        "wav": wav_path.name,
        "arrays": arrays.entries,
        "variants": variants,
        "checks": checks,
    }
    write_json(out / "manifest.json", manifest)
    size = sum(p.stat().st_size for p in out.iterdir())
    counts = [checks[f"n_notes_{v}"] for v in VARIANTS]
    print(f"{name}: {len(windows)} windows, {n_frames} frames, {size / 1e6:.2f} MB, notes per variant {counts}")


def main() -> None:
    assert metadata.version("basic-pitch") == "0.4.0", metadata.version("basic-pitch")
    model_sha = sha256(MODEL_PATH.read_bytes())
    generate_clip(
        "synth",
        "Deterministic 5 s polyphonic synthetic clip (scripts/fixtures/generate.py, synth_clip).",
        synth_clip(),
        {"generator": "scripts/fixtures/generate.py:synth_clip"},
        model_sha,
    )
    pcm, source = vocal_clip()
    generate_clip("vocal", "First 4 s of basic-pitch-ts test_data/vocal-da-80bpm.22050.wav.", pcm, source, model_sha)
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"testdata/fixtures: {total / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
