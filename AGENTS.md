# AGENTS.md

Guidance for coding agents (Claude Code and others) working in this repository.

## Project overview

`algo-transcribe` is a pure-Go library for automatic music transcription:
audio in, notes and MIDI out. The first model is a faithful port of Spotify's
[basic-pitch](https://github.com/spotify/basic-pitch) (Apache-2.0). It runs
natively and under `GOOS=js GOARCH=wasm` without any Python, TensorFlow or ONNX
runtime. Its first consumer is the WebAssembly kernel of
[algo-audio-editor](https://github.com/cwbudde/algo-audio-editor).

**Module**: `github.com/cwbudde/algo-transcribe`

The roadmap and its state live in [PLAN.md](PLAN.md). What the port must
reproduce, with every constant and upstream quirk, is in
[docs/port-spec.md](docs/port-spec.md). Read it before touching `basicpitch`.

## Layout (target)

| Path                           | What                                                                                   |
| ------------------------------ | -------------------------------------------------------------------------------------- |
| `basicpitch/`                  | Public API: options, windowing, front end, model graph, note creation, resumable `Job` |
| `basicpitch/weights.bin`       | Embedded model weights extracted from upstream `nmp.onnx` (`go:embed`)                 |
| `nn/`                          | Minimal float32 inference ops (NHWC Conv2D with TF "same" padding, activations)        |
| `smfout/`                      | MIDI (via `github.com/cwbudde/midi`) and CSV writers with basic-pitch semantics        |
| `cmd/algo-transcribe/`         | CLI: WAV in, MIDI/CSV/JSON out                                                         |
| `internal/onnx/`               | Dependency-free ONNX protobuf reader, used only by the weight extractor                |
| `internal/cmd/extractweights/` | Generator for `basicpitch/weights.bin`                                                 |
| `internal/fixture/`            | Loader for the golden fixtures in `testdata/`                                          |
| `scripts/fixtures/`            | Pinned `uv` project that produces golden data from upstream Python                     |
| `docs/benchmarks/`             | Dated benchmark evidence                                                               |

## Commands

```bash
just test         # go test ./...
just lint         # golangci-lint
just fmt          # treefmt: gofumpt, gci, shfmt, prettier
just wasm-check   # GOOS=js GOARCH=wasm build + vet
just test-wasm    # tests under js/wasm in Node
just bench        # benchmarks
just fixtures     # regenerate golden data (needs uv; never in tests)
just ci           # everything CI runs
```

## Rules

1. **Upstream Python is the reference.** Port behaviour verbatim, including
   quirks such as the per-window normalisation, differing end offsets and the
   time-conversion floor. Write each quirk down where it is implemented and in
   `docs/port-spec.md`. If upstream and the spec disagree, upstream wins: fix the
   spec in the same PR.
2. **Parity is tested against committed fixtures.** Go tests never run Python.
   Use the tolerances of `docs/port-spec.md` §6. Do not loosen a tolerance to
   make a test pass; find the divergence.
3. **Everything must build for js/wasm.** No cgo, no `unsafe` tricks that break
   wasm, no OS-specific code outside `cmd/`.
4. **Bounded work per step.** The editor drives transcription from a 2 ms
   budget loop that shares a worker with playback. Long computations (conv
   layers, CQT octaves, windows) must be resumable in small units. Guard the unit
   cost with benchmarks.
5. **Generic DSP goes upstream.** The constant-Q transform, resampling and
   filter design come from `github.com/cwbudde/algo-dsp`. Do not copy DSP into
   this repo. Model-specific code (graph, weights, note creation) stays here.
6. **Zero allocations in hot paths** after setup; check with `-benchmem`.

## Conventions

- Concrete constructors with functional options: `New…(…, opts ...Option) (*T, error)`.
- Errors are wrapped as `fmt.Errorf("pkg: context: %w", err)`.
- Use table-driven tests, golden vectors and runnable `Example`s for every
  exported symbol.
- Files stay under 1500 lines (revive `file-length-limit`).
- Conventional commits; technical writing in English.
- PLAN.md holds open work. When an item is done, tick it and add a one-line note
  (what was covered, numbers, deliberate gaps).
- Benchmark numbers in docs come from real runs, with date, CPU and method.

## Licensing

The repository is Apache-2.0 because it is a derivative of basic-pitch. Keep
`NOTICE` intact and extend it when another upstream's behaviour or data is
reproduced. The ONNX model in `testdata/` and the extracted weights are Spotify's
(Apache-2.0).

## Releasing, and not drifting

This module is part of the `github.com/cwbudde/algo-*` family and follows its
anti-drift rules (the history behind them is in algo-dsp's AGENTS.md):

```bash
just check-deps        # all github.com/cwbudde/* deps at their latest tags?
just check-unreleased  # untagged work piling up on main?
just tag-release vX.Y.Z  # the only way to tag; runs scripts/release-guard.sh
```

- Tag finished work promptly; consumers cannot take untagged work.
- Keep siblings at their latest tags, or write down in PLAN.md why a bump is
  deferred.
- Any incompatible API change needs a **minor** bump, even on v0.x. Describe it
  in CHANGELOG.md as old signature → new signature → call-site rewrite.
- Releases flow up the graph. Never pin pseudo-versions:

```
algo-fft ─┐
          ├─→ algo-dsp ─┐
midi ─────┼─────────────┼─→ algo-transcribe ──→ algo-audio-editor
wav ──────┘             │
```
