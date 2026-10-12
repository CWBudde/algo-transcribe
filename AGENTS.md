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
| `nn/gemm_{arm64,amd64}.s`      | arm64 NEON and amd64 AVX2+FMA GEMM kernels in front of the Go tiles (see "Assembly")   |
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

## Assembly

`nn`'s GEMM has native SIMD kernels (PLAN Phase 4): arm64 NEON in
`nn/gemm_arm64.s` and amd64 AVX2+FMA in `nn/gemm_amd64.s`, covering the same
shapes. The rules, adapted from algo-fft's "When Working with Assembly":

- **js/wasm must not regress.** Go emits no wasm SIMD, so wasm always runs the
  Go tiles of `nn/gemm.go`. Assembly only ever adds a path in front of them;
  check `just test-wasm` and the wasm benchmarks are unchanged.
- **Always a Go fallback.** Only the `.s` file, its `//go:noescape`
  declarations and the architecture's `gemmSIMD` carry
  `//go:build <arch> && !purego`. A stub `gemmSIMD` that returns false is
  built under `(!arm64 && !amd64) || purego`, and the Go tiles are built on
  every target: they run for every shape a kernel does not cover (and on
  amd64 CPUs without AVX2 or FMA). `-tags purego` runs exactly the wasm code
  natively; CI tests both.
- **Every asm entry point is called by a test**, directly, against the float64
  reference and against the Go tiles. Declared-but-uncalled assembly is
  untested assembly; an optimisation that measures no change at all most
  likely does not run. A test also asserts that the SIMD path is taken for
  every basic-pitch layer shape. On amd64, `TestGemmAVX2Selected` fails
  rather than skips whenever the CPU reports AVX2 and FMA and the AVX2 path
  is not taken, so the ubuntu CI job (`CI=true`) provably runs the kernels;
  only a CPU without the features skips it.
- **Plan 9 operand order**: sources first, destination last.
  `VFMLA V1.S4, V2.S4, V3.S4` is V3 += V2 × V1 (arm64);
  `VFMADD231PS Y8, Y12, Y0` is Y0 += Y12 × Y8 (amd64). Comment every
  instruction whose effect is not obvious from its name.
- **Frames**: entry points are leaf functions marked `NOSPLIT` with a zero
  local frame and the exact argument size (`$0-NN`, a slice is 24 bytes, an
  `int` 8). `go vet` (asmdecl) checks the frames against the declarations; CI
  vets for `GOARCH=arm64` and `GOARCH=amd64`.
- **`//go:noescape`** on every declaration, so the slices passed to assembly
  stay on the stack. The Go wrapper reslices every operand to the exact length
  the kernel touches before the call, so a short slice panics in Go instead of
  corrupting memory in assembly.
- **Missing mnemonics**: Go's arm64 assembler lacks several vector float
  instructions (FADD, FMUL, FADDP, by-element FMLA). Encode them with `WORD`
  macros in the style of algo-fft's `internal/asm/arm64/neon_fp.h`, with the
  ARM ARM encoding in a comment and attribution. A wrong encoding assembles
  happily; only the reference tests catch it.
- **NEON is mandatory on arm64**, so there is no runtime feature detection.
  **AVX2 and FMA are not mandatory on amd64**: `nn/gemm_amd64.go` evaluates
  `cpu.X86.HasAVX2 && cpu.X86.HasFMA` (`golang.org/x/sys/cpu`) once into the
  package variable `haveSIMD`, and `gemmSIMD` returns false without both.
- **amd64 encoding**: use VEX encodings throughout and never mix VEX and
  legacy-SSE instructions in one function (algo-fft measured a 152× slowdown
  for a partially converted loop): `VMOVSS`/`VADDSS`, not `MOVSS`/`ADDSS`.
  Run `VZEROUPPER` before every `RET`.
- **amd64 registers**: leave BP (frame pointer) and R15 (clobbered by global
  accesses under dynamic linking) alone. R14 and X15 may be used in ABI0 code;
  the ABI wrapper restores them for ABIInternal callers.
- **Local amd64 checks on Apple silicon**: `GOARCH=amd64 go vet ./...`,
  `GOARCH=amd64 go test -c -o /dev/null ./nn`, and `just test-amd64`, which
  runs the amd64 test binaries in Docker. Docker Desktop's Rosetta reports
  AVX2 and FMA, so the kernels run, but its timings are no x86 benchmark
  evidence.
- **Determinism, not scalar order**: a kernel must give bit-identical output
  for identical input in every call (`ForwardRows` chunking relies on it), but
  it need not reproduce the summation order of the Go tiles. Results stay
  within the float64-reference tolerance of the GEMM tests and the model's
  §6 parity tolerance.
- **Zero allocations** and no runtime calls from assembly; dispatch per shape
  happens in the Go wrapper. A kernel lands only if it is at least as fast as
  the Go tile on every basic-pitch layer shape it covers; otherwise the wrapper
  returns false for that shape.

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
