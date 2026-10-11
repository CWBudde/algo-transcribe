# algo-transcribe

Pure-Go automatic music transcription: audio in, notes and MIDI out.

The first model is a faithful port of Spotify's
[basic-pitch](https://github.com/spotify/basic-pitch), a lightweight
polyphonic note transcription network with pitch-bend detection. The port has
no Python, TensorFlow or ONNX runtime dependency and builds for
`GOOS=js GOARCH=wasm`. The model weights (about 145 KB) are embedded.

> **Status:** under construction. See [PLAN.md](PLAN.md) for the roadmap and
> [docs/port-spec.md](docs/port-spec.md) for what the port reproduces.

## Planned usage

```go
res, err := basicpitch.Transcribe(ctx, mono, sampleRate, basicpitch.Options{})
if err != nil {
	log.Fatal(err)
}
for _, n := range res.Notes {
	fmt.Printf("%.3f–%.3f s  MIDI %d  amp %.2f\n", n.Start, n.End, n.Pitch, n.Amplitude)
}
```

```sh
go run github.com/cwbudde/algo-transcribe/cmd/algo-transcribe in.wav -o out.mid
```

## Performance

The CNN runs one 172-frame window (1.64 s of new audio) in these times on an
Apple M5 Pro with Go 1.27.1 (see [docs/benchmarks](docs/benchmarks)):

| Target                       | GEMM               | Time per window | Realtime factor |
| ---------------------------- | ------------------ | --------------: | --------------: |
| darwin/arm64                 | NEON assembly      |         15.3 ms |           107 × |
| darwin/arm64, `-tags purego` | Go tiles           |         38.8 ms |            42 × |
| js/wasm in Node 24           | Go tiles (no SIMD) |        104.6 ms |            16 × |

On arm64 the convolutions use NEON kernels, on amd64 CPUs with AVX2 and FMA
AVX2 kernels; every other target or CPU, and any build with the `purego` tag,
uses the register-blocked Go tiles. The amd64 figures will be added once
measured on a named x86 CPU (see [docs/benchmarks](docs/benchmarks)).

## Part of the algo-\* family

- Constant-Q transform, resampling and filter design: [algo-dsp](https://github.com/cwbudde/algo-dsp)
- MIDI file writing: [midi](https://github.com/cwbudde/midi)
- Consumer: [algo-audio-editor](https://github.com/cwbudde/algo-audio-editor)

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). basic-pitch
and its model weights are Copyright 2022 Spotify AB.
