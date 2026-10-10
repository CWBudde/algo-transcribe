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

## Part of the algo-\* family

- Constant-Q transform, resampling and filter design: [algo-dsp](https://github.com/cwbudde/algo-dsp)
- MIDI file writing: [midi](https://github.com/cwbudde/midi)
- Consumer: [algo-audio-editor](https://github.com/cwbudde/algo-audio-editor)

## License

Apache License 2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). basic-pitch
and its model weights are Copyright 2022 Spotify AB.
