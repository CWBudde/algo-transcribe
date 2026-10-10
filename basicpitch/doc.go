// Package basicpitch is a pure-Go port of Spotify's basic-pitch (v0.4.0)
// automatic music transcription model.
//
// The package is under construction. So far it embeds the model weights,
// extracted from the upstream nmp.onnx by internal/cmd/extractweights, and
// ports upstream note creation (model outputs to note events). The
// upstream Python implementation is the reference; docs/port-spec.md pins the
// constants and quirks the port reproduces.
package basicpitch

//go:generate go run ../internal/cmd/extractweights -onnx ../testdata/nmp.onnx -out .
