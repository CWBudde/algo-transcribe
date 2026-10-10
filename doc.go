// Package transcribe is the root of algo-transcribe, a pure-Go library for
// automatic music transcription: audio in, notes (and MIDI) out.
//
// The first model is a faithful port of Spotify's basic-pitch (package
// basicpitch). The upstream Python implementation is the reference; see
// docs/port-spec.md for the constants and quirks the port reproduces.
package transcribe
