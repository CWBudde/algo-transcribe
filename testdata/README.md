# testdata

Committed reference data. Go tests read these files; they never run Python.

## `nmp.onnx`

The basic-pitch ICASSP 2022 model, exported to ONNX by upstream.

| Field   | Value                                                                                                      |
| ------- | ---------------------------------------------------------------------------------------------------------- |
| Source  | https://raw.githubusercontent.com/spotify/basic-pitch/v0.4.0/basic_pitch/saved_models/icassp_2022/nmp.onnx |
| Tag     | basic-pitch `v0.4.0` (commit `9991303bba609a3b93089d13ec80d1d495083596`)                                   |
| Size    | 230,444 bytes                                                                                              |
| SHA-256 | `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`                                         |
| Graph   | IR 8, opsets `ai.onnx` 15 and `ai.onnx.ml` 2, producer tf2onnx 1.15.1                                      |
| License | Apache-2.0, Copyright 2022 Spotify AB (see `NOTICE`)                                                       |

When it was fetched (2026-10-10), the file on `main` was byte-identical to the
`v0.4.0` one.

`internal/cmd/extractweights` reads this file and writes
`basicpitch/weights.bin` and `basicpitch/weights.go`. It refuses to run if the
SHA-256 differs. To regenerate them, run `go generate ./basicpitch`.
