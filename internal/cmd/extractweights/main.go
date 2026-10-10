// Command extractweights generates basicpitch/weights.bin and
// basicpitch/weights.go from the basic-pitch ONNX model.
//
// It verifies the SHA-256 of the model, maps the six model Conv layers and the
// front-end constants by following the graph's node inputs (never by
// initializer name or order), converts the kernels from ONNX OIHW to HWIO and
// writes them as little-endian float32 together with a Go manifest and an
// embed loader. See docs/port-spec.md sections 3 and 4.
//
// It is run by the go:generate line in package basicpitch:
//
//	go run ../internal/cmd/extractweights -onnx ../testdata/nmp.onnx -out .
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	onnxPath := flag.String("onnx", "", "path to the basic-pitch nmp.onnx model")
	outDir := flag.String("out", "", "directory that receives weights.bin and weights.go")

	flag.Parse()

	if *onnxPath == "" || *outDir == "" || flag.NArg() != 0 {
		flag.Usage()
		os.Exit(2)
	}

	if err := run(*onnxPath, *outDir); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// run reads the model, extracts the weights and writes both output files.
func run(onnxPath, outDir string) error {
	model, err := os.ReadFile(onnxPath)
	if err != nil {
		return fmt.Errorf("extractweights: read model: %w", err)
	}

	bin, src, err := extract(model)
	if err != nil {
		return err
	}

	if err := os.WriteFile(filepath.Join(outDir, binFile), bin, 0o644); err != nil {
		return fmt.Errorf("extractweights: write %s: %w", binFile, err)
	}

	if err := os.WriteFile(filepath.Join(outDir, srcFile), src, 0o644); err != nil {
		return fmt.Errorf("extractweights: write %s: %w", srcFile, err)
	}

	return nil
}
