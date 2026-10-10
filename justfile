set shell := ["bash", "-uc"]

export GOPRIVATE := "github.com/cwbudde"

# Default recipe - show available commands
default:
    @just --list

# Format all code using treefmt
fmt:
    treefmt --allow-missing-formatter

# Check if code is formatted correctly
check-formatted:
    treefmt --allow-missing-formatter --fail-on-change

# Run linters
lint:
    golangci-lint run --timeout=2m ./...

# Run linters with auto-fix
lint-fix:
    golangci-lint run --fix --timeout=2m ./...

# Ensure go.mod is tidy
check-tidy:
    go mod tidy
    git diff --exit-code go.mod
    if [ -f go.sum ]; then git diff --exit-code go.sum; fi

# Run all tests
test:
    go test ./...

# Run tests with race detector
test-race:
    go test -race ./...

# Run tests with coverage
test-coverage:
    go test -coverprofile=coverage.out ./...
    go tool cover -html=coverage.out -o coverage.html

# Build and vet every package for js/wasm (the editor runs this in its kernel)
wasm-check:
    GOOS=js GOARCH=wasm go build ./...
    GOOS=js GOARCH=wasm go vet ./...

# Run the tests under js/wasm in Node (slow; needs node on PATH)
test-wasm:
    GOOS=js GOARCH=wasm go test -exec="$(go env GOROOT)/lib/wasm/go_js_wasm_exec" ./...

# Run benchmarks
bench:
    go test -run=^$ -bench=. -benchmem ./...

# Regenerate the golden fixtures from the upstream Python reference (needs uv)
fixtures:
    cd scripts/fixtures && uv run generate.py

# Run all checks (formatting, linting, tests, tidiness, wasm, sibling deps)
ci: check-formatted test lint check-tidy wasm-check check-deps

fix:
    just lint-fix
    just fmt

# Clean build artifacts
clean:
    rm -f coverage.out coverage.html

# Are all github.com/cwbudde/* dependencies at their latest tags?
check-deps:
    ./scripts/release-guard.sh deps

# How much work is sitting on main past the latest tag?
check-unreleased:
    ./scripts/release-guard.sh unreleased

# Check every release precondition for VERSION without tagging anything.
release-check VERSION:
    ./scripts/release-guard.sh gate {{VERSION}}

# Tag VERSION: run the full gate, then create and push the annotated tag.
tag-release VERSION:
    ./scripts/release-guard.sh tag {{VERSION}}
