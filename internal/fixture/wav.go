package fixture

import (
	"encoding/binary"
	"errors"
	"fmt"
)

// SampleRate is the only sample rate of fixture audio (basic-pitch's
// AUDIO_SAMPLE_RATE).
const SampleRate = 22050

// Audio decodes the clip's WAV file to float32 samples as int16/32768, which is
// exactly what librosa.load returns for 16-bit PCM and what the generator feeds
// upstream.
func (c *Clip) Audio() ([]float32, error) {
	b, err := c.WAV()
	if err != nil {
		return nil, err
	}

	samples, err := decodeWAV(b)
	if err != nil {
		return nil, fmt.Errorf("fixture: decode %s: %w", c.Manifest.WAV, err)
	}

	return samples, nil
}

// decodeWAV decodes a RIFF/WAVE file with 16-bit mono PCM at [SampleRate].
// Chunks other than "fmt " and "data" are skipped.
func decodeWAV(b []byte) ([]float32, error) {
	if len(b) < 12 || string(b[0:4]) != "RIFF" || string(b[8:12]) != "WAVE" {
		return nil, errors.New("not a RIFF/WAVE file")
	}

	var (
		haveFmt bool
		data    []byte
	)

	for rest := b[12:]; len(rest) > 0; {
		if len(rest) < 8 {
			return nil, errors.New("truncated chunk header")
		}

		id, size := string(rest[0:4]), binary.LittleEndian.Uint32(rest[4:8])
		rest = rest[8:]

		if uint64(size) > uint64(len(rest)) {
			return nil, fmt.Errorf("chunk %q: size %d exceeds the remaining %d bytes", id, size, len(rest))
		}

		body := rest[:size]

		switch id {
		case "fmt ":
			err := checkFormat(body)
			if err != nil {
				return nil, err
			}

			haveFmt = true
		case "data":
			data = body
		}

		rest = rest[size:]
		if size%2 == 1 && len(rest) > 0 {
			rest = rest[1:] // RIFF chunks are padded to even size.
		}
	}

	if !haveFmt {
		return nil, errors.New(`missing "fmt " chunk`)
	}

	if data == nil {
		return nil, errors.New(`missing "data" chunk`)
	}

	if len(data)%2 != 0 {
		return nil, fmt.Errorf("data chunk has odd length %d", len(data))
	}

	out := make([]float32, len(data)/2)
	for i := range out {
		out[i] = float32(int16(binary.LittleEndian.Uint16(data[2*i:]))) / 32768
	}

	return out, nil
}

func checkFormat(body []byte) error {
	if len(body) < 16 {
		return fmt.Errorf(`"fmt " chunk has %d bytes, want at least 16`, len(body))
	}

	format := binary.LittleEndian.Uint16(body[0:2])
	channels := binary.LittleEndian.Uint16(body[2:4])
	rate := binary.LittleEndian.Uint32(body[4:8])
	bits := binary.LittleEndian.Uint16(body[14:16])

	if format != 1 || channels != 1 || rate != SampleRate || bits != 16 {
		return fmt.Errorf("format tag %d, %d channels, %d Hz, %d bits; want PCM (1), 1 channel, %d Hz, 16 bits",
			format, channels, rate, bits, SampleRate)
	}

	return nil
}
