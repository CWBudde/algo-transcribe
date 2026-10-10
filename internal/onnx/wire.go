package onnx

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// Protobuf wire types.
const (
	wireVarint     = 0
	wireFixed64    = 1
	wireBytes      = 2
	wireStartGroup = 3
	wireEndGroup   = 4
	wireFixed32    = 5
)

// maxFieldNumber is the largest field number protobuf allows (2^29 - 1).
const maxFieldNumber = 1<<29 - 1

// Wire-format errors. They are wrapped with the field path by the message
// parsers and with the "onnx:" prefix by Parse.
var (
	errTruncated      = errors.New("truncated input")
	errVarintOverflow = errors.New("varint overflows 64 bits")
	errGroup          = errors.New("group wire types are not supported")
	errBadWireType    = errors.New("invalid wire type")
	errBadField       = errors.New("invalid field number")
	errBadLength      = errors.New("length exceeds input")
)

// decoder walks one protobuf message. buf holds exactly the bytes of that
// message; pos is the read offset.
type decoder struct {
	buf []byte
	pos int
}

// more reports whether unread bytes remain.
func (d *decoder) more() bool {
	return d.pos < len(d.buf)
}

// varint reads a base-128 varint of at most 10 bytes.
func (d *decoder) varint() (uint64, error) {
	var v uint64

	for i := range 10 {
		if d.pos >= len(d.buf) {
			return 0, errTruncated
		}

		b := d.buf[d.pos]
		d.pos++

		// The tenth byte may only contribute the 64th bit.
		if i == 9 && b > 1 {
			return 0, errVarintOverflow
		}

		v |= uint64(b&0x7f) << (7 * i)
		if b < 0x80 {
			return v, nil
		}
	}

	return 0, errVarintOverflow
}

// tag reads a field key and validates field number and wire type.
func (d *decoder) tag() (num, wt int, err error) {
	k, err := d.varint()
	if err != nil {
		return 0, 0, fmt.Errorf("tag: %w", err)
	}

	n := k >> 3
	if n == 0 || n > maxFieldNumber {
		return 0, 0, fmt.Errorf("tag: %w %d", errBadField, n)
	}

	wt = int(k & 7)
	switch wt {
	case wireVarint, wireFixed64, wireBytes, wireFixed32:
	case wireStartGroup, wireEndGroup:
		return 0, 0, fmt.Errorf("field %d: %w", n, errGroup)
	default:
		return 0, 0, fmt.Errorf("field %d: %w %d", n, errBadWireType, wt)
	}

	return int(n), wt, nil
}

// fixed32 reads four little-endian bytes.
func (d *decoder) fixed32() (uint32, error) {
	if len(d.buf)-d.pos < 4 {
		return 0, errTruncated
	}

	v := binary.LittleEndian.Uint32(d.buf[d.pos:])
	d.pos += 4

	return v, nil
}

// fixed64 reads eight little-endian bytes.
func (d *decoder) fixed64() (uint64, error) {
	if len(d.buf)-d.pos < 8 {
		return 0, errTruncated
	}

	v := binary.LittleEndian.Uint64(d.buf[d.pos:])
	d.pos += 8

	return v, nil
}

// bytes reads a length-delimited payload. The result aliases the input.
func (d *decoder) bytes() ([]byte, error) {
	n, err := d.varint()
	if err != nil {
		return nil, fmt.Errorf("length: %w", err)
	}

	if n > uint64(len(d.buf)-d.pos) {
		return nil, fmt.Errorf("%w: need %d bytes, have %d", errBadLength, n, len(d.buf)-d.pos)
	}

	b := d.buf[d.pos : d.pos+int(n)]
	d.pos += int(n)

	return b, nil
}

// skip discards one field value of wire type wt.
func (d *decoder) skip(wt int) error {
	var err error

	switch wt {
	case wireVarint:
		_, err = d.varint()
	case wireFixed64:
		_, err = d.fixed64()
	case wireBytes:
		_, err = d.bytes()
	case wireFixed32:
		_, err = d.fixed32()
	default:
		err = fmt.Errorf("%w %d", errBadWireType, wt)
	}

	return err
}

// expect returns an error unless the field arrived with wire type want.
func expect(wt, want int) error {
	if wt != want {
		return fmt.Errorf("%w %d, want %d", errBadWireType, wt, want)
	}

	return nil
}

// readInt64 reads a varint field as two's-complement int64.
func (d *decoder) readInt64(wt int) (int64, error) {
	if err := expect(wt, wireVarint); err != nil {
		return 0, err
	}

	v, err := d.varint()

	return int64(v), err
}

// readInt32 reads a varint field as int32. Negative int32 values are encoded
// as 10-byte sign-extended varints; truncation recovers them.
func (d *decoder) readInt32(wt int) (int32, error) {
	v, err := d.readInt64(wt)

	return int32(v), err // protobuf int32 semantics: truncate
}

// readBytes reads a length-delimited field and returns a copy.
func (d *decoder) readBytes(wt int) ([]byte, error) {
	if err := expect(wt, wireBytes); err != nil {
		return nil, err
	}

	b, err := d.bytes()
	if err != nil {
		return nil, err
	}

	return append([]byte{}, b...), nil
}

// readString reads a length-delimited field as a string (always a copy).
func (d *decoder) readString(wt int) (string, error) {
	if err := expect(wt, wireBytes); err != nil {
		return "", err
	}

	b, err := d.bytes()

	return string(b), err
}

// readMessage reads a length-delimited field holding an embedded message and
// returns its bytes (aliasing the input; message parsers copy what they keep).
func (d *decoder) readMessage(wt int) ([]byte, error) {
	if err := expect(wt, wireBytes); err != nil {
		return nil, err
	}

	return d.bytes()
}

// readFloat32 reads a fixed32 field as IEEE-754 float.
func (d *decoder) readFloat32(wt int) (float32, error) {
	if err := expect(wt, wireFixed32); err != nil {
		return 0, err
	}

	v, err := d.fixed32()

	return math.Float32frombits(v), err
}

// appendInt64s appends one or more repeated int64 values, accepting both the
// unpacked (varint) and the packed (length-delimited) encoding.
func (d *decoder) appendInt64s(dst []int64, wt int) ([]int64, error) {
	switch wt {
	case wireVarint:
		v, err := d.varint()
		if err != nil {
			return dst, err
		}

		return append(dst, int64(v)), nil
	case wireBytes:
		b, err := d.bytes()
		if err != nil {
			return dst, err
		}

		p := decoder{buf: b}
		for p.more() {
			v, err := p.varint()
			if err != nil {
				return dst, fmt.Errorf("packed: %w", err)
			}

			dst = append(dst, int64(v))
		}

		return dst, nil
	default:
		return dst, fmt.Errorf("%w %d for repeated varint", errBadWireType, wt)
	}
}

// appendInt32s is appendInt64s for int32 fields (values truncated to 32 bits).
func (d *decoder) appendInt32s(dst []int32, wt int) ([]int32, error) {
	vs, err := d.appendInt64s(nil, wt)
	for _, v := range vs {
		dst = append(dst, int32(v)) // protobuf int32 semantics: truncate
	}

	return dst, err
}

// appendFloat32s appends repeated float values in unpacked (fixed32) or packed
// encoding.
func (d *decoder) appendFloat32s(dst []float32, wt int) ([]float32, error) {
	switch wt {
	case wireFixed32:
		v, err := d.fixed32()
		if err != nil {
			return dst, err
		}

		return append(dst, math.Float32frombits(v)), nil
	case wireBytes:
		b, err := d.bytes()
		if err != nil {
			return dst, err
		}

		if len(b)%4 != 0 {
			return dst, fmt.Errorf("packed float: length %d not a multiple of 4", len(b))
		}

		for i := 0; i < len(b); i += 4 {
			dst = append(dst, math.Float32frombits(binary.LittleEndian.Uint32(b[i:])))
		}

		return dst, nil
	default:
		return dst, fmt.Errorf("%w %d for repeated float", errBadWireType, wt)
	}
}

// appendFloat64s appends repeated double values in unpacked (fixed64) or
// packed encoding.
func (d *decoder) appendFloat64s(dst []float64, wt int) ([]float64, error) {
	switch wt {
	case wireFixed64:
		v, err := d.fixed64()
		if err != nil {
			return dst, err
		}

		return append(dst, math.Float64frombits(v)), nil
	case wireBytes:
		b, err := d.bytes()
		if err != nil {
			return dst, err
		}

		if len(b)%8 != 0 {
			return dst, fmt.Errorf("packed double: length %d not a multiple of 8", len(b))
		}

		for i := 0; i < len(b); i += 8 {
			dst = append(dst, math.Float64frombits(binary.LittleEndian.Uint64(b[i:])))
		}

		return dst, nil
	default:
		return dst, fmt.Errorf("%w %d for repeated double", errBadWireType, wt)
	}
}

// walk iterates over the fields of message b. visit handles a field and
// reports whether it consumed the value; unhandled fields are skipped.
func walk(b []byte, visit func(d *decoder, num, wt int) (bool, error)) error {
	d := &decoder{buf: b}
	for d.more() {
		num, wt, err := d.tag()
		if err != nil {
			return err
		}

		handled, err := visit(d, num, wt)
		if err == nil && !handled {
			err = d.skip(wt)
		}

		if err != nil {
			return fmt.Errorf("field %d: %w", num, err)
		}
	}

	return nil
}
