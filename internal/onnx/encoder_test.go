package onnx

import (
	"encoding/binary"
	"math"
)

// pb is a minimal protobuf encoder used to hand-build test messages.
type pb []byte

func (p pb) rawVarint(v uint64) pb {
	return binary.AppendUvarint(p, v)
}

func (p pb) key(num, wt int) pb {
	return p.rawVarint(uint64(num)<<3 | uint64(wt))
}

// varint appends a varint field; negative values become 10-byte varints.
func (p pb) varint(num int, v int64) pb {
	return p.key(num, wireVarint).rawVarint(uint64(v))
}

func (p pb) fixed32(num int, v uint32) pb {
	return binary.LittleEndian.AppendUint32(p.key(num, wireFixed32), v)
}

func (p pb) fixed64(num int, v uint64) pb {
	return binary.LittleEndian.AppendUint64(p.key(num, wireFixed64), v)
}

func (p pb) float(num int, f float32) pb {
	return p.fixed32(num, math.Float32bits(f))
}

func (p pb) double(num int, f float64) pb {
	return p.fixed64(num, math.Float64bits(f))
}

func (p pb) bytes(num int, b []byte) pb {
	p = p.key(num, wireBytes).rawVarint(uint64(len(b)))

	return append(p, b...)
}

func (p pb) str(num int, s string) pb {
	return p.bytes(num, []byte(s))
}

func (p pb) msg(num int, m pb) pb {
	return p.bytes(num, m)
}

func (p pb) packedVarints(num int, vs ...int64) pb {
	var body pb
	for _, v := range vs {
		body = body.rawVarint(uint64(v))
	}

	return p.bytes(num, body)
}

func (p pb) packedFloats(num int, vs ...float32) pb {
	var body []byte
	for _, v := range vs {
		body = binary.LittleEndian.AppendUint32(body, math.Float32bits(v))
	}

	return p.bytes(num, body)
}

func (p pb) packedDoubles(num int, vs ...float64) pb {
	var body []byte
	for _, v := range vs {
		body = binary.LittleEndian.AppendUint64(body, math.Float64bits(v))
	}

	return p.bytes(num, body)
}

// unknown appends one field of each skippable wire type with numbers the
// reader does not know, when enabled.
func (p pb) unknown(enabled bool) pb {
	if !enabled {
		return p
	}

	return p.varint(1000, -1).
		fixed64(1001, 0xdeadbeefcafebabe).
		bytes(1002, []byte("ignored \x00\xff")).
		fixed32(1003, 0x01020304)
}

// rawLE returns float32 values as little-endian bytes.
func rawLE(vs ...float32) []byte {
	var b []byte
	for _, v := range vs {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(v))
	}

	return b
}

// rawLE64 returns int64 values as little-endian bytes.
func rawLE64(vs ...int64) []byte {
	var b []byte
	for _, v := range vs {
		b = binary.LittleEndian.AppendUint64(b, uint64(v))
	}

	return b
}
