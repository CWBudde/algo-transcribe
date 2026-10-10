package fixture_test

import (
	"fmt"

	"github.com/cwbudde/algo-transcribe/internal/fixture"
)

func ExampleOpen() {
	clip, err := fixture.Open("synth")
	if err != nil {
		fmt.Println(err)

		return
	}

	_, shape, err := clip.Float32("cqt")
	if err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println("windows:", clip.Manifest.NWindows)
	fmt.Println("cqt shape:", shape)
	// Output:
	// windows: 4
	// cqt shape: [4 172 309]
}

func ExampleClip_HarmonicStackNHWC() {
	clip, err := fixture.Open("vocal")
	if err != nil {
		fmt.Println(err)

		return
	}

	_, shape, err := clip.HarmonicStackNHWC(0)
	if err != nil {
		fmt.Println(err)

		return
	}

	fmt.Println("time, freq, harmonic:", shape)
	// Output:
	// time, freq, harmonic: [172 264 8]
}
