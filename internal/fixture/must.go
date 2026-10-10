package fixture

import "testing"

// MustOpen opens the clip called name and fails the test on error.
func MustOpen(tb testing.TB, name string) *Clip {
	tb.Helper()

	c, err := Open(name)
	if err != nil {
		tb.Fatal(err)
	}

	return c
}
