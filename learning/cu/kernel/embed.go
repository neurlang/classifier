package kernel

import _ "embed" // Import the embed package

// Embed the PTX file
//go:embed reduceCUDA.ptx
var PTXreduceCUDA string
