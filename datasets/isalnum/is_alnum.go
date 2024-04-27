// Package isalnum implements the IsAlnum Dataset
package isalnum

import "github.com/neurlang/classifier/datasets"

var Dataset datasets.Dataset

func isAlphanumeric(c rune) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}

func init() {
	Dataset.Init()

	// Loop through ASCII characters
	for i := uint32(0); i <= 255; i++ {
		// Check if the character is alphanumeric
		isAlnum := isAlphanumeric(rune(i))
		// Add the character and its alnum status to the map
		Dataset[i] = isAlnum
	}
}
