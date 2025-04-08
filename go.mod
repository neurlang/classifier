module github.com/neurlang/classifier

go 1.18

//replace gorgonia.org/cu => /home/m2/go/src/example.com/repo.git/cu

require (
	github.com/klauspost/cpuid/v2 v2.2.8
	gorgonia.org/cu v0.9.6
)

require github.com/jbarham/primegen v0.0.0-20200302115600-8ce4838491a0

require github.com/neurlang/NumToWordsGo v0.0.0-20250407021724-7b23483906ba // indirect

require (
	github.com/google/uuid v1.1.1 // indirect
	github.com/neurlang/quaternary v0.1.1
	github.com/pkg/errors v0.9.1 // indirect
	golang.org/x/sys v0.5.0 // indirect
)
