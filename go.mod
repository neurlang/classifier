module github.com/neurlang/classifier

go 1.18

//replace gorgonia.org/cu => /home/m2/go/src/example.com/repo.git/cu

require github.com/klauspost/cpuid/v2 v2.2.8

require github.com/jbarham/primegen v0.0.0-20200302115600-8ce4838491a0

require github.com/neurlang/NumToWordsGo v0.0.0-20250407021724-7b23483906ba

require (
	github.com/neurlang/quaternary v0.1.1
	golang.org/x/sys v0.5.0 // indirect
)
