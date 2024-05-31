# Is virus inference

## Installation

1. Install golang
```
sudo apt-get install golang
```
2. Install infer_is_virus
```
go install github.com/neurlang/classifier/cmd/infer_is_virus@latest
```
## Usage

1. Start infer_is_virus
```
~/go/bin/infer_is_virus
```
```
Please choose model number:
0 ) [v0.0.4] - Initial (96%) - Mirror quantum-computing.cz
1 ) [v0.0.4] - Initial (96%) - Mirror quantum-computing.sk
2 ) [v0.0.5] - Improved (97%) - Mirror quantum-computing.cz
3 ) [v0.0.5] - Improved (97%) - Mirror quantum-computing.sk
```
2. Choose model, type `2`
3. Type TLSH hash:
```
T171F58CCBF799592EC87701F28DAE92F6665B8C068A439F036C48371C28775D42F49BD8
```
```
output, T171F58CCBF799592EC87701F28DAE92F6665B8C068A439F036C48371C28775D42F49BD8 , 0, no virus.
Total viruses:  0  total clean:  1
```
