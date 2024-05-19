// Package Cu implements the learning stage of the Neurlang classifier on CUDA
package cu

//import "math/bits"
import "fmt"
import "sync"
import "math/rand"
import crypto_rand "crypto/rand"
import "time"
import "encoding/binary"

import "github.com/neurlang/classifier/datasets"
import "github.com/neurlang/classifier/hash"
import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/learning/cu/kernel"

import "gorgonia.org/cu"

import "unsafe"

type modulo_t = uint32

// Training trains a single hashtron on a dataset d. It outputs the trained hashtron if successful, or an error.
func (h *HyperParameters) Training(d datasets.Splitter) (*hashtron.Hashtron, error) {

	if h.EOL == nil || len(h.EOL) == 0 {
		h.EOL = []byte{';', ' '}
	}

	var sd = d.Split()

	if h.Seed {
		var b [8]byte
		_, err := crypto_rand.Read(b[:])
		if err == nil {
			rand.Seed(int64(binary.LittleEndian.Uint64(b[:])))
		}
	}

	sd = datasets.BalanceDataset(sd)

	var backup = h.InitialLimit
	var result *hashtron.Hashtron

	h.InitialLimit, result = h.Solve(sd)
	for !h.EndWhenSolved || result == nil {
		h.InitialLimit, result = h.Solve(sd)
	}
	h.InitialLimit = backup

	return result, nil
}

func progressBar(progress, width int) string {
	progressBar := ""
	for i := 0; i < progress; i++ {
		progressBar += "="
	}
	return progressBar
}

func emptySpace(space int) string {
	emptySpace := ""
	for i := 0; i < space; i++ {
		emptySpace += " "
	}
	return emptySpace
}

// Solve directly solves a single hashtron on a splitted dataset d. It outputs the size of solution
// and the trained hashtron if successful. Most callers should use Training instead.
func (h *HyperParameters) Solve(d datasets.SplittedDataset) (int, *hashtron.Hashtron) {

	if len(d[1]) == 0 && len(d[0]) == 0 {
		tron, err := hashtron.New(nil, 0)
		if err != nil {
			return h.InitialLimit, nil
		}
		return 1, tron
	}

	var bits uint16
	var alphabet [][]uint32
	for n := range d {

		alphabet = append(alphabet, make([]uint32, 0, len(d[n])))

		for v := range d[n] {
			alphabet[n] = append(alphabet[n], v)
			if n == 0 && bits < uint16(v>>16) {
				bits = uint16(v >> 16)
			}

		}
	}
	if bits >= 64 {
		bits = 0
	}

	var sols [][2]uint32
	var maxl = modulo_t(len(d[0]))
	var maxmaxl = maxl
	var max uint32 = uint32((uint64(maxl) * uint64(maxl)) / uint64(h.Factor))
	var maxmax uint32 = max
	const progressBarWidth = 40
looop:
	for max <= maxmax {
		if !h.DisableProgressBar {
			if maxmaxl > 0 {
				progress := progressBarWidth - int(maxl*progressBarWidth/maxmaxl)
				percent := 100 - int(maxl*100/maxmaxl)
				fmt.Printf("\r[%s%s] %d%% ", progressBar(progress, progressBarWidth), emptySpace(progressBarWidth-progress), percent)
			}
		}
		var alphabet2 = [2][]uint32{alphabet[0], alphabet[1]}
		var sol [2]uint32
		if maxl == 1 {
			sol = h.reduce1(&alphabet2)
		} else if maxl == 2 {
			sol = h.reduce2(&alphabet2)
		} else {
			sol = h.reduce(max, maxl, &alphabet2)
		}
		if sol[1] == 0 {
			if len(sols) > 0 && sols[len(sols)-1][1] > max+1 {
				max++
				continue looop
			}
			return h.InitialLimit, nil
		}
		var set = make(map[uint32]struct{})

		for n := range d {
			dst := alphabet[n][0:0:maxl]
			for i := modulo_t(0); i < maxl; i++ {
				hsh := hash.Hash(alphabet[n][i], sol[0], uint32(sol[1]))
				if _, ok := set[hsh]; ok {
					continue
				}
				dst = append(dst, hsh)
				set[hsh] = struct{}{}
			}
			alphabet[n] = dst
		}
		var correct = len(alphabet[0]) + len(alphabet[1]) < 10
		if correct {
outer:
			for n := uint32(0); n < 2; n++ {
				for _, v := range alphabet[n] {
					if v & 1 != n {
						correct = false
						break outer
					}
				}
			}
		}
		for n := 0; n < 2; n++ {
			for len(alphabet[n]) < len(alphabet[n^1]) {
				var w = uint32(uint16(rand.Uint32()))
				if _, ok := set[w]; ok {
					continue
				}
				alphabet[n] = append(alphabet[n], w)
				set[w] = struct{}{}
			}
		}
		set = nil

		sols = append(sols, sol)

		maxl = modulo_t(len(alphabet[0]))

		if correct || maxl < 2 {
			if len(d) == 2 {
				if alphabet[0][0]&1 == 1 {
					continue looop
				}
				if alphabet[1][0]&1 == 0 {
					continue looop
				}
			}

			if len(sols) >= h.InitialLimit {
				println("SOLUTION SIZE", len(sols), "is below LIMIT ", h.InitialLimit)
				return h.InitialLimit, nil
			}
			var v1decrease uint32
			if len(sols) > 0 {
				v1decrease = sols[0][1] * 2
			}
			for i := range sols {
				sols[i][1], v1decrease = v1decrease-sols[i][1], sols[i][1]
			}
			tron, err := hashtron.New(sols, byte(bits)+1)
			if err != nil {
				println("Error creating hashtron:", err.Error())
				return h.InitialLimit, nil
			}
			if h.l != nil {
				buf, err := tron.BytesBuffer(h.Name, h.EOL...)
				if err != nil {
					println("Hashtron serialization problem:", err.Error())
				} else {
					h.l.Println(buf)
					println("SOLUTION saved! SIZE == ", len(sols))
				}
			} else {
				println("SOLUTION! SIZE == ", len(sols))
			}
			return len(sols), tron
		}

		var sub = h.Subtractor

		if sub >= maxl {
			sub = maxl - 1
		}

		max = uint32(uint64(max) * (uint64(maxl-sub) * uint64(maxl-sub)) / (uint64(maxl) * uint64(maxl)))
		if max <= 1 {
			max = 1
		}
		if max <= maxl {
			max = maxl + 1
		}
	}
	return h.InitialLimit, nil
}

func (h *HyperParameters) reduce2(alphabet *[2][]uint32) (off [2]uint32) {
	var out [2]uint32
	mutex.Lock()
	where++
	mutex.Unlock()
	for t := 1; t < h.Threads; t++ {
		go func(tt byte) {
			mutex.RLock()
			var my_where = where
			mutex.RUnlock()
		outer:
			for s := uint32(tt); true; s += uint32(h.Threads) {
				mutex.RLock()
				if my_where != where || out[0] != 0 || out[1] != 0 {
					mutex.RUnlock()
					return
				} else {
					mutex.RUnlock()
				}
				h00 := hash.Hash(alphabet[0][0], s, 2)
				h01 := hash.Hash(alphabet[0][1], s, 2)
				h10 := hash.Hash(alphabet[1][0], s, 2)
				h11 := hash.Hash(alphabet[1][1], s, 2)

				if h00 == h10 || h00 == h11 {
					continue outer
				}
				if h01 == h10 || h01 == h11 {
					continue outer
				}
				if h00 != h01 {
					continue outer
				}
				if h10 != h11 {
					continue outer
				}
				mutex.Lock()
				if h.DisableProgressBar {
					println("Size: ", "2", "Modulo:", "2")
				}
				//println("{", s, ",", max, "}, // ", len(set0))
				if out[1] > 2 || (out[1] == 0 && out[0] == 0) {
					out[0] = s
					out[1] = 2
				}
				mutex.Unlock()
				return
			}
		}(byte(t))
	}
	mutex.RLock()
	var deadline = h.DeadlineMs
	for out[0] == 0 && out[1] == 0 && deadline > 0 {
		mutex.RUnlock()
		time.Sleep(time.Millisecond)
		deadline--
		mutex.RLock()
	}
	off = out
	mutex.RUnlock()
	if deadline == 0 {
		mutex.Lock()
		out[0] = ^uint32(0)
		out[1] = ^uint32(0)
		where++
		mutex.Unlock()
		if h.DisableProgressBar {
			println("Deadline 2")
		}
	}

	return
}

func (h *HyperParameters) reduce1(alphabet *[2][]uint32) (off [2]uint32) {
	var out [2]uint32
	mutex.Lock()
	where++
	mutex.Unlock()
	for t := 1; t < h.Threads; t++ {
		go func(tt byte) {
			mutex.RLock()
			var my_where = where
			mutex.RUnlock()
		outer:
			for s := uint32(tt); true; s += uint32(h.Threads) {
				mutex.RLock()
				if my_where != where || out[0] != 0 || out[1] != 0 {
					mutex.RUnlock()
					return
				} else {
					mutex.RUnlock()
				}
				if hash.Hash(alphabet[0][0], s, 2)&1 != 0 {
					continue outer
				}
				if hash.Hash(alphabet[1][0], s, 2)&1 != 1 {
					continue outer
				}
				mutex.Lock()
				if h.DisableProgressBar {
					println("Size: ", "1", "Modulo:", "2")
				}
				//println("{", s, ",", max, "}, // ", len(set0))
				if out[1] > 2 || (out[1] == 0 && out[0] == 0) {
					out[0] = s
					out[1] = 2
				}
				mutex.Unlock()
				return
			}
		}(byte(t))
	}
	mutex.RLock()
	var deadline = h.DeadlineMs
	for out[0] == 0 && out[1] == 0 && deadline > 0 {
		mutex.RUnlock()
		time.Sleep(time.Millisecond)
		deadline--
		mutex.RLock()
	}
	off = out
	mutex.RUnlock()
	if deadline == 0 {
		mutex.Lock()
		out[0] = ^uint32(0)
		out[1] = ^uint32(0)
		where++
		mutex.Unlock()
		if h.DisableProgressBar {
			println("Deadline 1")
		}
	}

	return
}

// we do this only once for a faster modulo
func real_modulo_recip(y uint32) uint32 {
	return uint32((uint64(1<<32)) / (uint64(y)))
}

//X % Y = (BITAND(CEILING(X*256/Y),255)*Y)>>8
// manually inlined in reduce
func real_modulo(x, recip, y uint32) uint32 {
	return uint32((uint64(uint32((x+1) * recip)) * uint64(y)) >> 32)
}

// where is used to kill threads when it increases
var where byte
var mutex sync.RWMutex


func nvTasks(tasks int) [2][3]int {
	const (
		MaxThreadsPerBlock = 1024
		MaxGridDimX        = 1024
		MaxGridDimY        = 1024
		MaxGridDimZ        = 64
	)

	// Calculate the number of blocks needed in each dimension
	blockTasks := MaxThreadsPerBlock
	numBlocksX := (tasks + blockTasks - 1) / blockTasks
	numBlocksY := 1
	numBlocksZ := 1

	// Check if the number of blocks exceeds the maximum dimensions
	if numBlocksX > MaxGridDimX {
		numBlocksY = (numBlocksX + MaxGridDimX - 1) / MaxGridDimX
		numBlocksX = MaxGridDimX
	}

	if numBlocksY > MaxGridDimY {
		numBlocksZ = (numBlocksY + MaxGridDimY - 1) / MaxGridDimY
		numBlocksY = MaxGridDimY
	}

	if numBlocksZ > MaxGridDimZ {
		fmt.Println("Too many tasks for the GPU.")
		return [2][3]int{}
	}

	// Return the optimized block and grid dimensions
	return [2][3]int{{32, 32, 1}, {numBlocksX, numBlocksY, numBlocksZ}}
}


func reduceCUDA(tasks int, max, maxl uint32, alphabet []uint32) (result0, result1 uint32) {
	var (
		d_input  cu.DevicePtr
		d_result cu.DevicePtr
		d_set    cu.DevicePtr
		result   [2]uint32
	)

	// Initialize CUDA
	device, err := cu.GetDevice(0)
	if err != nil {
		fmt.Printf("Failed to get device: %v", err)
	}
	ctx, err := device.MakeContext(cu.SchedAuto)
	if err != nil {
		fmt.Printf("Failed to create context: %v", err)
	}
	defer ctx.Destroy()

	// Allocate device memory
	inputSize := int64(maxl) * 2 * int64(unsafe.Sizeof(uint32(0)))
	resultSize := 2 * int64(unsafe.Sizeof(uint32(0)))
	setSize := int64(tasks) * int64((max + 3) / 4) * int64(unsafe.Sizeof(uint8(0)))

	d_input, err = cu.MemAlloc(inputSize)
	if err != nil {
		fmt.Printf("Failed to allocate device memory for input: %v", err)
	}
	defer cu.MemFree(d_input)

	d_result, err = cu.MemAlloc(resultSize)
	if err != nil {
		fmt.Printf("Failed to allocate device memory for result: %v", err)
	}
	defer cu.MemFree(d_result)

	d_set, err = cu.MemAlloc(setSize)
	if err != nil {
		fmt.Printf("Failed to allocate device memory for set: %v", err)
	}
	defer cu.MemFree(d_set)

	cu.MemsetD8(d_result, 0, resultSize)
	cu.MemsetD8(d_set, 0, setSize)

	// Copy data from host to device
	err = cu.MemcpyHtoD(d_input, unsafe.Pointer(&alphabet[0]), inputSize)
	if err != nil {
		fmt.Printf("Failed to copy input data to device: %v", err)
	}

	// Launch the kernel
	mod, err := cu.LoadData(kernel.PTXreduceCUDA)
	if err != nil {
		fmt.Printf("Failed to load module: %v", err)
	}


	fn, err := mod.Function("reduce")
	if err != nil {
		fmt.Printf("Failed to get function: %v", err)
	}

	stream, err := cu.MakeStream(cu.DefaultStream)

	x := nvTasks(tasks)

	args := []unsafe.Pointer{
		unsafe.Pointer(&d_set),
		unsafe.Pointer(&max),
		unsafe.Pointer(&maxl),
		unsafe.Pointer(&d_input),
		unsafe.Pointer(&d_result),
	}


	err = fn.LaunchAndSync(x[1][0], x[1][1], x[1][2], x[0][0], x[0][1], x[0][2], 0, stream, args)
	if err != nil {
		fmt.Printf("Failed to launch kernel: %v", err)
	}


	// Copy result from device to host
	err = cu.MemcpyDtoH(unsafe.Pointer(&result[0]), d_result, resultSize)
	if err != nil {
		fmt.Printf("Failed to copy result data from device: %v", err)
	}

	result0 = result[0]
	result1 = result[1]

	return result0, result1
}

func (h *HyperParameters) reduce(max uint32, maxl modulo_t, alphabet *[2][]uint32) (off [2]uint32) {

	if h.Shuffle {
		rand.Shuffle(int(maxl), func(i, j int) { alphabet[0][i], alphabet[0][j] = alphabet[0][j], alphabet[0][i] })
		rand.Shuffle(int(maxl), func(i, j int) { alphabet[1][i], alphabet[1][j] = alphabet[1][j], alphabet[1][i] })
	}
/*
	fmt.Printf("\nCUDA version: %v\n", cu.Version())
	devices, err := cu.NumDevices()
	if err != nil {
		fmt.Printf("issue found: %s\n", err.Error())
		return
	}
	fmt.Printf("CUDA devices: %v\n\n", devices)

	for d := 0; d < devices; d++ {
		name, _ := cu.Device(d).Name()
		cr, _ := cu.Device(d).Attribute(cu.ClockRate)
		mem, _ := cu.Device(d).TotalMem()
		maj, _ := cu.Device(d).Attribute(cu.ComputeCapabilityMajor)
		min, _ := cu.Device(d).Attribute(cu.ComputeCapabilityMinor)
		fmt.Printf("Device %d\n========\nName      :\t%q\n", d, name)
		fmt.Printf("Clock Rate:\t%v kHz\n", cr)
		fmt.Printf("Memory    :\t%v bytes\n", mem)
		fmt.Printf("Compute   : \t%d.%d\n\n", maj, min)
	}
*/
	alphabetCUDA := make([]uint32, 0, maxl*2)
	for _, v := range alphabet[0] {
		alphabetCUDA = append(alphabetCUDA, v)
	}
	for _, v := range alphabet[1] {
		alphabetCUDA = append(alphabetCUDA, v)
	}
	mem := h.CuMemoryBytes
	if mem == 0 {
		// half of device by default
		memory, err := cu.Device(0).TotalMem()
		if err == nil {
			mem = uint64(memory)/2
		}
		// raise if big problem
		if mem < uint64((max+3)/4) {
			mem = uint64((max+3)/4)
		}
	}

	off[0], off[1] = reduceCUDA(int(mem/uint64((max+3)/4)), max, maxl, alphabetCUDA)

	return
}
