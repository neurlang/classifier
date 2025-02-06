// Package Cu implements the learning stage of the Neurlang classifier on CUDA
package cu

//import "math/bits"
import "fmt"
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
//import "sort"

type modulo_t = uint32

const INPUTS = 16
const RESULTS = 50

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

func (h *HyperParameters) cudaTasks(max uint32, problem int) int {
	mem := h.CuMemoryBytes
	if mem == 0 {
		memory, err := cu.Device(0).TotalMem()
		if err == nil && memory > 0 {
			portion := uint64(h.CuMemoryPortion)
			if portion == 0 {
				mem = uint64(problem)  * 10
			} else {
				mem = uint64(memory) / portion
			}
		}
		// raise if big problem
		if mem < uint64(((max+3)/4)+4) {
			mem = uint64(((max + 3) / 4) + 4)
		}
		// cap by avail memory
		if mem > uint64(memory) {
			mem = uint64(memory)
		}
	}
	t := int(mem / uint64(((max+3)/4)+4))
	if t == 0 {
		t = 1
	}
	return t
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
	var maxl = uint32(len(alphabet[0]))
	if len(alphabet[1]) > len(alphabet[0]) {
		maxl = uint32(len(alphabet[1]))
	}
	var maxmaxl = maxl
	var max uint32 = uint32((uint64(maxl) * uint64(maxl)) / uint64(h.Factor))
	var maxmax uint32 = max
	const progressBarWidth = 40
	var center uint32
	var inited bool
	cudaInitFn := func() func() {
		if err := h.initCUDA(max, uint32(len(alphabet[0])), uint32(len(alphabet[1]))); err != nil {
			println(err.Error())
			h.backoff++
			time.Sleep(time.Duration(h.backoff) * time.Millisecond)
			h.backoff <<= 1
			return func() {}
		} else {
			inited = true
			return func() {h.destroyCUDA() }
		}
	}
	var initial = true
	var minadd uint32
	u := uint32(h.DeadlineRetry)
looop:
	for max <= maxmax {
		if max <= 2 {
			max = 2
		}

		if !h.DisableProgressBar {
			if maxmaxl > 0 {
				progress := progressBarWidth - int(maxl*progressBarWidth/maxmaxl)
				percent := 100 - int(maxl*100/maxmaxl)
					fmt.Printf("\r[%s%s] %d%% PROBLEM SIZE = %d ", progressBar(progress, progressBarWidth), emptySpace(progressBarWidth-progress), percent, max)
			}
		}
		var alphabet2 = [2][]uint32{ alphabet[0], alphabet[1] }
		cloneCPU := func() {
			// remove this once CPU Reducing() doesn't mutate the alphabet
			alphabet2 = [2][]uint32{make([]uint32, len(alphabet[0])), make([]uint32, len(alphabet[1]))}
			copy(alphabet2[0], alphabet[0])
			copy(alphabet2[1], alphabet[1])
		}
		var newsols [][2]uint32
		if maxl > h.CuCutoff && h.CuCutoff != 0 {
			cloneCPU()
			//println("\nCPU:")
			newsols = h.H().Reducing(alphabet2, h.CuCutoff, max)
			//println("\nEND CPU:")
		} else if initial {
			cloneCPU()
			//println("\nCPU:")
			newsols = h.H().Reducing(alphabet2, maxl, max)
			//println("\nEND CPU:")
		} else {
			if !inited {
				defer cudaInitFn()()
				if !inited {
					cloneCPU()
					//println("\nCPU:")
					newsols = h.H().Reducing(alphabet2, 0, max)
					//println("\nEND CPU:")
				} else {
					//println("\nCUDA:")
					newsols = h.reduce(center, max, minadd, &alphabet2)
					//println("\nEND CUDA:")
				}
			} else {
				//println("\nCUDA:")
				newsols = h.reduce(center, max, minadd, &alphabet2)
				//println("\nEND CUDA:")
			}
		}
		if len(newsols) == 0 {
			if initial {
				// unstucker
				maxmax = max
				max *= u
				max /= uint32(h.DeadlineRetry + 1)
				if max == 0 {
					break
				}
				if (max * u) / uint32(h.DeadlineRetry + 1) == max {
					break
				}
				continue looop
			} else {
				max++
				minadd += uint32(h.DeadlineMs)
				continue looop
			}
		}
		if newsols[0][1] > max {
			println("CU bug: sol 1 must be max or below")
			continue looop
		}
		var m = max
		for _, sol := range newsols {
			if m < sol[1] {
				fmt.Println(newsols)
				println("CU bug: max increased")
				continue looop
			}
			m = sol[1]
		}
		const PRODUCTION = true
		if PRODUCTION {
			//println("begin", len(newsols))
			for j := range alphabet {
				for i := range alphabet[j] {
					for _, sol := range newsols {
						alphabet[j][i] = hash.Hash(alphabet[j][i], sol[0], sol[1])
					}
				}
				
			}
			for q := range alphabet {
				// Insertion Sort, we're on go 1.16 at the moment
				for i := 1; i < len(alphabet[q]); i++ {
					j := i
					for j > 0 && alphabet[q][j-1] > alphabet[q][j] {
						alphabet[q][j-1], alphabet[q][j] = alphabet[q][j], alphabet[q][j-1]
						j--
					}
				}
				// compact, we're on go 1.16 at the moment
				s := alphabet[q]
				if len(s) > 1 {
					for k := 1; k < len(s); k++ {
						if s[k] == s[k-1] {
							s2 := s[k:]
							for k2 := 1; k2 < len(s2); k2++ {
								if s2[k2] != s2[k2-1] {
									s[k] = s2[k2]
									k++
								}
							}
							s = s[:k]
							break
						}
					}
					alphabet[q] = s
				}
			}
			//println("done", len(newsols))
		} else { // self checking
			var set = [2]map[uint32]struct{} {make(map[uint32]struct{}), make(map[uint32]struct{})}
		checkloop:
			for q := range newsols {
				if q == 0 {
					for j := range alphabet {
						for i := range alphabet[j] {
							set[j][hash.Hash(alphabet[j][i], newsols[0][0], newsols[0][1])] = struct{}{}
						}
					}
					for v0 := range set[0] {
						for v1 := range set[1] {
							if v0 == v1 {
								println("CU bug: sets overlapped")
								continue looop
							}
						}
					}
				} else {
					var set_next = [2]map[uint32]struct{} {make(map[uint32]struct{}), make(map[uint32]struct{})}
					for j := range set {
						for i := range set[j] {
							set_next[j][hash.Hash(i, newsols[q][0], newsols[q][1])] = struct{}{}
						}
					}
					for v0 := range set[0] {
						for v1 := range set[1] {
							if v0 == v1 {
								println("CU bug: sets overlapped")
								newsols = newsols[:q-1]
								break checkloop
							}
						}
					}
					set = set_next
				}
			}
			alphabet[0] = alphabet[0][:0]
			alphabet[1] = alphabet[1][:0]

			for j := range set {
				for v := range set[j] {
					alphabet[j] = append(alphabet[j], v)
				}
			}
		}


		//println("done sets")


		sols = append(sols, newsols...)
		win_centers := sols[len(sols)-1][0]
		max = sols[len(sols)-1][1]
		maxmax = max



		initial = false
		maxl = uint32(len(alphabet[0]))
		if len(alphabet[1]) > len(alphabet[0]) {
			maxl = uint32(len(alphabet[1]))
		}
		if maxl == 1 && len(alphabet[0]) == len(alphabet[1]) && alphabet[0][0] == 0 && alphabet[1][0] == 1 {
			goto success
		}
		var sub = h.Subtractor
		if sub > maxl {
			sub = maxl - 1
		}
		newmaxx := uint32(uint64(max) * (uint64(maxl-sub) * uint64(maxl-sub)) / (uint64(maxl) * uint64(maxl)))
		if newmaxx >= max {
			minadd = 0
			center = win_centers // store last solution salt as the future center for xor search heuristics
		} else {
			maxmax = max
			max = newmaxx
			minadd = 0
			center = win_centers // store last solution salt as the future center for xor search heuristics
		}
		if max <= maxl {
			max = maxl
		}
		if max == 0 {
			break
		}
	}
	u--
	if u == 0 {
		return h.InitialLimit, nil
	}
	goto looop
success:
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

	if numBlocksX == 0 {
		numBlocksX = 1
	}

	//fmt.Println(numBlocksX, numBlocksY, numBlocksZ)

	// Return the optimized block and grid dimensions
	return [2][3]int{{32, 32, 1}, {numBlocksX, numBlocksY, numBlocksZ}}
}

func (h *HyperParameters) initCUDA(max, l0, l1 uint32) error {
	if l0 == 0 || l1 == 0 {
		panic("one set is empty")
	}

	// Initialize CUDA
	device, err := cu.GetDevice(0)
	if err != nil {
		fmt.Printf("Failed to get device: %v\n", err)
		return err
	}
	ctx := cu.NewContext(device, cu.SchedAuto)
	// Lock context for thread safety
	err = ctx.Lock()
	if err != nil {
		fmt.Printf("Failed to lock context: %v\n", err)
		return err
	}

	input0Size := int64(l0) * int64(unsafe.Sizeof(uint32(0)))
	d_input0, err := cu.MemAlloc(input0Size)
	if err != nil {
		fmt.Printf("Failed to allocate device memory for input0: %v\n", err)
		return err
	}
	input1Size := int64(l1) * int64(unsafe.Sizeof(uint32(0)))
	d_input1, err := cu.MemAlloc(input1Size)
	if err != nil {
		fmt.Printf("Failed to allocate device memory for input1: %v\n", err)
		return err
	}
	inputsSize := INPUTS * int64(unsafe.Sizeof(uint32(0)))
	d_input_nums, err := cu.MemAlloc(inputsSize)
	if err != nil {
		fmt.Printf("Failed to allocate device memory for inputs: %v\n", err)
		return err
	}
	resultSize := RESULTS * 2 * int64(unsafe.Sizeof(uint32(0)))
	d_result, err := cu.MemAlloc(resultSize)
	if err != nil {
		fmt.Printf("Failed to allocate device memory for result: %v\n", err)
		return err
	}
	arenaSize := int64(h.CuArenaBytes) * int64(unsafe.Sizeof(uint32(0)))
	if h.CuArenaBytes == 0 {
		arenaSize = 1 * int64(unsafe.Sizeof(uint32(0)))
	}
	d_arena, err := cu.MemAlloc(arenaSize)
	if err != nil {
		fmt.Printf("Failed to allocate device memory for arena: %v\n", err)
		return err
	}
	// Launch the kernel
	mod, err := cu.LoadData(kernel.PTXreduceCUDA)
	if err != nil {
		fmt.Printf("Failed to load module: %v\n", err)
		return err
	}

	fn, err := mod.Function("reduce")
	if err != nil {
		fmt.Printf("Failed to get function: %v\n", err)
		return err
	}

	stream, err := ctx.MakeStream(cu.NonBlocking)
	if err != nil {
		fmt.Printf("Failed to make stream: %v\n", err)
		return err
	}
	h.ctx = ctx
	h.input0 = &d_input0
	h.input1 = &d_input1
	h.inputNums = &d_input_nums
	h.result = &d_result
	h.arena = &d_arena
	h.fn = &fn
	h.stream = &stream
	return nil
}
func (h *HyperParameters) destroyCUDA() {
	h.fn = nil
	if h.stream != nil {
		h.stream.Destroy()
		h.stream = nil
	}
	if h.input0 != nil {
		cu.MemFree(*h.input0)
		h.input0 = nil
	}
	if h.input1 != nil {
		cu.MemFree(*h.input1)
		h.input1 = nil
	}
	if h.inputNums != nil {
		cu.MemFree(*h.inputNums)
		h.inputNums = nil
	}
	if h.result != nil {
		cu.MemFree(*h.result)
		h.result = nil
	}
	if h.arena != nil {
		cu.MemFree(*h.arena)
		h.arena = nil
	}
	if h.set != nil {
		cu.MemFree(*h.set)
		h.set = nil
		h.setSize = 0
	}
	if h.ctx != nil {
		h.ctx.Unlock()
		h.ctx.Destroy()
		h.ctx = nil
	}
}

func (h *HyperParameters) reduceCUDA(tasks int, center, max, minadd uint32, alphabet [2][]uint32) (results [][2]uint32) {
	var (
		d_input0 cu.DevicePtr
		d_input1 cu.DevicePtr
		d_result cu.DevicePtr
		d_set    cu.DevicePtr
	)

	if max < 2 {
		panic("max is small")
	}

	if tasks == 0 {
		panic("there are no tasks")
	}
	var result = make([]uint32, 2*RESULTS, 2*RESULTS)
	x := nvTasks(tasks)

	// Allocate device memory
	input0Size := int64(len(alphabet[0])) * int64(unsafe.Sizeof(uint32(0)))
	input1Size := int64(len(alphabet[1])) * int64(unsafe.Sizeof(uint32(0)))
	inputNumsSize := INPUTS * int64(unsafe.Sizeof(uint32(0)))
	resultSize := RESULTS * 2 * int64(unsafe.Sizeof(uint32(0)))
	setSize := int64(tasks) * int64(((max+3)/4)+4) * int64(unsafe.Sizeof(uint8(0)))
	_ = inputNumsSize
	d_input0 = *h.input0
	d_input1 = *h.input1
	d_result = *h.result
	d_arena := *h.arena
	d_fn := *h.fn
	d_stream := *h.stream
	d_input_nums := *h.inputNums
	//d_ctx := *h.ctx

	var err error

	if h.set != nil {
		d_set = *h.set
		if h.setSize > 2*setSize || h.setSize < setSize {
			err = cu.MemFree(d_set)
			if err != nil {
				fmt.Printf("Failed to free set: %v\n", err)
				return
			}
			d_set, err = cu.MemAlloc(setSize)
			if err != nil {
				fmt.Printf("Failed to allocate device memory for set: %v\n", err)
				return
			}
			h.setSize = setSize
			h.set = &d_set
		}
	} else {
		if setSize == 0 {
			setSize = 4
		}
		d_set, err = cu.MemAlloc(setSize)
		if err != nil {
			fmt.Printf("Failed to initially allocate device memory for set: %v\n", err)
			return
		}
		h.setSize = setSize
		h.set = &d_set
	}



	err = cu.MemsetD8Async(d_result, 0, resultSize, *h.stream)
	if err != nil {
		fmt.Printf("Failed to set device memory for result: %v\n", err)
		return
	}

	if h.CuErase {
		err = cu.MemsetD8Async(d_set, 0, setSize, *h.stream)
		if err != nil {
			fmt.Printf("Failed to set device memory for set: %v\n", err)
			return
		}
	}

	// Copy data from host to device
	err = cu.MemcpyHtoDAsync(d_input0, unsafe.Pointer(&alphabet[0][0]), input0Size, d_stream)
	if err != nil {
		fmt.Printf("Failed to copy input data 0 to device: %v\n", err)
		return
	}
	err = cu.MemcpyHtoDAsync(d_input1, unsafe.Pointer(&alphabet[1][0]), input1Size, d_stream)
	if err != nil {
		fmt.Printf("Failed to copy input data 1 to device: %v\n", err)
		return
	}

	var res = 2*uint32(RESULTS)
	var input_numbers = [INPUTS]uint32{max, uint32(len(alphabet[0])), uint32(len(alphabet[1])),
					uint32(h.DeadlineMs*RESULTS), uint32(tasks), h.iter, center,
					0, 0, res, minadd, uint32(h.CuArenaBytes), 0, 0, 0, h.Subtractor}

	//fmt.Println(input_numbers, x, input0Size, input1Size, inputNumsSize, resultSize, setSize)

	err = cu.MemcpyHtoDAsync(d_input_nums, unsafe.Pointer(&input_numbers[0]), inputNumsSize, d_stream)
	if err != nil {
		fmt.Printf("Failed to copy input data to device: %v\n", err)
		return
	}

	//println(d_set, d_input_nums, d_input0, d_input1, d_result)
	args := []unsafe.Pointer{
		unsafe.Pointer(&d_set),
		unsafe.Pointer(&d_input_nums),
		unsafe.Pointer(&d_input0),
		unsafe.Pointer(&d_input1),
		unsafe.Pointer(&d_arena),
		unsafe.Pointer(&d_result),
	}
	//println("entering kernel")
	err = d_fn.Launch(x[1][0], x[1][1], x[1][2], x[0][0], x[0][1], x[0][2], 0, d_stream, args)
	if err != nil {
		fmt.Printf("Failed to launch kernel: %v\n", err)
		return
	}
	//println("after kernel, gonna copy:", resultSize, "bytes")
	{
		// Copy result from device to host
		err = cu.MemcpyDtoHAsync(unsafe.Pointer(&result[0]), d_result, resultSize, d_stream)
		if err != nil {
			fmt.Printf("Failed to copy result data from device: %v\n", err)
			return
		}
	}
/*
	{
		// Synchronize the stream to ensure the memcpy completes
		err = d_stream.Synchronize()
		if err != nil {
			fmt.Printf("Stream synchronization failed after async memcpy: %v\n", err)
			return
		}
	}
*/
	h.iter++
/*
	{
		errChan := make(chan error)
		go d_ctx.Run(errChan);
		if err := <- errChan; err != nil {
			fmt.Printf("Context run error: %v\n", err)
			return
		}
		
	}
*/
	for i := 0; i < RESULTS; i++ {
		result0 := result[2*i+0]
		result1 := result[2*i+1]
		if result0 == 0 && result1 == 0 {
			break
		}
		if result1 > max {
			println("CU BUG: result1 has increaseed past max:", result1, max)
			break
		}
		max = result1
		results = append(results, [2]uint32{result0, result1})
		//println(i, result0, result1)
	}

	//println("results:", max, len(results), h.iter)

	return results
}

func (h *HyperParameters) reduce(center, max, minadd uint32, alphabet *[2][]uint32) (off [][2]uint32) {

	if h.Shuffle {
		rand.Shuffle(len(alphabet[0]), func(i, j int) { alphabet[0][i], alphabet[0][j] = alphabet[0][j], alphabet[0][i] })
		rand.Shuffle(len(alphabet[1]), func(i, j int) { alphabet[1][i], alphabet[1][j] = alphabet[1][j], alphabet[1][i] })
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

	if max == 0 {
		panic("max is 0")
	}

	return h.reduceCUDA(h.cudaTasks(max, len(alphabet[0]) + len(alphabet[1])), center, max, minadd, *alphabet)
}
