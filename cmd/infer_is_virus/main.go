package main

import "github.com/neurlang/classifier/datasets/isvirus"
import "github.com/neurlang/classifier/hashtron"
import "github.com/neurlang/classifier/layer/majpool2d"
import "github.com/neurlang/classifier/net/feedforward"

import "sync"
import "fmt"
import "bufio"
import "strings"
import "encoding/hex"
import "os"
import "log"
import "net/http"
import "io/ioutil"
import "bytes"
import "encoding/json"
import "compress/gzip"
import "crypto/sha256"

func error_abs(a, b uint32) uint32 {
	if a > b {
		return a - b
	}
	return b - a
}

func get_hashtron(h *hashtron.Hashtron, err error) hashtron.Hashtron {
	if err != nil {
		log.Fatalf(err.Error())
	}
	return *h
}

func download_and_get_model(num int, hash, url string) (net feedforward.FeedforwardNetwork) {

	switch num {
	case 0, 1:
		const fanout = 10
		net.NewLayer(fanout*fanout, 0)
		net.NewCombiner(majpool2d.MustNew(fanout, 1, fanout, 1, fanout, 1, 1))
		net.NewLayer(1, 0)
	case 2, 3:
		const fanout = 13
		net.NewLayer(fanout*fanout, 0)
		net.NewCombiner(majpool2d.MustNew(fanout, 1, fanout, 1, fanout, 1, 1))
		net.NewLayer(1, 0)
	}

	// download
	// URL of the gzip file

	// Step 1: Download the file
	resp, err := http.Get(url)
	if err != nil {
		log.Fatalf("Failed to download file: %v", err)
	}
	defer resp.Body.Close()

	// Step 2: Read the compressed data into memory
	compressedData, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Failed to read compressed data: %v", err)
	}
	

	if hash != fmt.Sprintf("%x", sha256.Sum256(compressedData)) {
		log.Fatalf("Hash of hosted file doesn't match: %s", hash)
	}

	// Step 3: Decompress the data in memory
	gzipReader, err := gzip.NewReader(bytes.NewReader(compressedData))
	if err != nil {
		log.Fatalf("Failed to create gzip reader: %v", err)
	}
	defer gzipReader.Close()

	// Step 4: Read the decompressed data into memory
	decompressedData, err := ioutil.ReadAll(gzipReader)
	if err != nil {
		log.Fatalf("Failed to read decompressed data: %v", err)
	}

	// Step 5: Parse the JSON data into the specified type
	var data [][][2]uint32
	err = json.Unmarshal(decompressedData, &data)
	if err != nil {
		log.Fatalf("Failed to parse JSON data: %v", err)
	}
	// Load the weights into the network
	for i, v := range data {
		*(net.GetHashtron(i)) =  get_hashtron(hashtron.New(v, 1))
	}
	return
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

func main() {

	println("Welcome to infer_is_virus")
	println("commands: <number>, <tlsh hash>, exit")

	var models = [][3]string{
		{
			"0ae3a23ae50e935f3fab464ffc38a021813c6f34f096ddea176ee8716eda030f",
			"[v0.0.4] - Initial (96%) - Mirror quantum-computing.cz",
			"https://quantum-computing.cz/neurlang_initial_v0_0_4_weights.json.gz",
		},
		{
			"0ae3a23ae50e935f3fab464ffc38a021813c6f34f096ddea176ee8716eda030f",
			"[v0.0.4] - Initial (96%) - Mirror quantum-computing.sk",
			"https://quantum-computing.sk/neurlang_initial_v0_0_4_weights.json.gz",
		},
		{
			"6ac88bc3535848cf88d5f1f07913107987e696ac11e4f911ec4a59f2a4df27a3",
			"[v0.0.5] - Improved (97%) - Mirror quantum-computing.cz",
			"https://quantum-computing.cz/neurlang_improved_v0_0_5_weights.json.gz",
		},
		{
			"6ac88bc3535848cf88d5f1f07913107987e696ac11e4f911ec4a59f2a4df27a3",
			"[v0.0.5] - Improved (97%) - Mirror quantum-computing.sk",
			"https://quantum-computing.sk/neurlang_improved_v0_0_5_weights.json.gz",
		},
	}

	println("Please choose model number:")

	for model_id, model_hash_name_url := range models {
		println(model_id, ")", model_hash_name_url[1])
	}

	var interrupt_mut sync.Mutex
	var interrupt bool

	var viruses, cleans int

	var net feedforward.FeedforwardNetwork 

	reader := bufio.NewScanner(os.Stdin)
	println("Please type model number to download or \"exit\":")
	print("")
	for reader.Scan() {
		text := reader.Text()
		interrupt_mut.Lock()
		interrupt = true
		interrupt_mut.Unlock()
		if text == "exit" {
			return
		}
		if len(text) == 1 {
			if text[0] == '#' || text[0] == '/' {
				continue
			}
			num := int(text[0] - '0')
			if num >= 0 && num < len(models) {
				net = download_and_get_model(num, models[num][0], models[num][2])

				println("[backtesting net... please wait, or press enter to interrupt backtesting]")
				
				interrupt_mut.Lock()
				interrupt = false
				interrupt_mut.Unlock()

				go func() {

					var percent_net int
					var errsum uint64
					var percent int
					var scanned int
					for j := range isvirus.Inputs {
						interrupt_mut.Lock()
						interrupted := interrupt
						interrupt_mut.Unlock()
						
						if interrupted {
							break
						}
						
						if percent != j*100/len(isvirus.Inputs) {
							percent = j*100/len(isvirus.Inputs)
							const progressBarWidth = 40
							progress := int(percent*progressBarWidth/100)
							fmt.Printf("\r[%s%s] %d%% ", progressBar(progress, progressBarWidth), emptySpace(progressBarWidth-progress), percent)
						}
					
						{
							var input = isvirus.Input(isvirus.Inputs[j])
							var output = isvirus.Output(isvirus.Outputs[j])

							var predicted = net.Infer(&input).Feature(0)
							if predicted == output.Feature(0) {
								percent_net++
							}
							errsum += uint64(error_abs(predicted, output.Feature(0)))
						}
						scanned ++
					}
					if scanned > 0 {
						println("\n[success rate]", percent_net*100/scanned, "%", "with", errsum, "errors")
						
						println("Please type/paste TLSH hash/hashes to do inference on this model.")
						print("")
					}
				}()
				
			}
		} else if len(text) >= 72 {
			text = strings.Trim(text, " \t")
			bytes, _ := hex.DecodeString(text[2:])
			var buf isvirus.Input
			copy(buf[:], bytes)
			var input = isvirus.Input(buf)

			var predicted = net.Infer(&input).Feature(0)
			
			switch predicted {
				case 0: fmt.Println("\noutput,",text,", 0, no virus."); cleans++
				case 1: fmt.Println("\noutput,",text,", 1, is virus!"); viruses++
			}
			println("Total viruses: ", viruses, " total clean: ", cleans)
			print("")
		}
	}
}
