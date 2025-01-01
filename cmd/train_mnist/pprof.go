package main

import "runtime/pprof"
import "os"
import "os/signal"
import "syscall"

func init() {
	for _, arg := range os.Args {
		if arg == "-pgo" || arg == "--pgo" {
			// Create a channel to receive OS signals
			sigChan := make(chan os.Signal, 1)

			// Notify the channel on SIGINT and SIGKILL
			signal.Notify(sigChan, syscall.SIGINT, syscall.SIGKILL)

			// Start a goroutine to handle the signals
			go func() {
				// These 4 lines simply collect profile data into default.pgo file
		 		f, _ := os.Create("default.pgo")
				pprof.StartCPUProfile(f)
				// Wait for a signal
				<-sigChan
				pprof.StopCPUProfile()
				f.Close()
				
				os.Exit(130)
				return
			}()
			
			return
		}
	}
}

