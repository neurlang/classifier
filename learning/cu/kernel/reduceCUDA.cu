// reduce_kernel.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// Device hash function (adapted from the Go version)
__device__ uint32_t hash_d(uint32_t n, uint32_t salt, uint32_t maxx) {
    uint32_t m = n - salt;
    m ^= m << 2;
    m ^= m << 3;
    m ^= m >> 5;
    m ^= m >> 7;
    m ^= m << 11;
    m ^= m << 13;
    m ^= m >> 17;
    m ^= m << 19;
    m += salt;
    return (uint32_t)(((uint64_t)m * maxx) >> 32);
}

/*
  d_nums is expected to have the following layout:
    d_nums[0] : maxx
    d_nums[1] : len0  (current length of alphabet0)
    d_nums[2] : len1  (current length of alphabet1)
    d_nums[3] : DeadlineMs
    d_nums[4] : tasks (how many parallel entities should run)
    d_nums[5] : iteration (current iteration; may be used for exit checking)
    d_nums[6] : center
    d_nums[7] : out_offset  (used to assign program entries; each entry takes 2 uint32_t)
    d_nums[8] : minsub
    d_nums[9] : res (number of result tuples which can be stored in out)
    d_nums[10]: minadd
    d_nums[11]: arenasize
    d_nums[12]: mutex
    d_nums[13]: mustExit
    d_nums[14]: allocator
    d_nums[15]: subtractor
*/
// Declare the mutex variable in device memory.
// Mark it as volatile to prevent unwanted compiler optimizations.

// Declare the mutex variable in device memory.
// Mark it as volatile to prevent unwanted compiler optimizations.

__device__ static uint32_t hashCounter = 0;

__device__ static uint32_t* alpha0 = 0;
__device__ static uint32_t* alpha1 = 0;
///__device__ static uint32_t* free0 = 0;
///__device__ static uint32_t* free1 = 0;

// Full kernel definition with no omitted parts.
extern "C" __global__ void reduce(uint8_t *d_set, uint32_t *d_nums,
                                    uint32_t *alphabet0, uint32_t *alphabet1,
                                    uint32_t* arena, uint32_t* out) {

	// Lock the mutex: attempt to acquire the lock.
	auto lockMutex = [&] __device__ () -> int {
		
	    // Try to acquire the lock by comparing the value at the address of globalMutex with 0.
	    // If it is 0, then set it to 1 and the lock is acquired.
	    if (atomicCAS((uint32_t *)(void *)&d_nums[12], 0, 1) != 0) {
			if (atomicCAS((uint32_t *)(void *)&d_nums[12], 0, 1) != 0) {
				if (atomicCAS((uint32_t *)(void *)&d_nums[12], 0, 1) != 0) {
					return 1;
				}
			}
		}
	    // Memory fence to ensure subsequent memory accesses see updated values.
	    
	    return 0;
	};

	// Unlock the mutex.
	auto unlockMutex = [&] __device__ () -> void {
	    // Memory fence to ensure all previous writes are visible.
	    
	    atomicExch((uint32_t *)&d_nums[12], 0);
	    
	};

        uint32_t iteration = d_nums[5];  // not used explicitly for exit here
    // Compute a unique thread ID (we assume a full grid launch)
	int myFlag = iteration;

	// Must exit.
	auto mustExit = [&] __device__ () -> int {
	    // Memory fence to ensure all previous writes are visible.
	    
	    int ret = atomicAdd((uint32_t *)(void *)&d_nums[13], 0) > myFlag;
	    
	    return ret;
	};


	alpha0 = alphabet0;
	alpha1 = alphabet1;

	auto allocate = [&] __device__ (uint32_t sizee) -> uint32_t {
		return atomicAdd((uint32_t *)(void *)&d_nums[14], sizee);
	};




        //uint32_t maxmax      = d_nums[0];
        uint32_t scratch      = (((d_nums[0] + 3) / 4) + 4);

	uint64_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	uint64_t tid_z = blockIdx.z * blockDim.z + threadIdx.z;

	uint64_t numThreads_x = gridDim.x * blockDim.x;
	uint64_t numThreads_y = gridDim.y * blockDim.y;

	uint64_t tid = tid_x + tid_y * numThreads_x + tid_z * (numThreads_x * numThreads_y);

    // Load total number of nonce iterations (tasks)
    uint32_t tasks = d_nums[4];
    uint32_t arenasize    = d_nums[11];
    uint32_t DeadlineMs = d_nums[3];
	if ((uint32_t)(tid) >= tasks) {
		// stop unwanted threads
		return;
	}

    // Each thread processes a subset of nonce iterations using a grid–stride loop.
    for (uint32_t iterator = 0; iterator < 1 + DeadlineMs; iterator ++) {
	if (mustExit()) {
		goto exit;
	}


        // Reconstruct globals from d_nums.
	alphabet0 = alpha0;
	alphabet1 = alpha1;
        uint32_t maxx       = d_nums[0];
        uint32_t len0       = d_nums[1];
        uint32_t len1       = d_nums[2];
        //uint32_t DeadlineMs = d_nums[3];
        uint32_t center     = d_nums[6];
        // d_nums[7] is used as out_offset (in units of uint32_t)
        // d_nums[8] is minsub.
	//uint32_t phase     = d_nums[7];
        //uint32_t minsub    = d_nums[8];
        uint32_t res        = d_nums[9];
        uint32_t minadd     = d_nums[10];
        uint32_t subtractor = d_nums[15];
	//uint32_t u = 10;
	//uint32_t retry = 10;
	
	// if program offset exceeded the results, our solution wouldn't be possible
	// to be added, so we exit here
	if (atomicAdd(&d_nums[7], 0) >= res) {
		goto exit;
	}

/*
        // --- Deadline / Unstucker check ---
        if (DeadlineMs > 0 && (iterator % DeadlineMs == (DeadlineMs - 1))) {
            //minsub = nonce;
            iterator = 0;
            // If phase is 0 then we are in the initial phase.
            if (phase == 0) {
		//printf("[%d %d] %d\n", (uint32_t)tid, tasks, maxx);
		// unstucker
		maxx *= u;
		maxx /= uint32_t(retry + 1);
		if (maxx == 0) {
			goto exit;
		}
		atomicExch(&d_nums[0], maxx);
            } else {
		//printf("[%d %d] %d\n", (uint32_t)tid, tasks, maxx);
                // Otherwise, increment maxx.
		atomicAdd(&d_nums[0], 1);
		maxx++;
		if (maxx > maxmax) {
			u--;
			if (u == 0) {
				goto exit;
			}
			atomicExch(&d_nums[0], 0);
			maxx = 0;
		}
                // (You might want to compare against an upper bound and adjust here.)
            }
            // Atomically update d_nums[0] and d_nums[8] with the new maxx and minsub.
            //atomicExch(&d_nums[8], minsub);
            // Proceed to the next master loop iteration.
            continue;
        }
*/
        // --- Termination condition ---
        // If the alphabets have converged to a single element (alphabet0 = [0] and alphabet1 = [1]), exit.
        if ((len0 == 1) && (len1 == 1)) {
            if (alpha0[0] == 0 && alpha1[0] == 1) {
                goto exit;
            }
        }

        // --- Compute candidate salt ("centers") ---
        uint32_t centers = center ^ (minadd + atomicAdd((uint32_t*)(void*)&hashCounter, 1));
	
        // --- Set up per-thread scratch buffer ---
        // Each thread's scratch region is a contiguous block of bytes in d_set.
	uint8_t* buf = &d_set[(uint32_t)(tid) * scratch];
        // We use a simple scheme: our "buf" is divided into "subwords" of 1 byte each.
        // For our purposes, subwords is 4, meaning we treat each hash output v as selecting a byte and 2 bits within it.
        const uint32_t subwords = 4;


        // Zero-out the scratch buffer.
        for (uint32_t i = 0; i < scratch; i++) {
            buf[i] = 0;
        }

        // Local counters for new alphabet sizes.
        uint32_t size0 = 0, size1 = 0;
        const uint8_t twobitmask = 3;

        // Define inline lambda to update the scratch buffer for a given hash value.
        auto isvBad = [&] __device__ (uint32_t v, uint8_t which) -> bool {
            uint32_t w0 = v / subwords;
            uint32_t w1 = (v % subwords) << 1;
            uint8_t current = (buf[w0] >> w1) & twobitmask;
            if (current == 0) {
                if (which == 0)
                    size0++;
                else
                    size1++;
            }
            buf[w0] |= ((1 + which) << w1);
            // If adjacent bits are both nonzero, consider this a collision.
            return ((buf[w0] & (buf[w0] >> 1)) != 0);
        };

        // Define inline lambda to "mark" a hash value.
        auto mark = [&] __device__ (uint32_t v) -> bool {
            uint32_t w0 = v / subwords;
            uint32_t w1 = (v % subwords) << 1;
            if (((buf[w0] >> w1) & twobitmask) == twobitmask)
                return false;
            buf[w0] |= (twobitmask << w1);
            return true;
        };

	if (mustExit()) {
		goto exit;
	}

        bool skip = false;
        // Process the first min(len0,len1) entries from both alphabets.
        uint32_t minl = (len0 < len1) ? len0 : len1;
        for (uint32_t i = 0; i < minl; i++) {
            for (uint8_t j = 0; j < 2; j++) {
                uint32_t val = (j == 0) ? alpha0[i] : alpha1[i];
                uint32_t v = hash_d(val, centers, maxx);
                if (isvBad(v, j)) {
                    skip = true;
                    break;
                }
            }
            if (skip)
                break;
        }
        if (skip)
            continue;

	if (mustExit()) {
		goto exit;
	}

        // Process the remaining entries for each alphabet.
        for (uint8_t j = 0; j < 2; j++) {
            uint32_t currentLen = (j == 0) ? len0 : len1;
            for (uint32_t i = minl; i < currentLen; i++) {
                uint32_t val = (j == 0) ? alpha0[i] : alpha1[i];
                uint32_t v = hash_d(val, centers, maxx);
                if (isvBad(v, j)) {
                    skip = true;
                    break;
                }
            }
            if (skip)
                break;
        }
        if (skip)
            continue;

        // Early exit: if the total number of distinct markings is not 2
        // and equals the total number of inputs, then skip this candidate.
        if (((size0 + size1) != 2) && ((len0 + len1) == (size0 + size1)))
            continue;
	
	//printf("[%d] %d %d\n", (uint32_t)tid, centers, maxx);


	if (mustExit()) {
		goto exit;
	}

        // --- Candidate accepted: build new alphabets ---
        uint32_t win_centers = centers;
        // Temporary arrays for new alphabets.
	uint32_t add0 = allocate(size0);
        uint32_t* new_alpha0 = &arena[add0];
	if ((new_alpha0 == 0) || (size0+add0 >= arenasize)) {
		new_alpha0 = 0;
	}
	uint32_t add1 = allocate(size1);
        uint32_t* new_alpha1 = &arena[add1];
	if ((new_alpha1 == 0) || (size1+add1 >= arenasize)) {
		///free(new_alpha0);
		new_alpha1 = 0;
	}
	if ((new_alpha0 != 0) && (new_alpha1 != 0)) {
		uint32_t count0 = 0, count1 = 0;
		for (uint8_t j = 0; j < 2; j++) {
		    uint32_t currentLen = (j == 0) ? len0 : len1;
		    for (uint32_t i = 0; i < currentLen; i++) {
		        uint32_t val = (j == 0) ? alpha0[i] : alpha1[i];
		        uint32_t v = hash_d(val, centers, maxx);
		        if (mark(v)) {
		            if (j == 0)
		                new_alpha0[count0++] = v;
		            else
		                new_alpha1[count1++] = v;
		        }
		    }
		    if (skip)
		        break;
		}
		if (skip)
		    continue;
		size0 = count0;
		size1 = count1;
	}
	//

	// THE MUTEX
	if (lockMutex()) {
		///free(new_alpha0);
		///free(new_alpha1);
		continue;
	}
	if (mustExit()) {
		///free(new_alpha0);
		///free(new_alpha1);
		goto mutex_release_exit;
	}

        // --- Write a "program" entry (a pair: win_centers and maxx) ---
        // We use atomicAdd on d_nums[7] to reserve two uint32_t slots.
        uint32_t progOffset = atomicAdd(&d_nums[7], 2);
	//printf("[%d] %d %d %d %d\n", (uint32_t)tid, centers, maxx, progOffset, res);
	if (progOffset >= res) {
		///free(new_alpha0);
		///free(new_alpha1);
        	goto mutex_release_exit;
	}
	if (progOffset != 0) {
		if (atomicAdd(& out[progOffset - 1], 0) <= maxx) {
			///free(new_alpha0);
			///free(new_alpha1);
        		goto mutex_release_exit;
		}
	}
        atomicAdd(&out[progOffset], win_centers);
        atomicAdd(& out[progOffset + 1], maxx);
	if (progOffset+2 >= res) {
		///free(new_alpha0);
		///free(new_alpha1);
        	goto mutex_release_exit;
	}

	// we can't continue forcing since the arena was exhausted
	if ((new_alpha0 == 0) || (new_alpha1 == 0)) {
        	goto mutex_release_exit;
	}

        // --- Update global parameters based on new alphabets ---
        uint32_t new_maxl = (size0 > size1) ? size0 : size1;
        // Use a simple subtractor (could be replaced by a parameter from the host)
        uint32_t sub = subtractor;
        if (sub > new_maxl)
            sub = new_maxl - 1;
        // Compute newmaxx: maxx * ((new_maxl - sub)^2) / (new_maxl^2)
        uint32_t newmaxx = (uint32_t)(((uint64_t)maxx * (new_maxl - sub) * (new_maxl - sub)) /
                                      ((uint64_t)new_maxl * new_maxl));

        if (newmaxx <= 1) {
		//printf("[%d] %d <= 1\n", (uint32_t)tid, newmaxx);
		///free(new_alpha0);
		///free(new_alpha1);
            goto mutex_release_exit;
	}

        if (newmaxx >= maxx) {
		//printf("[%d] %d >= %d\n", (uint32_t)tid, newmaxx, maxx);
            //minsub = 0;
            center = win_centers;
		///free(new_alpha0);
		///free(new_alpha1);
            goto mutex_release_exit;
        } else {
            maxx = newmaxx;
            //minsub = nonce;
            center = win_centers;
        }
        if (maxx <= new_maxl)
            maxx = new_maxl;

	
        // Update the global parameters in d_nums.
        atomicExch(&d_nums[0], maxx);
        atomicExch(&d_nums[1], size0);
        atomicExch(&d_nums[2], size1);
        atomicExch(&d_nums[6], center);
        //atomicExch(&d_nums[8], minsub);

        // Overwrite the global alphabets with the new alphabets.
        // (This code assumes that count0 and count1 do not exceed the allocated sizes.)
	{
		if (alphabet0 != alpha0 && alphabet1 != alpha1) {
			///uint32_t* ptr0 = alpha0;
			///uint32_t* ptr1 = alpha1;
			alpha0 = new_alpha0;
			alpha1 = new_alpha1;
			///free(free0);
			///free(free1);
			///free0 = ptr0;
			///free1 = ptr1;
			
		} else {
			alpha0 = new_alpha0;
			alpha1 = new_alpha1;
			
		}
	};
	unlockMutex();


    } // end grid-stride loop over nonce
mutex_release_exit:
	unlockMutex();
        {
		if (alphabet0 != alpha0 && alphabet1 != alpha1) {
			///uint32_t* ptr0 = alpha0;
			///uint32_t* ptr1 = alpha1;
			alpha0 = alphabet0;
			alpha1 = alphabet1;
			///free(free0);
			///free(free1);
			///free0 = ptr0;
			///free1 = ptr1;

			
		}
	};
exit:
	//printf("[%d] Timeouted\n", (uint32_t)tid);
	// once some thread expires the timeout, all threads must quit
	//cudaFree(data0); cudaFree(data1);
	///free(free0);
	///free(free1);
	///free0 = 0;
	///free1 = 0;
	
	atomicAdd((uint32_t *)(void *)&d_nums[13], myFlag+1);
	
	return;

}

