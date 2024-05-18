#include <stdio.h>
#include <stdint.h>



__device__ uint32_t hash(uint32_t n, uint32_t s, uint32_t max) {
	// mixing stage, mix input with salt using subtraction
	// (could also be addition)
	uint32_t m = n - s;

	// hashing stage, use xor shift with prime coefficients
	m ^= m << 2;
	m ^= m << 3;
	m ^= m >> 5;
	m ^= m >> 7;
	m ^= m << 11;
	m ^= m << 13;
	m ^= m >> 17;
	m ^= m << 19;

	// mixing stage 2, mix input with salt using addition
	m += s;

	// modular stage
	// to force output in range 0 to max-1 we could do regular modulo
	// however, the faster multiply shift trick by Daniel Lemire is used instead
	// https://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
	return (uint32_t)(((uint64_t)m * (uint64_t)max) >> 32);
}


__device__ uint32_t real_modulo_recip(uint32_t y) {
	return uint32_t((uint64_t(1) << 32) / uint64_t(y));
}

__device__ uint32_t real_modulo(uint32_t x, uint32_t recip, uint32_t y) {
	return uint32_t((uint64_t(uint32_t((x + 1) * recip)) * uint64_t(y)) >> 32);
}

__device__ int exitFlag = 0;


extern "C" __global__ void reduce(uint8_t *d_set, uint32_t max, uint32_t maxl, uint32_t *alphabet, uint32_t* out) {
	int myFlag = atomicAdd(&exitFlag, 0);
	__syncthreads();
	uint32_t tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t tid_y = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t tid_z = blockIdx.z * blockDim.z + threadIdx.z;

	uint32_t numThreads_x = gridDim.x * blockDim.x;
	uint32_t numThreads_y = gridDim.y * blockDim.y;
	uint32_t numThreads_z = gridDim.z * blockDim.z;

	uint32_t tid = tid_x + tid_y * numThreads_x + tid_z * (numThreads_x * numThreads_y);

	uint32_t totalThreads = numThreads_x * numThreads_y * numThreads_z;
	uint32_t s = tid;
	uint32_t maxl_recip = real_modulo_recip(maxl);
	clock_t start = clock();

	for (; ((clock() - start) / (float)CLOCKS_PER_SEC < 0.1f); s += totalThreads) {
		if (atomicAdd(&exitFlag, 0) != myFlag) {
			return;
		}
		//__syncthreads();
		if (maxl > 4) {
			uint8_t* set = &d_set[tid * ((max + 3) / 4)];
			for (uint32_t i = 0; i < ((max + 3) / 4); i++) {
				set[i] = 0;
			}
			uint32_t i = 0;
			uint32_t v = alphabet[i];
			for (uint32_t j = 0; j < 2 * maxl; j++) {
				i = hash(v, s, max);
				v = alphabet[(j & 1) * maxl + uint32_t(((uint64_t)((i + 1) * maxl_recip) * uint64_t(maxl)) >> 32)];

				uint32_t imodmax = i;
				if ((set[imodmax >> 2] >> ((imodmax & 3) << 1)) & 3 != 0) {
					if ((set[imodmax >> 2] >> ((imodmax & 3) << 1)) & 3 == ((j ^ 1) & 1) + 1) {
						goto next_iteration;
					}
				}
				set[imodmax >> 2] |= (j & 1) + 1 << ((imodmax & 3) << 1);
			}
			if (atomicAdd(&exitFlag, 0) != myFlag) {
				return;
			}
			//__syncthreads();
		}
		for (uint32_t i = 0; i < maxl; i++) {
			uint32_t v = hash(alphabet[i], s, max);
			for (uint32_t j = 0; j < maxl; j++) {
				uint32_t w = hash(alphabet[maxl+j], s, max);
					if (v == w) {
						goto next_iteration;
				}
			}
		}
		if (atomicAdd(&exitFlag, 0) != myFlag) {
			return;
		}
		//__syncthreads();
		// Atomic operations to update output
		out[0] = s;
		out[1] = max;
		atomicAdd(&exitFlag, 1);
		//__syncthreads();

		return;

		next_iteration:;
	}
}

extern "C" void reduceCUDA(uint32_t blk, uint32_t grid, uint32_t max, uint32_t maxl,
				uint32_t *alphabet, uint32_t* result0, uint32_t* result1)
{
	uint32_t result[2] = {0,0};

	uint32_t *d_input;
	uint32_t *d_result;
	uint8_t *d_set;

	int blockSize = (int)blk;
	int gridSize = (int)grid;
	cudaMalloc((void**)&d_input, sizeof(uint32_t) * maxl * 2);
	cudaMalloc((void**)&d_result, sizeof(uint32_t) * 2);
	cudaMalloc((void**)&d_set, sizeof(uint8_t) * blockSize * gridSize);
	cudaMemcpy(d_input, alphabet, sizeof(uint32_t) * maxl * 2, cudaMemcpyHostToDevice);
	reduce<<<gridSize, blockSize>>>(d_set, max, maxl, d_input, d_result);
	cudaMemcpy(&result, d_result, sizeof(uint32_t) * 2, cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_result);
	cudaFree(d_set);

	*result0 = result[0];
	*result1 = result[1];

	return;
}
