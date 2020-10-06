#include <pperm.hh>
#include <algorithm>
#include <cuda.h>

#define FULL_MASK 0xffffffffu
#define MAX_N 20

template<class T>
__device__ 
void swap(T& a, T& b) {
	T& c = a;
	a = b;
	b = c;
}

__global__
void genperm_device(int n, int prefix_len, int* counter) {
	// int task_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int perm_s = 0;

	int stack[MAX_N], top;

	stack[top = 0] = 0;
	while (top >= 0) {
		if (top == n) {
			perm_s += 1;
		}
		if (top == n || stack[top] == n) {
			--top;
			continue;
		}
		int i = stack[top];
		if (i == top) {
			++stack[top];
			if (top + 1 < n) {
				stack[top + 1] = top + 1;
			}
			++top;
			continue;
		}
		// swap(a[i], a[top]);
		++stack[top];
		if (top + 1 < n) {
			stack[top + 1] = top + 1;
		}
		++top;
		// swap(a[i], a[top]);
	}

#pragma unroll
	for (int i = 1; i < 32; i <<= 1) {
		perm_s += __shfl_sync(FULL_MASK, perm_s, (threadIdx.x + i) % 32, 32);
	}
	if (threadIdx.x == 0) {
		counter[blockIdx.x] = blockIdx.x;
	}
}


inline int ceil(int a, int b) {
	return (a - 1) / b + 1;
}


class GPURecurse: public PermAlgorithm {
private:
	static const int block_size = 512;
	int *a, prefix_len, nth;

protected:
	virtual void setup_() {
		nth = 1;
		prefix_len = 0;
		while (prefix_len < n && nth < 6000) {
			nth *= (n - prefix_len);
			prefix_len += 1;
		}
		cudaMalloc(&a, sizeof(int) * ceil(nth, block_size));
	}

	virtual void generate_() {
		dim3 grid_dim(ceil(nth, block_size));
		dim3 block_dim(block_size);
		genperm_device<<<grid_dim, block_dim>>>(n - prefix_len, prefix_len, a);
		cudaDeviceSynchronize();
	}
};

REGISTER_PERM_ALGORITHM("gpu_recurse", GPURecurse);

