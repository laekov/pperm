#include <cuda.h>

#include <algorithm>
#include <pperm.hh>

__global__ void genperm_recr_device(int n, int prefix_len, int* counter) {
  int task_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int perm_s = 0;

  int a[MAX_N], stack[MAX_N], top;

	if (idx2prefix(n, prefix_len, task_idx, a)) {
    stack[top = 0] = 0;
    while (top >= 0) {
      if (top == n) {
        perm_s += 1;
      }
      if (top == n || stack[top] == n) {
        --top;
        if (top >= 0) {
          if (++stack[top] < n) {
            if ((n - top) & 1) {
              int tmp = a[top];
              a[top] = a[n - 1];
              a[n - 1] = tmp;
            } else {
              int tmp = a[top];
              a[top] = a[stack[top]];
              a[stack[top]] = tmp;
            }
          }
        }
        continue;
      }
      if (top + 1 < n) {
        stack[top + 1] = top + 1;
      }
      ++top;
    }
  }

#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    perm_s += __shfl_sync(FULL_MASK, perm_s, (threadIdx.x + i) % 32, 32);
  }
	if (threadIdx.x == 0) {
		counter[blockIdx.x] = 0;
	}
	__syncthreads();
  if (threadIdx.x % 32 == 0) {
    atomicAdd(counter + blockIdx.x, perm_s);
  }
}

class RecrGpu : public PermAlgorithm {
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
    genperm_recr_device<<<grid_dim, block_dim>>>(n - prefix_len, prefix_len, a);
    cudaDeviceSynchronize();
  }
};

REGISTER_PERM_ALGORITHM("recr_gpu", RecrGpu)
