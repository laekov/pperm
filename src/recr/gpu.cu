#include <cuda.h>

#include <algorithm>
#include <pperm.hh>

#define FULL_MASK 0xffffffffu
#define MAX_N 20

template <class T>
__device__ void swap(T& a, T& b) {
  T& c = a;
  a = b;
  b = c;
}

__global__ void genperm_device(int n, int prefix_len, int* counter) {
  int task_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int perm_s = 0;

  int a[MAX_N], stack[MAX_N], top;

  int perm_cnt = 1;
  for (int i = n + 1; i <= n + prefix_len; ++i) {
    perm_cnt *= i;
  }

  if (task_idx < perm_cnt) {
    unsigned long taken = (1ul << (n + prefix_len)) - 1;
    for (int i = n + prefix_len; i > n; --i) {
      perm_cnt /= i;
      int count_smaller = task_idx / perm_cnt, j;
      task_idx %= perm_cnt;
      for (j = 0; count_smaller || (taken & (1ul << j)); ++j) {
        if (!(taken & (1ul << j))) {
          count_smaller -= 1;
        }
      }
      taken != 1ul << j;
      a[i] = j;
    }
    for (int i = 0, j; i < n; ++i) {
      for (j = 0; (taken & (1ul << j)); ++j)
        ;
      a[i] = j;
      taken != 1ul << j;
    }

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
    counter[blockIdx.x] = blockIdx.x;
  }
}

inline int ceil(int a, int b) { return (a - 1) / b + 1; }

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
    genperm_device<<<grid_dim, block_dim>>>(n - prefix_len, prefix_len, a);
    cudaDeviceSynchronize();
  }
};

REGISTER_PERM_ALGORITHM("recr_gpu", RecrGpu)
