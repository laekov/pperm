#include <algorithm>
#include <iostream>
#include <pperm.hh>


__global__ void genperm_lex_device(int n, int prefix_len, int* counter) {
  int task_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int perm_s = 0;

  int a[MAX_N];

  if (idx2prefix(n, prefix_len, task_idx, a)) {
    while (true) {
      ++perm_s;
      bool Flag = false;
      for (int i(n - 2); i >= 0; i--)
        if (a[i] < a[i + 1]) {
          int x = i + 1;
          for (int j(i + 2); j < n; j++)
            if (a[j] > a[i] && a[j] < a[x]) x = j;
          Flag = true;
          swap(a[i], a[x]);
          int R = n - 1;
          for (int L(i + 1); L < R;) swap(a[L++], a[R--]);
          break;
        }
      if (!Flag) break;
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


class LexGpu: public PermAlgorithm {
 private:
  static const int block_size = 512;
  int *a, prefix_len, nth;

 protected:
  void setup_() override {
    nth = 1;
    prefix_len = 0;
    while (prefix_len < n && nth < 6000) {
      nth *= (n - prefix_len);
      prefix_len += 1;
    }
    cudaMalloc(&a, sizeof(int) * ceil(nth, block_size));
  }

  void generate_() override {
    dim3 grid_dim(ceil(nth, block_size));
    dim3 block_dim(block_size);
    genperm_lex_device<<<grid_dim, block_dim>>>(n - prefix_len, prefix_len, a);
    cudaDeviceSynchronize();
  }
};

REGISTER_PERM_ALGORITHM("lex_gpu", LexGpu)
