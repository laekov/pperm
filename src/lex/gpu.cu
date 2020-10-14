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
	GPU_ALGO_ARGS

 protected:
  void setup_() override {
		SETUP_GPU_ALGO()
  }

  void generate_() override {
		LAUNCH_GPU_ALGO(genperm_lex_device);
  }
};

REGISTER_PERM_ALGORITHM("lex_gpu", LexGpu)
