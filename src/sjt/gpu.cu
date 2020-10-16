#include <algorithm>
#include <pperm.hh>

__global__ void genperm_sjt_device(int n, int prefix_len, int* counter) {
  int task_idx = threadIdx.x + blockIdx.x * blockDim.x;
  int perm_s = 0;

  int perm[MAX_N], inv[MAX_N]; 
  int8_t a[MAX_N];
  bool dir[MAX_N];

  if (idx2prefix(n, prefix_len, task_idx, a)) {
    for (int i = 0; i < n; ++i) {
      perm[i] = i;
      inv[i] = i;
      dir[i] = false;
    }
    while (true) {
			/*
			for (int i = 0; i < n + prefix_len; ++i) {
				printf("%d/%d ", (int)a[i], perm[i]);
			}
			printf("\n");
			*/
      perm_s += 1;
      int top = -1, nxt;
      for (int i(n - 1); i >= 0; i--) {
        nxt = inv[i] + (dir[i] ? 1 : -1);
        if (nxt >= 0 && nxt < n) {
          if (perm[nxt] < i) {
            top = i;
            break;
          }
        }
      }
      if (top == -1) {
        break;
      }
      swap(a[inv[top]], a[nxt]);
      inv[perm[nxt]] = inv[top];
      perm[inv[top]] = perm[nxt];
      inv[top] = nxt;
      perm[nxt] = top;
      for (int i(top + 1); i < n; i++) dir[i] ^= 1;
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


class SJTGpu: public PermAlgorithm<SJTGpu> {

  GENERATE_CONSTRUCTOR(SJTGpu)

 private:
  GPU_ALGO_ARGS;

 protected:
  void setup_() override {
    SETUP_GPU_ALGO()
  }

 public:
  template <typename F>
  void do_generate_(F&& callback)  {
    LAUNCH_GPU_ALGO(genperm_sjt_device);
  }
};

REGISTER_PERM_ALGORITHM("sjt_gpu", SJTGpu)
