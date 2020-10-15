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

class RecrGpu : public PermAlgorithm<RecrGpu> {

  GENERATE_CONSTRUCTOR(RecrGpu)

 private:
	GPU_ALGO_ARGS

 protected:
  void setup_() override {
		SETUP_GPU_ALGO()
  }

 public:
  template <typename F>
  void do_generate_(F&& callback)  {
		LAUNCH_GPU_ALGO(genperm_recr_device);
  }
};

REGISTER_PERM_ALGORITHM("recr_gpu", RecrGpu)
