/*
 * Smart O(n!) recursion, but use SIMD & MPI
 * For SIMD, we use AVX2 in a way that a 256-bit register is used as 32 int8. So the max possible permutation length is 127.
 * For MPI, we simply use it to split tasks to each core.
 */

#include <immintrin.h>
#include <pperm.hh>

#ifdef PPERM_MPI
#include <mpi.h>
#endif


#ifdef PPERM_AVX2

#define AVX_LANES 32
#define MAX_AVX_REGISTERS 12

class SmartRecrCpuMpiSimd : public PermAlgorithm<SmartRecrCpuMpiSimd> {

 GENERATE_CONSTRUCTOR(SmartRecrCpuMpiSimd)

private:
  int suffix_len;
  int worker_count;
  int prefix_len = 0;
  int task_size = 1;
  int round_count = 0;
  int task_offset;
  int8_t *prefix, *suffix;
  int8_t *stack, top;

 protected:
  void setup_() override {
    // MPI rank * SIMD lanes
    worker_count = mpi_size * AVX_LANES;
    task_offset = mpi_rank * AVX_LANES;
    // estimate the task count on each worker
    while (prefix_len < n) {
      task_size *= (n - prefix_len++);
      if (task_size >= worker_count * distribution_factor) break;
    }
    suffix_len = n - prefix_len;
    round_count = (task_size - 1) / worker_count + 1;
    if (suffix_len <= 0) {
      if (mpi_rank == 0) {
        fprintf(stderr, "Too few tasks, please increase n or decrease worker count * distribution factor\n");
      }
      exit(1);
    } else if (suffix_len > MAX_AVX_REGISTERS) {
      if (mpi_rank == 0) {
        fprintf(stderr, "Too long sequence, please decrease n or increase worker count * distribution factor\n");
      }
      exit(1);
    } else {
      if (mpi_rank == 0) {
        fprintf(stderr, "Prefix / Suffix: %d / %d\n", prefix_len, suffix_len);
        fprintf(stderr, "Tasks / Workers / Rounds: %d / %d / %d\n", task_size, worker_count, round_count);
      }
    }
    prefix = new int8_t[AVX_LANES * prefix_len];
    suffix = new int8_t[AVX_LANES * suffix_len];
    stack = new int8_t[suffix_len];
  }

 public:
  template <typename F>
  void do_generate_(F&& callback) {
    __m256i a[MAX_AVX_REGISTERS];
    bool valid[AVX_LANES] = {false};
    for (int i = 0; i < round_count; ++i) {
      int begin_index = i * worker_count + task_offset;
      for (int j = 0; j < AVX_LANES; ++j) {
        valid[j] = idx2prefix(suffix_len, prefix_len,begin_index + j, prefix + j * prefix_len, suffix + j, AVX_LANES);
      }
      for (int j = 0; j < suffix_len; ++j) {
        a[j] = _mm256_load_si256(reinterpret_cast<const __m256i*>(suffix + AVX_LANES * j));
      }
      stack[top = 0] = 0;
      while (top >= 0) {
        if (top == suffix_len) {
          for (bool j : valid) {
            if (j) callback();
          }
        }
        if (top == suffix_len || stack[top] == suffix_len) {
          --top;
          if (top >= 0) {
            if (++stack[top] < suffix_len) {
              if ((suffix_len - top) & 1) {
                std::swap(a[top], a[suffix_len - 1]);
              } else {
                std::swap(a[top], a[stack[top]]);
              }
            }
          }
          continue;
        }
        if (top + 1 < suffix_len) {
          stack[top + 1] = top + 1;
        }
        ++top;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
};

REGISTER_PERM_ALGORITHM("smart_recr_cpu_mpi_simd", SmartRecrCpuMpiSimd)

#else
#warning "Your processor does not support AVX2, SIMD algorithm will not be included"
#endif
