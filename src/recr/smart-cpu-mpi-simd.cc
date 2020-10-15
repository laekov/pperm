/*
 * Smart O(n!) recursion, but use SIMD & MPI
 * For SIMD, we use AVX2 in a way that a 256-bit register is used as 32 int8. So the max possible permutation length is 127.
 * For MPI, we simply use it to split tasks to each core.
 */

#include <immintrin.h>
#include <pperm.hh>

#ifdef PPERM_AVX2

class SmartRecrCpuMpiSimd : public PermAlgorithm<SmartRecrCpuMpiSimd> {

 GENERATE_CONSTRUCTOR(SmartRecrCpuMpiSimd)

 PPERM_SIMD_INIT

 protected:
  void setup_() override {
    setup_common_();
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
  }
};

REGISTER_PERM_ALGORITHM("smart_recr_cpu_mpi_simd", SmartRecrCpuMpiSimd)

#else
#warning "Your processor does not support AVX2, SIMD algorithm will not be included"
#endif
