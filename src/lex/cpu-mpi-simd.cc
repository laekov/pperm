#include <immintrin.h>
#include <pperm.hh>

#ifdef PPERM_MPI
#include <mpi.h>
#endif

#ifdef PPERM_AVX2

class LexCpuMpiSimd : public PermAlgorithm<LexCpuMpiSimd> {
 GENERATE_CONSTRUCTOR(LexCpuMpiSimd)

 PPERM_SIMD_INIT

 private:
  int8_t *idx;

 protected:
  void setup_() override {
    setup_common_();
    idx = new int8_t[suffix_len];
  }

 public:
  template <typename F>
  void do_generate_(F &&callback) {

    __m256i a[MAX_AVX_REGISTERS];
    bool valid[AVX_LANES] = {false};

    for (int i = 0; i < round_count; ++i) {
      int begin_index = i * worker_count + task_offset;
      for (int j = 0; j < AVX_LANES; ++j) {
        valid[j] = idx2prefix(suffix_len, prefix_len, begin_index + j, prefix + j * prefix_len, suffix + j, AVX_LANES);
      }
      for (int j = 0; j < suffix_len; ++j) {
        a[j] = _mm256_load_si256(reinterpret_cast<const __m256i *>(suffix + AVX_LANES * j));
        idx[j] = j + 1;
      }
      while (true) {
        for (bool v : valid) {
          if (v) callback();
        }
        bool Flag = false;
        for (int i(suffix_len - 2); i >= 0; i--)
          if (idx[i] < idx[i + 1]) {
            int x = i + 1;
            for (int j(i + 2); j < suffix_len; j++)
              if (idx[j] > idx[i] && idx[j] < idx[x]) x = j;
            Flag = true;
            std::swap(a[i], a[x]);
            std::swap(idx[i], idx[x]);
            int R = suffix_len - 1;
            for (int L(i + 1); L < R;) {
              std::swap(a[L], a[R]);
              std::swap(idx[L], idx[R]);
              L++, R--;
            }
            break;
          }
        if (!Flag) break;
      }
#ifdef PPERM_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
  }
};

REGISTER_PERM_ALGORITHM("lex_cpu_mpi_simd", LexCpuMpiSimd)

#else
#warning "Your processor does not support AVX2, SIMD algorithm will not be included"
#endif
