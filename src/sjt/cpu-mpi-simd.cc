#include <immintrin.h>
#include <pperm.hh>

#ifdef PPERM_MPI
#include <mpi.h>
#endif

#ifdef PPERM_AVX2

class SJTCpuMpiSimd : public PermAlgorithm<SJTCpuMpiSimd> {
  GENERATE_CONSTRUCTOR(SJTCpuMpiSimd)

  PPERM_SIMD_INIT

 private:
  int *perm, *inv;
  bool *dir;

 protected:
  void setup_() override {
    setup_common_();
    perm = new int[suffix_len];
    inv = new int[suffix_len];
    dir = new bool[suffix_len];
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
      }
      for (int j = 0; j < suffix_len; ++j) {
        perm[j] = j;
        inv[j] = j;
        dir[j] = false;
      }
      while (true) {
        for (bool v : valid) {
          if (v) callback();
        }
        int top = -1, nxt;
        for (int i(suffix_len - 1); i >= 0; i--) {
          nxt = inv[i] + (dir[i] ? 1 : -1);
          if (nxt >= 0 && nxt < suffix_len) {
            if (perm[nxt] < i) {
              top = i;
              break;
            }
          }
        }
        if (top == -1) {
          break;
        }
        std::swap(a[perm[top]], a[perm[nxt]]);
        inv[perm[nxt]] = inv[top];
        perm[inv[top]] = perm[nxt];
        inv[top] = nxt;
        perm[nxt] = top;
        for (int i(top + 1); i < suffix_len; i++) dir[i] ^= 1;
      }
#ifdef PPERM_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
  }
};

REGISTER_PERM_ALGORITHM("sjt_cpu_mpi_simd", SJTCpuMpiSimd)

#else
#warning "Your processor does not support AVX2, SIMD algorithm will not be included"
#endif
