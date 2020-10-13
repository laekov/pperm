/*
 * Smart O(n!) recursion, but use SIMD & MPI
 * For SIMD, we use AVX2 in a way that a 256-bit register is used as 32 int8. So the max possible permutation length is 127.
 * For MPI, we simply use it to split tasks to each core.
 */

#include <immintrin.h>
#include <pperm.hh>

#if true
//#ifdef PPERM_MPI
#include <mpi.h>
#endif


//#ifdef __AVX2__
#if true

class SmartRecrCpuMpiSimd : public PermAlgorithm {
 protected:
  void setup_() override {
  }

  void generate_() override {
    __m256i a;
  }
};

REGISTER_PERM_ALGORITHM("smart_recr_cpu_mpi_simd", SmartRecrCpuMpiSimd)

#else
#warning "Your processor does not support AVX2, SIMD algorithm will not be included"
#endif
