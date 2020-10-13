
#include <immintrin.h>
#include <pperm.hh>

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
