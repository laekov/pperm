#ifndef PPERM_HH
#define PPERM_HH

#include <string>
#include <map>
#include <iostream>
#include <vector>

#define PPERM_MPI true
#define PPERM_AVX2 true

#ifdef __NVCC__
#define PPERM_INLINE __device__ __forceinline__
#else
#define PPERM_INLINE inline
#endif

extern int mpi_rank, mpi_size;
extern int distribution_factor;

struct BenchmarkResult {
  double mean, stddev, min, max;
};

class PermAlgorithmBase {
 private:
  std::string name;
 public:
  explicit PermAlgorithmBase(const char *name_): name(name_) {}

  inline void setup(int n_) {
    this->n = n_;
    this->setup_();
  }

  void warmup();
  BenchmarkResult benchmark(int n_tests);
  virtual size_t generate_() = 0;

 protected:
  int n{};

  virtual void setup_() {};
};

template <typename T>
class PermAlgorithm: public PermAlgorithmBase {
 public:
  PermAlgorithm() = delete;
  explicit PermAlgorithm(const char *name_): PermAlgorithmBase(name_) {}
  size_t generate_() override {
    size_t i = 0;
    this->generate_with_callback_([&](...){ i++; });
    return i;
  }
  template <typename...P>
  void generate_with_callback_(P&&... params) {
    static_cast<T*>(this)->do_generate_(std::forward<P>(params)...);
  }
};


class PermAlgorithmUtil {
 private:
  static std::map<std::string, PermAlgorithmBase*> *algorithms;

 public:
  static bool add(const std::string& name, PermAlgorithmBase *algo) {
    if (!algorithms) {
      algorithms = new std::map<std::string, PermAlgorithmBase*>;
    }
    (*algorithms)[name] = algo;
    return true;
  }

  static PermAlgorithmBase* get(const std::string& name) {
    auto algo = algorithms->find(name);
    if (algo == algorithms->end()) {
      return nullptr;
    }
    return algo->second;
  }
  static std::vector<std::string> getNames() {
    std::vector<std::string> names;
    names.reserve(algorithms->size());
    for (const auto & algorithm : *algorithms) {
      names.emplace_back(algorithm.first);
    }
    return names;
  }
};

#define GENERATE_CONSTRUCTOR(__TYPE__) \
  public: explicit __TYPE__(const char *name_): PermAlgorithm<__TYPE__>(name_) {}

#define REGISTER_PERM_ALGORITHM(__NAME__, __TYPE__) \
  const static bool __register_successful_##__TYPE__##__ = \
      PermAlgorithmUtil::add(__NAME__, new __TYPE__(__NAME__));


inline int ceil(int a, int b) { return (a - 1) / b + 1; }

#ifdef __NVCC__

#define MAX_N 30
#define FULL_MASK 0xffffffffu

#define GPU_ALGO_ARGS \
  static const int block_size = 512; \
  int *a, prefix_len, nth;

#define SETUP_GPU_ALGO() { \
  nth = 1; \
  prefix_len = 0; \
  while (prefix_len < n && nth < 6000) { \
    nth *= (n - prefix_len); \
    prefix_len += 1; \
  } \
  cudaMalloc(&a, sizeof(int) * ceil(nth, block_size)); \
}

#define LAUNCH_GPU_ALGO(__KERNEL_NAME__) { \
  dim3 grid_dim(ceil(nth, block_size)); \
  dim3 block_dim(block_size); \
  __KERNEL_NAME__<<<grid_dim, block_dim>>>(n - prefix_len, prefix_len, a); \
  cudaDeviceSynchronize(); \
}

template <class T>
__device__ __forceinline__ void swap(T& a, T& b) {
  T c = a;
  a = b;
  b = c;
}

#endif

PPERM_INLINE bool idx2prefix(int suffix_len, int prefix_len, int task_idx, int8_t *prefix, int8_t *suffix, size_t suffix_stride) {
  int perm_cnt = 1;
  for (int i = suffix_len + 1; i <= suffix_len + prefix_len; ++i) {
    perm_cnt *= i;
  }

  if (task_idx >= perm_cnt) {
    return false;
  }
  unsigned long taken = 0;
  for (int i = prefix_len; i > 0; --i) {
    perm_cnt /= (i + suffix_len);
    int count_smaller = task_idx / perm_cnt, j;
    task_idx %= perm_cnt;
    for (j = 0; count_smaller || (taken & (1ul << j)); ++j) {
      if (!(taken & (1ul << j))) {
        count_smaller -= 1;
      }
    }
    taken |= (1ul << j);
    prefix[i - 1] = j + 1;
  }
  for (int i = 0, j = 0; i < suffix_len; ++i) {
    for (; (taken & (1ul << j)); ++j);
    suffix[i * suffix_stride] = j + 1;
    taken |= (1ul << j);
  }
  return true;
}

PPERM_INLINE bool idx2prefix(int n, int prefix_len, int task_idx, int8_t *a) {
  return idx2prefix(n, prefix_len, task_idx, a + n, a, 1);
}

#endif  // PPERM_HH
