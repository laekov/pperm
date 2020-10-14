#ifndef PPERM_HH
#define PPERM_HH

#include <string>
#include <map>
#include <vector>

struct BenchmarkResult {
  double mean, stddev, min, max;
};

class PermAlgorithm {
private:
  static std::map<std::string, PermAlgorithm*> *algorithms;

public:
  template<class A>
  static bool add(std::string name) {
    if (!algorithms) {
      algorithms = new std::map<std::string, PermAlgorithm*>;
    }
    (*algorithms)[name] = new A();
    return true;
  }

  static PermAlgorithm* get(std::string name) {
    auto algo = algorithms->find(name);
    if (algo == algorithms->end()) {
      return nullptr;
    }
    return algo->second;
  }
  static std::vector<std::string> getNames() {
    std::vector<std::string> names;
    names.reserve(algorithms->size());
    for (auto iter = algorithms->cbegin(); iter != algorithms->cend(); iter++) {
      names.emplace_back(iter->first);
    }
    return names;
  }
  
public:
  PermAlgorithm() {};

  inline void setup(int n_) {
    this->n = n_;
    this->setup_();
  }

  void warmup();
  BenchmarkResult benchmark(int n_tests);

protected:
  int n;

  virtual void setup_() {}
  virtual void generate_() = 0;
};


bool registerAlgorithm(std::string, PermAlgorithm*);


#define REGISTER_PERM_ALGORITHM(__NAME__, __CLASS__) \
  bool __register_successful_##__CLASS__##__ = \
      PermAlgorithm::add<__CLASS__>(__NAME__);

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

__device__ __forceinline__ bool idx2prefix(int n, int prefix_len, int task_idx, int* a) {
#else
inline bool idx2prefix(int n, int prefix_len, int task_idx, int* a) {
#endif
  int perm_cnt = 1;
  for (int i = n + 1; i <= n + prefix_len; ++i) {
    perm_cnt *= i;
  }

  if (task_idx >= perm_cnt) {
    return false;
  }
  unsigned long taken = 0;
  for (int i = n + prefix_len; i > n; --i) {
    perm_cnt /= i;
    int count_smaller = task_idx / perm_cnt, j;
    task_idx %= perm_cnt;
    for (j = 0; count_smaller || (taken & (1ul << j)); ++j) {
      if (!(taken & (1ul << j))) {
        count_smaller -= 1;
      }
    }
    taken |= (1ul << j);
    a[i] = j + 1;
  }
  for (int i = 0, j = 0; i < n; ++i) {
    for (; (taken & (1ul << j)); ++j)
      ;
    a[i] = j + 1;
    taken |= (1ul << j);
  }
  return true;
}

#endif  // PPERM_HH
