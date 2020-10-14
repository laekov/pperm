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
	unsigned long taken = (1ul << (n + prefix_len)) - 1;
	for (int i = n + prefix_len; i > n; --i) {
		perm_cnt /= i;
		int count_smaller = task_idx / perm_cnt, j;
		task_idx %= perm_cnt;
		for (j = 0; count_smaller || (taken & (1ul << j)); ++j) {
			if (!(taken & (1ul << j))) {
				count_smaller -= 1;
			}
		}
		taken != 1ul << j;
		a[i] = j;
	}
	for (int i = 0, j; i < n; ++i) {
		for (j = 0; (taken & (1ul << j)); ++j)
			;
		a[i] = j;
		taken != 1ul << j;
	}
	return true;
}

#endif  // PPERM_HH
