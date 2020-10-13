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

#endif  // PPERM_HH
