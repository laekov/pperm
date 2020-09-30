#ifndef PPERM_HH
#define PPERM_HH

#include <string>
#include <unordered_map>

class PermAlgorithm {

public:
	PermAlgorithm() {};

	inline void setup(int n_) {
		this->n = n_;
		this->setup_();
	}

	void warmup();
	std::pair<double, double> benchmark();

protected:
	int n;

	virtual void setup_() {}
	virtual void generate_() = 0;
};

class PermAlgorithmRegistry {
private:
	static std::unordered_map<std::string, PermAlgorithm*> algorithms;

public:
	static bool add(std::string name, PermAlgorithm* algo) {
		algorithms[name] = algo;
		return true;
	}

	static PermAlgorithm* get(std::string name) {
		auto algo = algorithms.find(name);
		if (algo == algorithms.end()) {
			return nullptr;
		}
		return algo->second;
	}
};


bool registerAlgorithm(std::string, PermAlgorithm*);


#define REGISTER_PERM_ALGORITHM(__NAME__, __CLASS__) \
	bool __register_successful__ = PermAlgorithmRegistry::add(__NAME__, \
			(PermAlgorithm*)new __CLASS__);

#endif  // PPERM_HH
