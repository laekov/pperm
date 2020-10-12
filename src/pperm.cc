#include <pperm.hh>
#include <timer.hh>

#include <algorithm>
#include <numeric>
#include <cmath>

std::unordered_map<std::string, PermAlgorithm*>* \
		PermAlgorithm::algorithms = nullptr;


void PermAlgorithm::warmup() {
	for (int i = 0; i < 10; ++i) {
		this->generate_();
	}
}


BenchmarkResult PermAlgorithm::benchmark(int n_tests) {

	std::vector<double> durations;

	for (int i = 0; i < n_tests; ++i) {
		timestamp(perm_begin);
		this->generate_();
		timestamp(perm_end);
		auto dur = getDuration(perm_begin, perm_end);
		durations.push_back(dur);
	}

	auto mean = std::accumulate(durations.begin(), durations.end(), 0.0) / n_tests;
	auto variance_func = [=](double accumulator, double val) {
		return accumulator + ((val - mean)*(val - mean) / (n_tests - 1));
	};
	auto variance = std::accumulate(durations.begin(), durations.end(), 0.0, variance_func);
	auto min = *std::min_element(durations.begin(), durations.end());
	auto max = *std::max_element(durations.begin(), durations.end());

	return BenchmarkResult{mean, std::sqrt(variance), min, max};
}
