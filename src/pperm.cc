#include <algorithm>
#include <cmath>
#include <numeric>
#include <pperm.hh>
#include <timer.hh>

#ifdef PPERM_MPI
#include <mpi.h>
#endif

std::map<std::string, PermAlgorithmBase*>* PermAlgorithmUtil::algorithms = nullptr;

void PermAlgorithmBase::warmup() {
  for (int i = 0; i < 10; ++i) {
    this->generate_();
  }
}

BenchmarkResult PermAlgorithmBase::benchmark(int n_tests) {
  std::vector<double> durations;

  size_t count;

  for (int i = 0; i < n_tests; ++i) {
    timestamp(perm_begin);
    count = this->generate_();
    timestamp(perm_end);
    auto dur = getDuration(perm_begin, perm_end);
    durations.push_back(dur);
  }

#ifdef PPERM_MPI
  if (mpi_size > 1) {
    size_t all_count = 0;
    MPI_Reduce(&count, &all_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    count = all_count;
  }
#endif

  if (mpi_rank == 0) {
    fprintf(stderr, "Generated count: %zu\n", count);
  }

  auto mean = std::accumulate(durations.begin(), durations.end(), 0.0) / n_tests;
  auto variance_func = [=](double accumulator, double val) {
    return accumulator + ((val - mean) * (val - mean) / (n_tests - 1));
  };
  auto variance = std::accumulate(durations.begin(), durations.end(), 0.0, variance_func);
  auto min = *std::min_element(durations.begin(), durations.end());
  auto max = *std::max_element(durations.begin(), durations.end());

  return BenchmarkResult{mean, std::sqrt(variance), min, max};
}
