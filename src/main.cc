#include "pperm.hh"


int main(int argc, char* args[]) {
	// TODO: Use getopt here
	int n = argc >= 2 ? atoi(args[1]) : 10;
	double n_compute = n;
	for (int i = 2; i <= n; ++i) {
		n_compute *= i;
	}
	
	for (int i = 2; i < argc; ++i) {
		std::string algo_name(args[i]);
		auto algo = PermAlgorithm::get(algo_name);
		if (algo == nullptr) {
			fprintf(stderr, "No such algorithm: %s\n", algo_name.c_str());
			continue;
		}

		fprintf(stderr, "Setting up %s\n", algo_name.c_str());
		algo->setup(n);

		fprintf(stderr, "Warming up\n");
		algo->warmup();

		fprintf(stderr, "Testing\n");
		auto res = algo->benchmark();
		printf("Algorithm %s, n = %d, mean = %.3lf ms, max = %.3lf, "
				"GEPs = %.3lf\n",
				algo_name.c_str(), n, res.first * 1e3, res.second * 1e3,
				n_compute / res.first * 1e-9);
	}
}
