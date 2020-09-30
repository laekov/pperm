#include <pperm.hh>
#include <timer.hh>


std::unordered_map<std::string, PermAlgorithm*>* \
		PermAlgorithm::algorithms = 0;


void PermAlgorithm::warmup() {
	for (int i = 0; i < 10; ++i) {
		this->generate_();
	}
}


std::pair<double, double> PermAlgorithm::benchmark() {
	static const int n_tests = 16;
	double max_time = 0.;
	double tot_time = 0.;
	for (int i = 0; i < 10; ++i) {
		timestamp(perm_begin);
		this->generate_();
		timestamp(perm_end);
		auto dur = getDuration(perm_begin, perm_end);
		tot_time += dur;
		if (dur > max_time) {
			max_time = dur;
		}
	}
	return std::pair<double, double>(tot_time / n_tests, max_time);
}
