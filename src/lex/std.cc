#include <algorithm>
#include <pperm.hh>

class Std : public PermAlgorithm<Std> {

 GENERATE_CONSTRUCTOR(Std)

 private:
  int *a;

 protected:
  void setup_() override {
    a = new int[n];
    for (int i = 0; i < n; ++i) {
      a[i] = i;
    }
  }

 public:
  template <typename F>
  void do_generate_(F&& callback) {
    int s = 0, is_end = 0;
    while (!is_end) {
      s += 1;
      is_end = 1;
      for (int i = 0; i + 1 < n; ++i) {
        if (a[i] < a[i + 1]) {
          is_end = 0;
          break;
        }
      }
      std::next_permutation(a, a + n);
    }
  }
};

