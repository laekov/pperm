#include <algorithm>
#include <iostream>
#include <pperm.hh>

class LexCPUSimple : public PermAlgorithm {
 private:
  int *a;

 protected:
  virtual void setup_() {
    a = new int[n];
    for (int i = 0; i < n; ++i) a[i] = i;
  }
  virtual void generate_() {
    int s = 0;
    while (1) {
      for (int i(0); i < n; i++) printf("%d ", a[i]);
      puts("");
      bool Flag = false;
      for (int i(n - 2); i >= 0; i--)
        if (a[i] < a[i + 1]) {
          int x = i + 1;
          for (int j(i + 2); j < n; j++)
            if (a[j] > a[i] && a[j] < a[x]) x = j;
          Flag = true;
          std::swap(a[i], a[x]);
          int R = n - 1;
          for (int L(i + 1); L < R;) std::swap(a[L++], a[R--]);
          break;
        }
      if (!Flag) break;
    }
  }
};

REGISTER_PERM_ALGORITHM("lex_cpu_simple", LexCPUSimple)