/*
 * Naive O(n! * n) recursion.
 * Should be slow.
 */
#include <pperm.hh>

class NaiveRecrCpuSimple : public PermAlgorithm {
 private:
  int *a, s;

  void DFS(int cur, int occupied) {
    if (cur == n) {
      s += 1;
      return;
    }
    for (int i = 0; i < n; ++i) {
      if (occupied & (1 << i)) {
        continue;
      }
      a[cur] = i;
      DFS(cur + 1, occupied | (1 << i));
    }
  }

 protected:
  void setup_() override { a = new int[n]; }

  void generate_() override {
    s = 0;
    DFS(0, 0);
  }
};

REGISTER_PERM_ALGORITHM("naive_recr_cpu_simple", NaiveRecrCpuSimple)

/*
 * Naive O(n! * n) recursion with manually implemented stack.
 * Expected to save great amount of function call overhead.
 * But it cannot really generate the permutation.
 */
#include <algorithm>
#include <pperm.hh>

class NaiveRecrSimStackCpuSimple : public PermAlgorithm {
 private:
  int *stack, top, s;

 protected:
  void setup_() override { stack = new int[n * n]; }

  void generate_() override {
    s = 0;
    stack[top = 0] = 0;
    while (top >= 0) {
      int occ = stack[top--];
      if (occ == ((1 << n) - 1)) {
        s += 1;
        continue;
      }
      for (int i = 0; i < n; ++i) {
        if (occ & (1 << i)) {
          continue;
        }
        stack[++top] = occ | (1 << i);
      }
    }
  }
};

REGISTER_PERM_ALGORITHM("naive_recr_simstack_cpu_simple", NaiveRecrSimStackCpuSimple)
