/*
 * Naive O(n! * n) recursion.
 * Should be slow.
 */
#include <pperm.hh>

class NaiveRecrCpuSimple : public PermAlgorithm<NaiveRecrCpuSimple> {

 GENERATE_CONSTRUCTOR(NaiveRecrCpuSimple)

 private:
  int *a, s;

  template <typename F>
  void DFS(int cur, int occupied, F&& callback) {
    if (cur == n) {
      callback();
      s += 1;
      return;
    }
    for (int i = 0; i < n; ++i) {
      if (occupied & (1 << i)) {
        continue;
      }
      a[cur] = i;
      DFS(cur + 1, occupied | (1 << i), std::forward<F>(callback));
    }
  }

 protected:
  void setup_() override { a = new int[n]; }

 public:
  template <typename F>
  void do_generate_(F&& callback) {
    s = 0;
    DFS(0, 0, std::forward<F>(callback));
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

class NaiveRecrSimStackCpuSimple : public PermAlgorithm<NaiveRecrSimStackCpuSimple> {

 GENERATE_CONSTRUCTOR(NaiveRecrSimStackCpuSimple)

 private:
  int *stack, top, s;

 protected:
  void setup_() override { stack = new int[n * n]; }

 public:
  template <typename F>
  void do_generate_(F&& callback)  {
    s = 0;
    stack[top = 0] = 0;
    while (top >= 0) {
      int occ = stack[top--];
      if (occ == ((1 << n) - 1)) {
        s += 1;
        callback();
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
