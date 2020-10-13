/*
 * Smart O(n!) recursion.
 * On i-th step, swap the i-th element with the rest elements in the array.
 * Should be faster than smart, but still slow because of too many recursion.
 */
#include <algorithm>
#include <pperm.hh>

class SmartRecrCpuSimple : public PermAlgorithm {
 private:
  int *a, s;

  void DFS(int cur) {
    if (cur == n) {
      s += 1;
      return;
    }
    for (int i = cur; i < n; ++i) {
      if (i == cur) {
        DFS(cur + 1);
        continue;
      }
      std::swap(a[i], a[cur]);
      DFS(cur + 1);
      std::swap(a[i], a[cur]);
    }
  }

 protected:
  void setup_() override {
    a = new int[n];
    for (int i = 0; i < n; ++i) {
      a[i] = i;
    }
  }

  void generate_() override {
    s = 0;
    DFS(0);
  }
};

REGISTER_PERM_ALGORITHM("smart_recr_cpu_simple", SmartRecrCpuSimple)

/*
 * Use manually implemented stack to run the O(n!) algorithm
 * Expected to be fast
 */

class SmartRecrSimStackCpuSimple : public PermAlgorithm {
 private:
  int *a, *stack, top, s;

 protected:
  void setup_() override {
    a = new int[n];
    for (int i = 0; i < n; ++i) {
      a[i] = i;
    }
    stack = new int[n];
  }

  void generate_() override {
    s = 0;
    stack[top = 0] = 0;
    while (top >= 0) {
      if (top == n) {
        s += 1;
        /* TODO: do some callback
        for (int i = 0; i < n; ++i) {
                printf("%d ", a[i]);
        }
        putchar(10);
        */
      }
      if (top == n || stack[top] == n) {
        --top;
        if (top >= 0) {
          if (++stack[top] < n) {
            if ((n - top) & 1) {
              std::swap(a[top], a[n - 1]);
            } else {
              std::swap(a[top], a[stack[top]]);
            }
          }
        }
        continue;
      }
      if (top + 1 < n) {
        stack[top + 1] = top + 1;
      }
      ++top;
    }
  }
};

REGISTER_PERM_ALGORITHM("smart_recr_simstack_cpu_simple", SmartRecrSimStackCpuSimple)
