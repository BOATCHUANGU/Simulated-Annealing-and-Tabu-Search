#include <bits/stdc++.h>
using namespace std;

using State = vector<int>;
class TS {
  public:
    // 需要自己实现
    double evaluate(const State &s) {
        // TODO
        return 0;
    }

    // 生成邻居：随机交换两个位置
    State neighbor(State s, mt19937 &rng) {
        if (s.size() < 2)
            return s;

        uniform_int_distribution<int> dist(0, s.size() - 1);
        int i = dist(rng), j = dist(rng);

        while (i == j)
            j = dist(rng);

        swap(s[i], s[j]);
        return s;
    }

    State TabuSearch(State cur, int maxIter = 10000, int tabuTenure = 20) {
        mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

        State best = cur;
        double bestVal = evaluate(best);

        // 记录方案及其禁忌到期时间
        map<State, int> tabu;

        for (int iter = 0; iter < maxIter; iter++) {
            State chosen;
            double chosenVal = 1e100;
            bool found = false;

            // 每轮生成一批候选邻居
            for (int k = 0; k < 50; k++) {
                State nxt = neighbor(cur, rng);
                double val = evaluate(nxt);

                bool forbidden = tabu.count(nxt) && tabu[nxt] > iter;

                // 特赦准则：即使被禁忌，若刷新历史最佳仍可接受
                if (forbidden && val >= bestVal)
                    continue;

                if (!found || val < chosenVal) {
                    chosen = nxt;
                    chosenVal = val;
                    found = true;
                }
            }

            if (!found)
                continue;

            cur = chosen;
            tabu[cur] = iter + tabuTenure;

            if (chosenVal < bestVal) {
                best = cur;
                bestVal = chosenVal;
            }

            // 可选：清理已经过期的禁忌记录
            for (auto it = tabu.begin(); it != tabu.end();) {
                if (it->second <= iter)
                    it = tabu.erase(it);
                else
                    ++it;
            }
        }

        return best;
    }
};