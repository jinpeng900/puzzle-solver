#ifndef NN_EVALUATOR_H
#define NN_EVALUATOR_H

#include "nn_model.h"

// ============================================================
// NN评估器：混合NN评分与启发式评分
// 用法：
//   NNEvaluator eval;
//   eval.load_weights("nn_weights.bin");
//   double score = eval.evaluate(board);
// ============================================================
template<typename BoardT>
class NNEvaluator {
    NNModel model_;
    double nn_weight_ = 0.5;   // NN 权重（0=纯启发式, 1=纯NN）
    bool enabled_ = false;
    std::string weights_path_ = "nn_solver/weights/nn_weights.bin";

    // 启发式评估（与 1.cpp 的 board_heuristic 相同）
    static double heuristic(const BoardT& b) {
        if (b.is_deadlocked()) return -1e5;
        int N = b.N, cp = 0, sp = 0, wc = 0, bb = 0, cnt[6] = {}, mxc = 0;
        constexpr int DR[] = {-1,1,0,0};
        constexpr int DC[] = {0,0,-1,1};
        std::vector<std::vector<bool>> vis(N, std::vector<bool>(N));

        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                int co = std::abs(b.at(r, c).value);
                if (b.at(r, c).value == 0) wc++;
                if (b.at(r, c).value < 0) bb++;
                cnt[std::min(co, 5)]++;
                if (c+1 < N) { int c2 = std::abs(b.at(r, c+1).value); if (bf_compat(co, c2)) { cp++; if (co == c2 && co != 0) sp++; } }
                if (r+1 < N) { int c2 = std::abs(b.at(r+1, c).value); if (bf_compat(co, c2)) { cp++; if (co == c2 && co != 0) sp++; } }
                if (!vis[r][c]) {
                    int sz = 0;
                    std::deque<std::pair<int, int>> q;
                    q.push_back({r, c}); vis[r][c] = true;
                    while (!q.empty()) {
                        auto p = q.front(); q.pop_front(); sz++;
                        int cr = p.first, cc = p.second;
                        int cco = std::abs(b.at(cr, cc).value);
                        for (int d = 0; d < 4; ++d) {
                            int nr = cr + DR[d], nc = cc + DC[d];
                            if (!b.in_bounds(nr, nc) || vis[nr][nc]) continue;
                            if (!bf_compat(cco, std::abs(b.at(nr, nc).value))) continue;
                            vis[nr][nc] = true; q.push_back(std::make_pair(nr, nc));
                        }
                    }
                    mxc = std::max(mxc, sz);
                }
            }
        double ent = 0;
        int tc = N * N - wc;
        if (tc > 0)
            for (int i = 1; i <= 5; ++i)
                if (cnt[i] > 0) { double p = (double)cnt[i] / tc; ent -= p * std::log(p); }
        double bb_w = (b.level == 4 || b.level == 5) ? 22.0 : 15.0;
        double mxc_w = (b.level <= 2) ? 25.0 : 15.0;
        return cp * 6.0 + sp * 10.0 + wc * 35.0 + bb * bb_w + mxc * mxc_w - ent * 12.0;
    }

    static bool bf_compat(int a, int b) { return a == 0 || b == 0 || a == b; }

public:
    void enable(bool flg) { enabled_ = flg; }
    bool is_enabled() const { return enabled_; }

    double evaluate(const BoardT& b) {
        double h = heuristic(b);
        if (!enabled_) return h;
        double nn = model_.predict(extract_features(b));
        return nn_weight_ * nn + (1.0 - nn_weight_) * h;
    }

    double pure_nn(const BoardT& b) {
        return model_.predict(extract_features(b));
    }

    NNModel& model() { return model_; }
    void set_nn_weight(double w) { nn_weight_ = w; }
    double nn_weight() const { return nn_weight_; }

    bool save_weights(const std::string& path = "") {
        return model_.save(path.empty() ? weights_path_ : path);
    }

    bool load_weights(const std::string& path = "") {
        return model_.load(path.empty() ? weights_path_ : path);
    }
};

#endif
