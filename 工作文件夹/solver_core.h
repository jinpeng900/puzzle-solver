#ifndef SOLVER_CORE_H
#define SOLVER_CORE_H
// solver_core.h - Core solver with optional NN evaluation (C++17)

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <array>
#include <deque>
#include <functional>
#include <limits>
#include <unordered_map>
#include <cstdint>
#include <set>
#include <memory>

// ============================================================
// Data Model
// ============================================================

struct Cell {
    int value = 1;
    int  color()       const { return std::abs(value); }
    bool is_bomb()     const { return value < 0; }
    bool is_wildcard() const { return value == 0; }
};

struct Board {
    int N = 0;
    int level = 1;
    std::vector<std::vector<Cell>> grid;
    std::shared_ptr<std::vector<std::vector<int>>> drop_queue;
    std::vector<int> queue_ptr;

    explicit Board(int n = 0) : N(n), grid(n, std::vector<Cell>(n)),
        drop_queue(std::make_shared<std::vector<std::vector<int>>>()) {}

    Cell&       at(int r, int c)       { return grid[r][c]; }
    const Cell& at(int r, int c) const { return grid[r][c]; }
    bool in_bounds(int r, int c) const { return r >= 0 && r < N && c >= 0 && c < N; }

    Board preview(const std::vector<std::pair<int,int>>& path) const {
        Board next_b = *this;
        if (path.size() < 2) return next_b;
        auto& dq = *next_b.drop_queue;
        std::vector<std::vector<bool>> in_path(N, std::vector<bool>(N));
        for (auto p : path) in_path[p.first][p.second] = true;
        std::vector<std::vector<bool>> to_remove = in_path;
        if (level >= 4)
            for (auto [r,c] : path) {
                if (!at(r,c).is_bomb()) continue;
                for (int dr = -1; dr <= 1; ++dr)
                    for (int dc = -1; dc <= 1; ++dc) {
                        int nr = r+dr, nc = c+dc;
                        if (in_bounds(nr,nc) && !in_path[nr][nc]) to_remove[nr][nc] = true;
                    }
            }
        for (int c = 0; c < N; ++c) {
            std::vector<Cell> remaining;
            for (int r = 0; r < N; ++r) if (!to_remove[r][c]) remaining.push_back(at(r,c));
            int empty = N - (int)remaining.size();
            for (int i = 0; i < empty; ++i)
                next_b.at(i,c).value = dq[c][next_b.queue_ptr[c]++];
            for (int i = 0; i < (int)remaining.size(); ++i)
                next_b.at(empty+i,c) = remaining[i];
        }
        return next_b;
    }

    bool is_deadlocked() const {
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                int ac = at(r,c).color();
                if (c+1<N) { int c2 = at(r,c+1).color(); if (ac==c2||ac==0||c2==0) return false; }
                if (r+1<N) { int c2 = at(r+1,c).color(); if (ac==c2||ac==0||c2==0) return false; }
            }
        return true;
    }

    int count_adjacent_pairs() const {
        int cnt = 0;
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                int ac = at(r,c).color();
                if (c+1<N) { int c2 = at(r,c+1).color(); if (ac==c2||ac==0||c2==0) cnt++; }
                if (r+1<N) { int c2 = at(r+1,c).color(); if (ac==c2||ac==0||c2==0) cnt++; }
            }
        return cnt;
    }
};

// ============================================================
// 工具函数
// ============================================================

constexpr int DR[] = {-1, 1, 0, 0};
constexpr int DC[] = { 0, 0,-1, 1};

inline int path_score(int k) {
    double t = std::sqrt((double)k) - 1.0;
    return 10 * k + 18 * (int)(t * t);
}

inline int path_score(const Board& board, const std::vector<std::pair<int,int>>& path) {
    int k = (int)path.size(), s = path_score(k);
    std::vector<std::vector<bool>> in_path(board.N, std::vector<bool>(board.N));
    for (auto [r,c] : path) in_path[r][c] = true;
    std::vector<std::vector<bool>> exploded(board.N, std::vector<bool>(board.N));
    for (auto [r,c] : path) {
        if (!board.at(r,c).is_bomb()) continue;
        for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc) {
                int nr = r+dr, nc = c+dc;
                if (board.in_bounds(nr,nc) && !in_path[nr][nc] && !exploded[nr][nc]) {
                    exploded[nr][nc] = true;
                    s += 10;
                }
            }
    }
    return s;
}

inline bool compatible_color(int target, int color) {
    return target == 0 || color == 0 || target == color;
}

inline std::uint64_t board_hash(const Board& b) {
    std::uint64_t h = 1469598103934665603ULL;
    const std::uint64_t prime = 1099511628211ULL;
    h ^= (std::uint64_t)b.N; h *= prime;
    h ^= (std::uint64_t)b.level; h *= prime;
    for (int r = 0; r < b.N; ++r)
        for (int c = 0; c < b.N; ++c) {
            h ^= (std::uint64_t)(b.at(r,c).value + 17);
            h *= prime;
        }
    return h;
}

// ============================================================
// 参数系统（基于GA优化结果 best_stable_v2.txt）
// ============================================================

struct LevelParams {
    int dfs_limit, keep1, keep2, keep3, cache_limit;
    double mxc_w, dq_w, bh_w, s3_dq_w, s3_bh_w, surv_mul;
    int beam_w, beam_d, short_pen, surv_check;
    double beam_bonus, len_bonus;
    double tfast_ratio, tdead_ratio;
    int max_time_ms, fallback_len_thresh;
};

extern LevelParams g_params[6];
extern int g_cur_level;
extern int g_cur_step;

void init_best_params();

// ============================================================
// 阶段检测与参数调度
// ============================================================

enum class Phase : int { EARLY=0, MID=1, LATE=2, URGENT=3, SURVIVAL=4 };

struct PhaseMultiplier {
    double dfs_mul, keep_mul, beam_w_mul, beam_d_mul;
    double dq_mul, bh_mul, surv_mul, short_pen_mul;
};

PhaseMultiplier get_phase_multiplier(Phase ph, int level);
Phase detect_phase(const Board& board, int step);

// ============================================================
// 求解器核心
// ============================================================

class ImprovedSolver {
    mutable std::chrono::steady_clock::time_point _t0;
    mutable bool _tinit = false;
    void tstart() const { if(!_tinit){_t0=std::chrono::steady_clock::now();_tinit=true;} }
    long long telapsed() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now()-_t0).count();
    }
    mutable int step_count_ = 0;
    bool tfast()  const { return telapsed() > g_params[g_cur_level].max_time_ms * g_params[g_cur_level].tfast_ratio; }
    bool tdead()  const { return telapsed() > g_params[g_cur_level].max_time_ms * g_params[g_cur_level].tdead_ratio; }

    Phase current_phase_ = Phase::EARLY;
    double phase_dfs_mul_ = 1.0, phase_keep_mul_ = 1.0;
    double phase_beam_w_mul_ = 1.0, phase_beam_d_mul_ = 1.0;
    double phase_dq_mul_ = 1.0, phase_bh_mul_ = 1.0;
    double phase_surv_mul_ = 1.0, phase_sp_mul_ = 1.0;

    static inline int cur_cache_limit_ = 60000;
    static inline int cur_keep1_ = 200, cur_keep2_ = 100, cur_keep3_ = 20;
    static inline double mxc_weight_ = 15.0, dq_weight_ = 0.8, bh_weight_ = 0.02;
    static inline double step3_dq_weight_ = 0.55, step3_bh_weight_ = 0.025;
    static inline double surv_mul_ = 1.0;
    static inline int cur_beam_w_ = 8, cur_beam_d_ = 4;

    void apply_phase_params(int level, int step);
    int cache_limit() const { return tfast()?(int)(cur_cache_limit_*0.6):cur_cache_limit_; }
    int keep1() const { return tfast()?(int)(cur_keep1_*0.65):cur_keep1_; }
    int keep2() const { return tfast()?(int)(cur_keep2_*0.65):cur_keep2_; }
    int keep3() const { return tfast()?(int)(cur_keep3_*0.65):cur_keep3_; }

    mutable int board_N_ = 6;
    mutable const Board* current_board_ptr_ = nullptr;

    static double board_heuristic(const Board& b);
    static double drop_quality(const Board& b);
    static int upper_bound_reachable(const Board& b, const std::vector<std::pair<int,int>>& path, int target);
    static int exact_best_one_step_score(const Board& b, int max_nodes);
    static std::pair<int,std::vector<std::pair<int,int>>>
    exact_best_with_path(const Board& b, int max_nodes);
    int survival_steps(const Board& b, int max_check) const;
    double beam_evaluate(const Board& start_b, int beam_w, int beam_d) const;
    static std::vector<std::pair<int,int>> fallback(const Board& b);

public:
    std::vector<std::pair<int,int>> solve(const Board& board);
};

// ============================================================
// 神经网络评估器
// ============================================================
constexpr int NN_NUM_F = 52;

struct NNFeatures {
    double f[NN_NUM_F];
    NNFeatures() { for(int i=0;i<NN_NUM_F;++i) f[i]=0.0; }
};

NNFeatures nn_extract_features(const Board& b);

class NNEvaluator {
    int L;
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<double>> b;
    std::vector<bool> relu;
    double scale;
public:
    NNEvaluator();
    double predict(const NNFeatures& bf);
    bool load(const char* path);
    bool loaded() const { return L > 0; }
};

extern NNEvaluator g_nn_eval;

#endif // SOLVER_CORE_H
