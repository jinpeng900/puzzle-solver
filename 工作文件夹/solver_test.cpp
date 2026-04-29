// ============================================================
// 本地测试版：LocalJudge + 求解器
// 编译: g++ -std=c++17 -O3 -o solver_test solver_core.cpp solver_test.cpp
// 用法: solver_test.exe [seed1] [seed2] ...
// ============================================================

#include "solver_core.h"
#include <cstdlib>

class LocalJudge {
    int level_;
    int N_;
    Board board_;
    int step_ = 0;
    int score_ = 0;
    int invalid_streak_ = 0;
    bool done_ = false;
    static constexpr int MAX_STEPS = 50;

    static int gen_block(std::mt19937& rng, int level) {
        if (level<=2) return (rng()%5)+1;
        if (level==3) return ((rng()%100)<15)?0:(rng()%5)+1;
        if (level==4) { int c=(rng()%5)+1; return ((rng()%100)<10)?-c:c; }
        if ((rng()%100)<15) return 0;
        int base=(rng()%5)+1; return ((rng()%100)<10)?-base:base;
    }

public:
    LocalJudge(int level, int seed, int N) : level_(level), N_(N) {
        board_ = Board(N); board_.level = level;
        board_.drop_queue = std::make_shared<std::vector<std::vector<int>>>();
        board_.drop_queue->assign(N, std::vector<int>(1000));
        board_.queue_ptr.assign(N, 0);

        std::mt19937 rng(seed);
        for (int c = 0; c < N; ++c)
            for (int i = 0; i < 1000; ++i)
                (*board_.drop_queue)[c][i] = gen_block(rng, level);

        std::mt19937 rng_board(seed ^ 0x9E3779B9);
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c)
                board_.at(r, c).value = gen_block(rng_board, level);
    }

    const Board& board() const { return board_; }
    int score() const { return score_; }
    int step() const { return step_; }
    bool done() const { return done_; }

    bool validate(const std::vector<std::pair<int,int>>& path, std::string& reason) const {
        if (path.size() < 2) { reason = "len<2"; return false; }
        std::vector<std::vector<bool>> used(N_, std::vector<bool>(N_));
        int anchor = 0;
        for (size_t i = 0; i < path.size(); ++i) {
            auto [r, c] = path[i];
            if (r < 0 || r >= N_ || c < 0 || c >= N_) { reason = "OOB"; return false; }
            if (used[r][c]) { reason = "dup"; return false; }
            used[r][c] = true;
            int color = board_.at(r, c).color();
            if (color != 0) {
                if (anchor == 0) anchor = color;
                else if (anchor != color) { reason = "color"; return false; }
            }
            if (i > 0) {
                auto [pr, pc] = path[i-1];
                if (std::abs(pr - r) + std::abs(pc - c) != 1) { reason = "4conn"; return false; }
            }
        }
        return true;
    }

    bool play(const std::vector<std::pair<int,int>>& path) {
        if (done_) return false;
        if (step_ >= MAX_STEPS) { done_ = true; return false; }

        std::string reason;
        bool valid = validate(path, reason);

        if (!valid) {
            invalid_streak_++; step_++;
            if (invalid_streak_ >= 3) done_ = true;
            return !done_;
        }

        invalid_streak_ = 0;
        int gained = path_score(path.size());

        if (level_ >= 4) {
            std::vector<std::vector<bool>> in_path(N_, std::vector<bool>(N_));
            std::vector<std::vector<bool>> exploded(N_, std::vector<bool>(N_));
            for (auto [r, c] : path) in_path[r][c] = true;
            for (auto [r, c] : path) {
                if (!board_.at(r, c).is_bomb()) continue;
                for (int dr = -1; dr <= 1; ++dr)
                    for (int dc = -1; dc <= 1; ++dc) {
                        int nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < N_ && nc >= 0 && nc < N_
                            && !in_path[nr][nc] && !exploded[nr][nc]) {
                            exploded[nr][nc] = true; gained += 10;
                        }
                    }
            }
        }

        score_ += gained; step_++;

        std::vector<std::vector<bool>> to_remove(N_, std::vector<bool>(N_));
        for (auto [r, c] : path) to_remove[r][c] = true;
        if (level_ >= 4) {
            for (auto [r, c] : path) {
                if (!board_.at(r, c).is_bomb()) continue;
                for (int dr = -1; dr <= 1; ++dr)
                    for (int dc = -1; dc <= 1; ++dc) {
                        int nr = r + dr, nc = c + dc;
                        if (nr >= 0 && nr < N_ && nc >= 0 && nc < N_ && !to_remove[nr][nc])
                            to_remove[nr][nc] = true;
                    }
            }
        }

        for (int c = 0; c < N_; ++c) {
            std::vector<Cell> remaining;
            for (int r = 0; r < N_; ++r)
                if (!to_remove[r][c]) remaining.push_back(board_.at(r, c));
            int empty = N_ - (int)remaining.size();
            for (int i = 0; i < empty; ++i)
                board_.at(i, c).value = (*board_.drop_queue)[c][board_.queue_ptr[c]++];
            for (int i = 0; i < (int)remaining.size(); ++i)
                board_.at(empty + i, c) = remaining[i];
        }

        if (board_.is_deadlocked()) done_ = true;
        if (step_ >= MAX_STEPS) done_ = true;
        return !done_;
    }
};

struct LevelConfig { int level; int N; const char* name; };
static const LevelConfig LEVELS[] = {
    {1, 10, "L1(10x10纯色)"}, {2, 10, "L2(10x10纯色)"},
    {3, 10, "L3(10x10万能)"}, {4, 10, "L4(10x10炸弹)"},
    {5, 12, "L5(12x12混合)"},
};

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    init_best_params();

    // 尝试加载NN权重
    if (g_nn_eval.load("nn_weights.bin"))
        std::printf("[NN] Weights loaded, using NN-enhanced evaluation\n");

    ImprovedSolver solver;

    std::vector<int> seeds;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) seeds.push_back(std::atoi(argv[i]));
    } else {
        seeds = {42, 114514, 888888, 123456, 654321};
    }
    std::printf("=== Solver Multi-Seed Test ===\n");
    std::printf("Seeds: ");
    for (int s : seeds) std::printf("%d ", s);
    std::printf("\n\n");

    double grand_total = 0;
    int total_tests = 0;

    for (int seed : seeds) {
        std::printf("--- Seed %d ---\n", seed);
        int seed_total = 0;

        for (auto& lc : LEVELS) {
            LocalJudge judge(lc.level, seed, lc.N);
            while (!judge.done()) {
                auto path = solver.solve(judge.board());
                if (path.size() < 2) path = {{0,0},{0,1}};
                judge.play(path);
            }
            std::printf("  %s: %5d pts (%d steps)\n",
                lc.name, judge.score(), judge.step());
            seed_total += judge.score();
        }
        std::printf("  Seed total: %d\n", seed_total);
        grand_total += seed_total;
        total_tests++;
    }

    double avg = total_tests > 0 ? grand_total / total_tests : 0;
    std::printf("\n=== Average Total Score: %.1f ===\n", avg);
    return 0;
}
