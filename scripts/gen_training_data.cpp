/**
 * gen_training_data.cpp
 * 跨种子自对弈生成 NN 训练数据
 * 编译: g++ -std=c++17 -O2 -o gen_data scripts/gen_training_data.cpp
 * 用法: ./gen_data <num_seeds> <output_file>
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <deque>
#include <string>
#include <sstream>
#include <random>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <set>
#include <memory>

#include "../nn_solver/board_features.h"
#include "../nn_solver/replay_buffer.h"

// ============================================================
// 基础数据结构（与 1.cpp 兼容）
// ============================================================
struct Cell {
    int value = 1;
    int color() const { return std::abs(value); }
    bool is_bomb() const { return value < 0; }
    bool is_wildcard() const { return value == 0; }
};

struct Board {
    int N = 0, level = 1;
    std::vector<std::vector<Cell>> grid;
    std::shared_ptr<std::vector<std::vector<int>>> drop_queue;
    std::vector<int> queue_ptr;

    Board(int n = 0) : N(n), grid(n, std::vector<Cell>(n)),
        drop_queue(std::make_shared<std::vector<std::vector<int>>>()) {}

    Cell& at(int r, int c) { return grid[r][c]; }
    const Cell& at(int r, int c) const { return grid[r][c]; }
    bool in_bounds(int r, int c) const { return r>=0 && r<N && c>=0 && c<N; }
    bool is_deadlocked() const {
        for (int r=0;r<N;++r) for (int c=0;c<N;++c) {
            int ac = std::abs(at(r,c).value);
            if (c+1<N) { int c2=std::abs(at(r,c+1).value); if (ac==c2||ac==0||c2==0) return false; }
            if (r+1<N) { int c2=std::abs(at(r+1,c).value); if (ac==c2||ac==0||c2==0) return false; }
        }
        return true;
    }
};

int color_of(const Cell& c) { return std::abs(c.value); }

// ============================================================
// 路径评分
// ============================================================
int path_score(int k) {
    double t = std::sqrt((double)k) - 1.0;
    return 10*k + 18*(int)(t*t);
}
int path_score(const Board& b, const std::vector<std::pair<int,int>>& path) {
    int k=(int)path.size(), s=path_score(k);
    std::vector<std::vector<bool>> in_path(b.N,std::vector<bool>(b.N));
    for (auto& p : path) in_path[p.first][p.second]=true;
    std::vector<std::vector<bool>> exploded(b.N,std::vector<bool>(b.N));
    for (auto& p : path) {
        int r=p.first, c=p.second;
        if (!b.at(r,c).is_bomb()) continue;
        for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
            int nr=r+dr,nc=c+dc;
            if (b.in_bounds(nr,nc)&&!in_path[nr][nc]&&!exploded[nr][nc]) { exploded[nr][nc]=true; s+=10; }
        }
    }
    return s;
}

// ============================================================
// 方块生成
// ============================================================
int gen_block(std::mt19937& rng, int level) {
    if (level<=2) return (rng()%5)+1;
    if (level==3) return ((rng()%100)<15)?0:(rng()%5)+1;
    if (level==4) { int c=(rng()%5)+1; return ((rng()%100)<10)?-c:c; }
    if ((rng()%100)<15) return 0;
    int base=(rng()%5)+1; return ((rng()%100)<10)?-base:base;
}

void init_board(Board& b, int level, int seed, int N) {
    b = Board(N);
    b.level = level;
    b.drop_queue->assign(N, std::vector<int>(1000));
    b.queue_ptr.assign(N, 0);
    std::mt19937 rng(seed);
    for (int c=0;c<N;++c)
        for (int i=0;i<1000;++i)
            (*b.drop_queue)[c][i] = gen_block(rng, level);
    std::mt19937 rng_b(seed ^ 0x9E3779B9);
    for (int r=0;r<N;++r)
        for (int c=0;c<N;++c)
            b.at(r,c).value = gen_block(rng_b, level);
}

Board preview(const Board& b, const std::vector<std::pair<int,int>>& path) {
    Board nb = b;
    if (path.size()<2) return nb;
    auto& dq = *nb.drop_queue;
    std::vector<std::vector<bool>> in_path(b.N,std::vector<bool>(b.N));
    for (auto p:path) in_path[p.first][p.second]=true;
    auto to_remove = in_path;
    if (b.level>=4)
        for (auto& p : path) {
            int r=p.first, c=p.second;
            if (!b.at(r,c).is_bomb()) continue;
            for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
                int nr=r+dr,nc=c+dc;
                if (b.in_bounds(nr,nc)&&!in_path[nr][nc]) to_remove[nr][nc]=true;
            }
        }
    for (int c=0;c<b.N;++c) {
        std::vector<Cell> rem;
        for (int r=0;r<b.N;++r) if (!to_remove[r][c]) rem.push_back(b.at(r,c));
        int empty = b.N - (int)rem.size();
        for (int i=0;i<empty;++i) nb.at(i,c).value = dq[c][nb.queue_ptr[c]++];
        for (int i=0;i<(int)rem.size();++i) nb.at(empty+i,c) = rem[i];
    }
    return nb;
}

// ============================================================
// 简化求解器（用于数据生成，比完整版速度快）
// ============================================================
bool compatible(int a, int b) { return a==0||b==0||a==b; }
constexpr int DR[]={-1,1,0,0}, DC[]={0,0,-1,1};

std::vector<std::pair<int,int>> simple_solve(const Board& b) {
    int best_sc = 0;
    std::vector<std::pair<int,int>> best_path;
    std::vector<std::pair<int,int>> path;
    std::vector<bool> vis(b.N*b.N, false);
    int nodes = 0;
    const int LIMIT = 800000;

    std::function<void(int,int,int)> dfs = [&](int r, int c, int target) {
        if (++nodes > LIMIT) return;
        int co = color_of(b.at(r,c));
        int fixed = target;
        if (fixed==0 && co!=0) fixed = co;
        if (path.size() >= 2) {
            int sc = path_score(b, path);
            if (sc > best_sc || (sc==best_sc && path.size()>best_path.size())) {
                best_sc = sc; best_path = path;
            }
        }
        std::pair<int,int> nb[4]; int nbc=0;
        for (int d=0;d<4;++d) {
            int nr=r+DR[d], nc=c+DC[d];
            if (!b.in_bounds(nr,nc)||vis[nr*b.N+nc]) continue;
            if (!compatible(fixed, color_of(b.at(nr,nc)))) continue;
            nb[nbc++] = {nr,nc};
        }
        for (int i=0;i<nbc;++i) {
            int nr = nb[i].first, nc = nb[i].second;
            path.push_back(std::make_pair(nr,nc)); vis[nr*b.N+nc]=true;
            dfs(nr, nc, fixed);
            vis[nr*b.N+nc]=false; path.pop_back();
            if (nodes > LIMIT) return;
        }
    };

    for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
        path.clear(); path.push_back({r,c});
        vis.assign(b.N*b.N, false); vis[r*b.N+c]=true;
        dfs(r,c,0);
        if (nodes > LIMIT) break;
    }

    if (best_path.size() < 2) {
        // fallback
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            if (b.at(r,c).is_wildcard()) for (int d=0;d<4;++d) {
                int nr=r+DR[d],nc=c+DC[d];
                if (b.in_bounds(nr,nc)) return {{r,c},{nr,nc}};
            }
            int a = color_of(b.at(r,c));
            if (c+1<b.N && compatible(a,color_of(b.at(r,c+1)))) return {{r,c},{r,c+1}};
            if (r+1<b.N && compatible(a,color_of(b.at(r+1,c)))) return {{r,c},{r+1,c}};
        }
        return {{0,0},{0,1}};
    }
    return best_path;
}

// ============================================================
// 数据生成主逻辑
// ============================================================
int main(int argc, char** argv) {
    int num_seeds = (argc > 1) ? std::atoi(argv[1]) : 100;
    const char* out_path = (argc > 2) ? argv[2] : "nn_solver/training_data.bin";
    int start_seed = (argc > 3) ? std::atoi(argv[3]) : 1;

    std::printf("[DataGen] %d seeds, start=%d, output=%s\n", num_seeds, start_seed, out_path);

    ReplayBuffer buf(500000);
    int total_samples = 0;

    struct LvlCfg { int lv, N; const char* name; };
    std::vector<LvlCfg> levels = {
        {1,10,"L1"},{2,10,"L2"},{3,10,"L3"},{4,10,"L4"},{5,12,"L5"}
    };

    for (int si = 0; si < num_seeds; ++si) {
        int seed = start_seed + si;
        for (auto& lc : levels) {
            Board board;
            init_board(board, lc.lv, seed, lc.N);

            constexpr int MAX_STEPS = 50;
            int step = 0;
            int total_score = 0;
            std::vector<BoardFeatures> state_features;
            std::vector<int> state_step_scores;  // score gained at each step

            while (step < MAX_STEPS && !board.is_deadlocked()) {
                // 记录当前状态的特征
                auto feats = extract_features(board);
                state_features.push_back(feats);
                state_step_scores.push_back(total_score);

                auto path = simple_solve(board);
                if (path.size() < 2) break;

                int gained = path_score(board, path);
                total_score += gained;
                step++;

                board = preview(board, path);
            }

            // 计算每个状态的剩余分数，存入缓冲区
            for (size_t i = 0; i < state_features.size(); ++i) {
                Experience e;
                e.features = state_features[i];
                e.value = (double)(total_score - state_step_scores[i]);
                e.step = (int)i;
                e.level = lc.lv;
                buf.push(e);
                total_samples++;
            }
        }
        if ((si+1) % 20 == 0)
            std::printf("  [%d/%d] seeds done, %d samples collected\n", si+1, num_seeds, total_samples);
    }

    std::printf("[DataGen] Total: %d samples from %d seeds\n", total_samples, num_seeds);
    if (buf.save(out_path))
        std::printf("[DataGen] Saved to %s\n", out_path);
    else
        std::fprintf(stderr, "[DataGen] ERROR: Failed to save!\n");

    return 0;
}
