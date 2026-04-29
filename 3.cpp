#if __cplusplus < 201700L
#error "C++17 required"
#endif

#include <algorithm>
#include <cassert>
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

// ============================================================
// 数据模型（带底层完美沙盘推演）
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
    
    // 底层引擎变量：未来掉落物预知队列
    std::vector<std::vector<int>> drop_queue; 
    std::vector<int> queue_ptr;

    explicit Board(int n = 0) : N(n), grid(n, std::vector<Cell>(n)) {}

    Cell&       at(int r, int c)       { return grid[r][c]; }
    const Cell& at(int r, int c) const { return grid[r][c]; }
    bool in_bounds(int r, int c) const { return r >= 0 && r < N && c >= 0 && c < N; }

    // 【核心功能：未来预知引擎】
    // 输入：一条打算行走的路径
    // 返回：走出这条路径后的完美盘面预测（包含确切的新掉落方块！）
    Board preview(const std::vector<std::pair<int,int>>& path) const {
        Board next_b = *this; 
        if (path.size() < 2) return next_b;

        std::vector<std::vector<bool>> in_path(N, std::vector<bool>(N, false));
        for (auto p : path) in_path[p.first][p.second] = true;

        std::vector<std::vector<bool>> to_remove = in_path;

        // 模拟炸弹爆炸
        if (level >= 4) {
            for (auto [r, c] : path) {
                if (!at(r, c).is_bomb()) continue;
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        int nr = r + dr, nc = c + dc;
                        if (in_bounds(nr, nc) && !in_path[nr][nc]) {
                            to_remove[nr][nc] = true;
                        }
                    }
                }
            }
        }

        // 模拟重力下落与新块生成
        for (int c = 0; c < N; ++c) {
            std::vector<Cell> remaining;
            for (int r = 0; r < N; ++r) {
                if (!to_remove[r][c]) remaining.push_back(at(r, c));
            }
            int empty = N - (int)remaining.size();
            for (int i = 0; i < empty; ++i) {
                int val = next_b.drop_queue[c][next_b.queue_ptr[c]++];
                next_b.at(i, c).value = val;
            }
            for (int i = 0; i < (int)remaining.size(); ++i) {
                next_b.at(empty + i, c) = remaining[i];
            }
        }
        return next_b;
    }
    
    // 是否死局检测
    bool is_deadlocked() const {
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                int ac = at(r, c).color();
                if (c + 1 < N) {
                    int ac2 = at(r, c + 1).color();
                    if (ac == ac2 || ac == 0 || ac2 == 0) return false;
                }
                if (r + 1 < N) {
                    int ac2 = at(r + 1, c).color();
                    if (ac == ac2 || ac == 0 || ac2 == 0) return false;
                }
            }
        }
        return true;
    }
};

// ============================================================
// 中间件：GameController
// ============================================================

class GameController {
public:
    struct DropObservation {
        int col = 0;
        int value = 0;
    };

private:
    Board _board;
    int   _level = 0;
    int   _step  = 0;
    int   _score = 0;
    bool  _done  = false;
    bool  _has_feedback = false;
    bool  _last_valid = true;
    int   _last_reward = 0;
    std::string _pending_line;
    std::vector<std::pair<int,int>> _last_path;
    std::vector<DropObservation> _drop_obs;

    static int try_parse_level(const std::string& line, int& level, int& seed) {
        int lv, sd, N, steps;
        if (std::sscanf(line.c_str(), "LEVEL %d SEED %d SIZE %d STEPS %d", &lv, &sd, &N, &steps) == 4) {
            level = lv;
            seed = sd;
            return N;
        }
        return 0;
    }

    static bool try_parse_step(const std::string& line, int& step, int& score, bool& valid) {
        char buf[16] = {};
        if (std::sscanf(line.c_str(), "STEP %d SCORE %d %15s", &step, &score, buf) >= 3) {
            valid = (std::string(buf) == "VALID");
            return true;
        }
        return false;
    }

    static int gen_block(std::mt19937& rng, int level) {
        if (level <= 2) return (rng() % 5) + 1;
        else if (level == 3) return ((rng() % 100) < 15) ? 0 : (rng() % 5) + 1;
        else if (level == 4) {
            int color = (rng() % 5) + 1;
            return ((rng() % 100) < 10) ? -color : color;
        } else {
            if ((rng() % 100) < 15) return 0;
            int base = (rng() % 5) + 1;
            return ((rng() % 100) < 10) ? -base : base;
        }
    }

    static void init_queues(Board& b, int seed, int N, int level) {
        b.level = level;
        std::mt19937 rng(seed);
        b.drop_queue.assign(N, std::vector<int>(1000));
        b.queue_ptr.assign(N, 0);
        for (int c = 0; c < N; ++c) {
            for (int i = 0; i < 1000; ++i) b.drop_queue[c][i] = gen_block(rng, level);
        }
    }

    static std::vector<int> removed_per_column(const Board& b, const std::vector<std::pair<int,int>>& path) {
        std::vector<int> removed(b.N, 0);
        if (path.size() < 2) return removed;

        std::vector<std::vector<bool>> in_path(b.N, std::vector<bool>(b.N, false));
        std::vector<std::vector<bool>> to_remove(b.N, std::vector<bool>(b.N, false));
        for (auto [r, c] : path) {
            in_path[r][c] = true;
            to_remove[r][c] = true;
        }

        if (b.level >= 4) {
            for (auto [r, c] : path) {
                if (!b.at(r, c).is_bomb()) continue;
                for (int dr = -1; dr <= 1; ++dr) {
                    for (int dc = -1; dc <= 1; ++dc) {
                        int nr = r + dr;
                        int nc = c + dc;
                        if (b.in_bounds(nr, nc) && !in_path[nr][nc]) to_remove[nr][nc] = true;
                    }
                }
            }
        }

        for (int c = 0; c < b.N; ++c) {
            int cnt = 0;
            for (int r = 0; r < b.N; ++r) {
                if (to_remove[r][c]) ++cnt;
            }
            removed[c] = cnt;
        }
        return removed;
    }

    bool read_line(std::string& line) {
        if (!_pending_line.empty()) {
            line = std::move(_pending_line);
            _pending_line.clear();
            return true;
        }
        return (bool)std::getline(std::cin, line);
    }

    Board read_board(int N) {
        Board board(N);
        for (int row = 0; row < N; ++row) {
            std::string line;
            read_line(line);
            std::istringstream ls(line);
            for (int c = 0; c < N; ++c) ls >> board.at(row, c).value;
        }
        return board;
    }

    void drain_trailing() {
        std::string line;
        while (std::cin.rdbuf()->in_avail() > 0) {
            if (!read_line(line)) break;
            if (line.empty() || line.find("LEVEL_END") != std::string::npos) continue;
            if (line.find("FINAL_SCORE") != std::string::npos) {
                _done = true; continue;
            }
            _pending_line = std::move(line); break;
        }
    }

public:
    const Board& board() const { return _board; }
    int level() const { return _level; }
    int step()  const { return _step;  }
    int score() const { return _score; }
    bool done() const { return _done;  }
    bool has_feedback() const { return _has_feedback; }
    bool last_valid() const { return _last_valid; }
    int last_reward() const { return _last_reward; }
    const std::vector<DropObservation>& drop_observations() const { return _drop_obs; }

    bool update() {
        std::string first_line;
        while (true) {
            if (!read_line(first_line)) { _done = true; return false; }
            if (!first_line.empty()) break;
        }

        if (first_line.find("LEVEL_END") != std::string::npos ||
            first_line.find("FINAL_SCORE") != std::string::npos) {
            _done = true; return false;
        }

        int seed;
        int new_N = try_parse_level(first_line, _level, seed);
        if (new_N > 0) {
            Board new_board = read_board(new_N);
            init_queues(new_board, seed, new_N, _level);
            _board = std::move(new_board);
            _step = 0; _score = 0;
            _has_feedback = false;
            _drop_obs.clear();
            drain_trailing();
            return true;
        }

        int step, score; bool valid;
        if (try_parse_step(first_line, step, score, valid)) {
            int prev_score = _score;
            _step = step; _score = score;
            _last_valid = valid;
            _last_reward = valid ? (_score - prev_score) : -30;
            _has_feedback = true;
            _drop_obs.clear();

            std::vector<int> removed_cols;
            if (valid && !_last_path.empty()) {
                removed_cols = removed_per_column(_board, _last_path);
            }
            
            // 同步指针：如果上一步合法，我们在本地走一遍以消耗掉正确的 queue_ptr
            Board predicted = (valid && !_last_path.empty()) ? _board.preview(_last_path) : _board;
            
            Board new_board = read_board(_board.N);
            if (!removed_cols.empty()) {
                for (int c = 0; c < _board.N; ++c) {
                    int drops = removed_cols[c];
                    for (int r = 0; r < drops && r < _board.N; ++r) {
                        _drop_obs.push_back(DropObservation{c, new_board.at(r, c).value});
                    }
                }
            }
            new_board.level = _level;
            new_board.drop_queue = std::move(predicted.drop_queue);
            new_board.queue_ptr = std::move(predicted.queue_ptr);
            _board = std::move(new_board);
            _last_path.clear();

            drain_trailing();
            if (!_pending_line.empty()) { 
                int next_level, next_seed;
                int next_N = try_parse_level(_pending_line, next_level, next_seed);
                if (next_N > 0) {
                    _level = next_level;
                    _pending_line.clear();
                    Board nb = read_board(next_N);
                    init_queues(nb, next_seed, next_N, next_level);
                    _board = std::move(nb);
                    _step = 0; _score = 0;
                    _has_feedback = false;
                    _drop_obs.clear();
                    drain_trailing();
                }
            }
            return true;
        }

        _done = true;
        return false;
    }

    void respond(const std::vector<std::pair<int,int>>& path) {
        _last_path = path;
        std::cout << path.size();
        for (auto [r, c] : path) std::cout << ' ' << r << ' ' << c;
        std::cout << '\n';
        std::cout.flush();
    }
};

// ============================================================
// 工具函数
// ============================================================

constexpr int DR[] = {-1, 1, 0, 0};
constexpr int DC[] = { 0, 0,-1, 1};

int path_score(int k) {
    double t = std::sqrt(static_cast<double>(k)) - 1.0;
    return 10 * k + 18 * static_cast<int>(t * t);
}

int path_score(const Board& board, const std::vector<std::pair<int,int>>& path) {
    int k = static_cast<int>(path.size());
    int s = path_score(k);

    std::vector<std::vector<bool>> in_path(board.N, std::vector<bool>(board.N, false));
    for (auto [r, c] : path) in_path[r][c] = true;
    std::vector<std::vector<bool>> exploded(board.N, std::vector<bool>(board.N, false));

    for (auto [r, c] : path) {
        if (!board.at(r, c).is_bomb()) continue;
        for (int dr = -1; dr <= 1; ++dr)
            for (int dc = -1; dc <= 1; ++dc) {
                int nr = r + dr, nc = c + dc;
                if (board.in_bounds(nr, nc) && !in_path[nr][nc] && !exploded[nr][nc]) {
                    exploded[nr][nc] = true;
                    s += 10;
                }
            }
    }
    return s;
}

// ============================================================
// 解题逻辑（如果你不理解这份代码其他部分在干什么，请仅在此处进行策略实现和修改）
// ============================================================

static bool compatible_color(int target, int color) {
    return target == 0 || color == 0 || target == color;
}

static bool in_path(const std::vector<std::pair<int,int>>& path, int r, int c) {
    for (auto [pr, pc] : path) {
        if (pr == r && pc == c) return true;
    }
    return false;
}

static int anchor_color_of_path(const Board& board, const std::vector<std::pair<int,int>>& path) {
    for (auto [r, c] : path) {
        int color = board.at(r, c).color();
        if (color != 0) return color;
    }
    return 0;
}

static bool is_valid_same_color_path(const Board& board, const std::vector<std::pair<int,int>>& path) {
    if (path.size() < 2) return false;

    std::vector<std::vector<bool>> used(board.N, std::vector<bool>(board.N, false));
    int anchor = 0;
    for (size_t i = 0; i < path.size(); ++i) {
        int r = path[i].first;
        int c = path[i].second;
        if (!board.in_bounds(r, c) || used[r][c]) return false;
        used[r][c] = true;

        int color = board.at(r, c).color();
        if (color != 0) {
            if (anchor == 0) anchor = color;
            else if (anchor != color) return false;
        }

        if (i > 0) {
            int pr = path[i - 1].first;
            int pc = path[i - 1].second;
            int manhattan = std::abs(pr - r) + std::abs(pc - c);
            if (manhattan != 1) return false;
        }
    }
    return true;
}

class DropValueModel {
    // 值域按 [-5, 5] 建桶：负数代表炸弹，0 代表通配。
    static constexpr int K = 11;
    std::vector<std::array<double, K>> col_hist;
    std::array<double, K> global_hist{};

    static int idx_of(int v) {
        if (v < -5) v = -5;
        if (v > 5) v = 5;
        return v + 5;
    }

    static int value_of(int idx) {
        return idx - 5;
    }

public:
    void ensure_size(int n) {
        if ((int)col_hist.size() >= n) return;
        int old = (int)col_hist.size();
        col_hist.resize(n);
        for (int i = old; i < n; ++i) {
            for (int k = 0; k < K; ++k) col_hist[i][k] = 1.0;
        }
        for (int k = 0; k < K; ++k) {
            if (global_hist[k] == 0.0) global_hist[k] = 1.0;
        }
    }

    void observe(int col, int value) {
        if (col < 0 || col >= (int)col_hist.size()) return;
        int idx = idx_of(value);
        col_hist[col][idx] += 1.0;
        global_hist[idx] += 1.0;
    }

    double prob(int col, int idx) const {
        if (idx < 0 || idx >= K) return 0.0;
        if (col < 0 || col >= (int)col_hist.size()) {
            double gs = 0.0;
            for (int k = 0; k < K; ++k) gs += global_hist[k];
            return (gs > 0.0) ? (global_hist[idx] / gs) : (1.0 / K);
        }

        double cs = 0.0;
        for (int k = 0; k < K; ++k) cs += col_hist[col][k];
        if (cs > 0.0) return col_hist[col][idx] / cs;

        double gs = 0.0;
        for (int k = 0; k < K; ++k) gs += global_hist[k];
        return (gs > 0.0) ? (global_hist[idx] / gs) : (1.0 / K);
    }

    double predict_column_gain(const Board& board, int col) const {
        if (!board.in_bounds(0, col)) return 0.0;
        double gain = 0.0;
        for (int idx = 0; idx < K; ++idx) {
            double p = prob(col, idx);
            int v = value_of(idx);
            int color = std::abs(v);
            bool wildcard = (v == 0);

            double local = 0.0;
            if (wildcard) local += 1.5;

            if (board.in_bounds(0, col - 1)) {
                int lc = board.at(0, col - 1).color();
                if (wildcard || lc == 0 || lc == color) local += 1.0;
            }
            if (board.in_bounds(0, col + 1)) {
                int rc = board.at(0, col + 1).color();
                if (wildcard || rc == 0 || rc == color) local += 1.0;
            }
            if (board.in_bounds(1, col)) {
                int dc = board.at(1, col).color();
                if (wildcard || dc == 0 || dc == color) local += 0.8;
            }
            gain += p * local;
        }
        return gain;
    }

    double predict_board_gain(const Board& board) const {
        double s = 0.0;
        for (int c = 0; c < board.N; ++c) s += predict_column_gain(board, c);
        return s;
    }
};

class OnlineLinearModel {
public:
    static constexpr int D = 8;
    std::array<double, D> w{0.0, 0.8, 0.3, 0.1, 0.2, 0.3, -0.5, 0.4};
    double lr = 0.03;
    double l2 = 1e-4;

    double predict(const std::array<double, D>& x) const {
        double y = 0.0;
        for (int i = 0; i < D; ++i) y += w[i] * x[i];
        return y;
    }

    void update(const std::array<double, D>& x, double target) {
        double pred = predict(x);
        double err = target - pred;
        for (int i = 0; i < D; ++i) {
            w[i] += lr * (err * x[i] - l2 * w[i]);
        }
    }
};

class MLPathAgent {
    struct Candidate {
        std::vector<std::pair<int,int>> path;
        double heuristic = 0.0;
    };

    OnlineLinearModel value_model;
    DropValueModel drop_model;
    std::mt19937 rng{std::random_device{}()};
    std::array<double, OnlineLinearModel::D> last_feat{};
    bool has_last = false;
    double epsilon = 0.10;

    static std::vector<std::pair<int,int>> fallback_path(const Board& board) {
        for (int r = 0; r < board.N; ++r) {
            for (int c = 0; c < board.N; ++c) {
                int a = board.at(r, c).color();
                if (c + 1 < board.N) {
                    int b = board.at(r, c + 1).color();
                    if (a == b || a == 0 || b == 0) return {{r, c}, {r, c + 1}};
                }
                if (r + 1 < board.N) {
                    int b = board.at(r + 1, c).color();
                    if (a == b || a == 0 || b == 0) return {{r, c}, {r + 1, c}};
                }
            }
        }
        return {{0, 0}, {0, 1}};
    }

    static std::vector<Candidate> enumerate_bfs(const Board& board, int max_depth, int cap) {
        struct Node {
            std::vector<std::pair<int,int>> path;
            int target = 0;
        };

        std::vector<Candidate> out;
        std::deque<Node> q;
        for (int r = 0; r < board.N; ++r) {
            for (int c = 0; c < board.N; ++c) {
                Node n;
                n.path.push_back({r, c});
                q.push_back(std::move(n));
            }
        }

        while (!q.empty() && (int)out.size() < cap) {
            Node cur = std::move(q.front());
            q.pop_front();
            auto [r, c] = cur.path.back();
            int cell_color = board.at(r, c).color();
            if (cur.target == 0 && cell_color != 0) cur.target = cell_color;

            if ((int)cur.path.size() >= 2) {
                if (is_valid_same_color_path(board, cur.path)) {
                    Candidate cc;
                    cc.path = cur.path;
                    cc.heuristic = (double)path_score(board, cc.path) + 2.0 * (double)cc.path.size();
                    out.push_back(std::move(cc));
                    if ((int)out.size() >= cap) break;
                }
            }

            if ((int)cur.path.size() >= max_depth) continue;

            for (int d = 0; d < 4; ++d) {
                int nr = r + DR[d];
                int nc = c + DC[d];
                if (!board.in_bounds(nr, nc)) continue;
                if (in_path(cur.path, nr, nc)) continue;
                int nc_color = board.at(nr, nc).color();
                if (!compatible_color(cur.target, nc_color)) continue;

                Node nx = cur;
                nx.path.push_back({nr, nc});
                if (nx.target == 0 && nc_color != 0) nx.target = nc_color;
                q.push_back(std::move(nx));
            }
        }
        return out;
    }

    static int upper_bound_reachable(
        const Board& board,
        const std::vector<std::pair<int,int>>& path,
        int target
    ) {
        std::vector<std::vector<bool>> blocked(board.N, std::vector<bool>(board.N, false));
        for (auto [r, c] : path) blocked[r][c] = true;

        auto [tr, tc] = path.back();
        std::deque<std::pair<int,int>> q;
        std::vector<std::vector<bool>> seen(board.N, std::vector<bool>(board.N, false));

        for (int d = 0; d < 4; ++d) {
            int nr = tr + DR[d];
            int nc = tc + DC[d];
            if (!board.in_bounds(nr, nc) || blocked[nr][nc]) continue;
            int color = board.at(nr, nc).color();
            if (!compatible_color(target, color)) continue;
            seen[nr][nc] = true;
            q.push_back({nr, nc});
        }

        int reachable = 0;
        while (!q.empty()) {
            auto [r, c] = q.front();
            q.pop_front();
            ++reachable;

            for (int d = 0; d < 4; ++d) {
                int nr = r + DR[d];
                int nc = c + DC[d];
                if (!board.in_bounds(nr, nc) || blocked[nr][nc] || seen[nr][nc]) continue;
                int color = board.at(nr, nc).color();
                if (!compatible_color(target, color)) continue;
                seen[nr][nc] = true;
                q.push_back({nr, nc});
            }
        }

        return (int)path.size() + reachable;
    }

    static int quick_next_best_len(const Board& board) {
        auto cands = enumerate_bfs(board, 4, 120);
        int best_len = 0;
        for (const auto& c : cands) {
            if (is_valid_same_color_path(board, c.path)) {
                best_len = std::max(best_len, (int)c.path.size());
            }
        }
        return best_len;
    }

    static void dfs_extend(
        const Board& board,
        std::vector<std::pair<int,int>>& path,
        int target,
        int max_depth,
        int cap,
        std::vector<Candidate>& out,
        int& incumbent_best_len
    ) {
        if ((int)out.size() >= cap) return;
        auto [r, c] = path.back();
        int cc = board.at(r, c).color();
        if (target == 0 && cc != 0) target = cc;

        int ub = upper_bound_reachable(board, path, target);
        if (ub <= incumbent_best_len) return;

        if ((int)path.size() >= 2) {
            if (is_valid_same_color_path(board, path)) {
                Candidate cd;
                cd.path = path;
                cd.heuristic = (double)path_score(board, cd.path) + 3.0 * (double)cd.path.size();
                out.push_back(std::move(cd));
                incumbent_best_len = std::max(incumbent_best_len, (int)path.size());
                if ((int)out.size() >= cap) return;
            }
        }

        if ((int)path.size() >= max_depth) return;

        for (int d = 0; d < 4; ++d) {
            int nr = r + DR[d];
            int nc = c + DC[d];
            if (!board.in_bounds(nr, nc)) continue;
            if (in_path(path, nr, nc)) continue;
            int ncolor = board.at(nr, nc).color();
            if (!compatible_color(target, ncolor)) continue;

            path.push_back({nr, nc});
            dfs_extend(board, path, target, max_depth, cap, out, incumbent_best_len);
            path.pop_back();
            if ((int)out.size() >= cap) return;
        }
    }

    static std::array<double, OnlineLinearModel::D> build_features(
        const Board& before,
        const std::vector<std::pair<int,int>>& path,
        const Board& after,
        double predicted_drop_gain
    ) {
        std::array<double, OnlineLinearModel::D> x{};
        x[0] = 1.0;
        x[1] = (double)path_score(before, path) / 100.0;
        x[2] = (double)path.size() / 6.0;

        int wildcard = 0;
        int bombs = 0;
        double center_sum = 0.0;
        double center = ((double)before.N - 1.0) / 2.0;
        for (auto [r, c] : path) {
            const Cell& cell = before.at(r, c);
            if (cell.is_wildcard()) ++wildcard;
            if (cell.is_bomb()) ++bombs;
            center_sum += std::abs((double)r - center) + std::abs((double)c - center);
        }

        double k = (double)std::max(1, (int)path.size());
        double maxd = std::max(1.0, 2.0 * (double)(before.N - 1));
        x[3] = (double)wildcard / k;
        x[4] = (double)bombs / k;
        x[5] = 1.0 - (center_sum / (k * maxd));
        x[6] = after.is_deadlocked() ? 1.0 : 0.0;
        x[7] = predicted_drop_gain / 10.0;
        return x;
    }

public:
    void observe_feedback(int reward, const std::vector<GameController::DropObservation>& drops) {
        if (has_last) value_model.update(last_feat, (double)reward);
        for (const auto& d : drops) drop_model.observe(d.col, d.value);
    }

    std::vector<std::pair<int,int>> choose_path(const Board& board) {
        drop_model.ensure_size(board.N);

        auto bfs = enumerate_bfs(board, 5, 600);
        if (bfs.empty()) return fallback_path(board);

        std::sort(bfs.begin(), bfs.end(), [](const Candidate& a, const Candidate& b) {
            return a.heuristic > b.heuristic;
        });

        std::vector<Candidate> all = bfs;
        int seed_count = std::min(80, (int)bfs.size());
        int incumbent_best_len = 0;
        for (const auto& c : bfs) incumbent_best_len = std::max(incumbent_best_len, (int)c.path.size());
        for (int i = 0; i < seed_count && (int)all.size() < 2400; ++i) {
            std::vector<std::pair<int,int>> seed = bfs[i].path;
            int seed_target = anchor_color_of_path(board, seed);
            dfs_extend(board, seed, seed_target, 10, 2400, all, incumbent_best_len);
        }

        if (all.empty()) return fallback_path(board);

        int global_longest = 0;
        for (const auto& c : all) {
            if (is_valid_same_color_path(board, c.path)) {
                global_longest = std::max(global_longest, (int)c.path.size());
            }
        }
        if (global_longest < 2) return fallback_path(board);

        std::vector<int> longest_ids;
        longest_ids.reserve(all.size());
        for (int i = 0; i < (int)all.size(); ++i) {
            if ((int)all[i].path.size() == global_longest && is_valid_same_color_path(board, all[i].path)) {
                longest_ids.push_back(i);
            }
        }
        if (longest_ids.empty()) return fallback_path(board);

        std::uniform_real_distribution<double> u01(0.0, 1.0);
        int chosen = longest_ids[0];
        if (u01(rng) < epsilon) {
            std::uniform_int_distribution<int> uid(0, (int)longest_ids.size() - 1);
            chosen = longest_ids[uid(rng)];
        } else {
            double best = -std::numeric_limits<double>::infinity();
            for (int id : longest_ids) {
                const auto& p = all[id].path;
                Board next_b = board.preview(p);
                double future_gain = drop_model.predict_board_gain(next_b);
                int next_longest = quick_next_best_len(next_b);
                auto feat = build_features(board, p, next_b, future_gain);
                double pred = value_model.predict(feat);
                double score = pred + 0.10 * all[id].heuristic + 0.35 * (double)next_longest + 0.25 * future_gain;
                if (score > best) {
                    best = score;
                    chosen = id;
                }
            }
        }

        Board selected_next = board.preview(all[chosen].path);
        double selected_future = drop_model.predict_board_gain(selected_next);
        last_feat = build_features(board, all[chosen].path, selected_next, selected_future);
        has_last = true;
        return all[chosen].path;
    }
};

static std::uint64_t board_hash_for_cache(const Board& b) {
    std::uint64_t h = 1469598103934665603ULL;
    const std::uint64_t prime = 1099511628211ULL;
    h ^= (std::uint64_t)b.N;
    h *= prime;
    h ^= (std::uint64_t)b.level;
    h *= prime;
    for (int r = 0; r < b.N; ++r) {
        for (int c = 0; c < b.N; ++c) {
            std::uint64_t v = (std::uint64_t)(b.at(r, c).value + 17);
            h ^= v;
            h *= prime;
        }
    }
    return h;
}

static int exact_best_one_step_score(const Board& board) {
    int best_score = 0;
    std::vector<std::vector<bool>> visited(board.N, std::vector<bool>(board.N, false));
    std::vector<std::pair<int,int>> path;

    std::function<void(int,int,int)> dfs = [&](int r, int c, int target) {
        int cur_color = board.at(r, c).color();
        int fixed = target;
        if (fixed == 0 && cur_color != 0) fixed = cur_color;

        if ((int)path.size() >= 2) {
            best_score = std::max(best_score, path_score(board, path));
        }

        for (int d = 0; d < 4; ++d) {
            int nr = r + DR[d];
            int nc = c + DC[d];
            if (!board.in_bounds(nr, nc) || visited[nr][nc]) continue;
            int next_color = board.at(nr, nc).color();
            if (!compatible_color(fixed, next_color)) continue;

            visited[nr][nc] = true;
            path.push_back({nr, nc});
            dfs(nr, nc, fixed);
            path.pop_back();
            visited[nr][nc] = false;
        }
    };

    for (int r = 0; r < board.N; ++r) {
        for (int c = 0; c < board.N; ++c) {
            visited[r][c] = true;
            path.clear();
            path.push_back({r, c});
            dfs(r, c, 0);
            visited[r][c] = false;
        }
    }
    return best_score;
}

std::vector<std::pair<int,int>> find_best_path(const Board& board) {
    std::vector<std::pair<int,int>> best_path;
    int best_two_step = std::numeric_limits<int>::min();
    int best_now = std::numeric_limits<int>::min();

    std::unordered_map<std::uint64_t, int> next_score_cache;
    std::vector<std::vector<bool>> visited(board.N, std::vector<bool>(board.N, false));
    std::vector<std::pair<int,int>> path;

    auto better = [&](int two_step, int now_score, const std::vector<std::pair<int,int>>& cand) {
        if (two_step != best_two_step) return two_step > best_two_step;
        if (now_score != best_now) return now_score > best_now;
        return cand < best_path;
    };

    std::function<void(int,int,int)> dfs = [&](int r, int c, int target) {
        int cur_color = board.at(r, c).color();
        int fixed = target;
        if (fixed == 0 && cur_color != 0) fixed = cur_color;

        if ((int)path.size() >= 2) {
            int now_score = path_score(board, path);
            Board next_b = board.preview(path);
            std::uint64_t h = board_hash_for_cache(next_b);
            int next_best = 0;
            auto it = next_score_cache.find(h);
            if (it != next_score_cache.end()) {
                next_best = it->second;
            } else {
                next_best = exact_best_one_step_score(next_b);
                next_score_cache.emplace(h, next_best);
            }
            int two_step = now_score + next_best;

            if (better(two_step, now_score, path)) {
                best_two_step = two_step;
                best_now = now_score;
                best_path = path;
            }
        }

        for (int d = 0; d < 4; ++d) {
            int nr = r + DR[d];
            int nc = c + DC[d];
            if (!board.in_bounds(nr, nc) || visited[nr][nc]) continue;

            int next_color = board.at(nr, nc).color();
            if (!compatible_color(fixed, next_color)) continue;

            visited[nr][nc] = true;
            path.push_back({nr, nc});
            dfs(nr, nc, fixed);
            path.pop_back();
            visited[nr][nc] = false;
        }
    };

    for (int r = 0; r < board.N; ++r) {
        for (int c = 0; c < board.N; ++c) {
            visited[r][c] = true;
            path.clear();
            path.push_back({r, c});
            dfs(r, c, 0);
            visited[r][c] = false;
        }
    }

    if (best_path.size() >= 2) return best_path;
    return {{0, 0}, {0, 1}};
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    GameController ctl;
    while (ctl.update()) {
        auto path = find_best_path(ctl.board());
        ctl.respond(path);
    }
    return 0;
}