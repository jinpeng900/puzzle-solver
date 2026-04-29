#if __cplusplus < 201700L
#error "C++17 required"
#endif

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
#include <fstream>
#include <memory>
#include <numeric>
#include <cstring>
#include <iomanip>

// ============================================================
// 可调参数系统
// ============================================================
struct LevelParams {
    int dfs_limit = 1200000;
    int keep1 = 200, keep2 = 100, keep3 = 20;
    int cache_limit = 80000;
    double mxc_w = 25.0;
    double dq_w = 0.8, bh_w = 0.02;
    double s3_dq_w = 0.55, s3_bh_w = 0.025;
    double surv_mul = 1.0;
    int beam_w = 8, beam_d = 4;
    int short_pen = 55;
    int surv_check = 6;
    double beam_bonus = 0.3;
    double len_bonus = 0.0;
    double tfast_ratio = 0.90, tdead_ratio = 0.97;
    int max_time_ms = 10000;
    int fallback_len_thresh = 6;
};
static LevelParams g_params[6];
static int g_cur_level = 1;

static void init_default_params() {
    g_params[1] = {1050000, 210, 75, 16, 70000, 21.0, 1.0, 0.02, 0.55, 0.0, 0.5, 7, 5, 10, 10, 0.0, 0.0, 0.89, 0.98, 10500, 5};
    g_params[2] = {950000, 180, 20, 12, 60000, 20.5, 0.95, 0.03, 0.85, 0.015, 0.6, 10, 0, 45, 7, 0.75, 0.25, 0.87, 0.94, 11000, 3};
    g_params[3] = {300000, 160, 85, 21, 50000, 25.5, 0.65, 0.0, 0.3, 0.035, 0.9, 2, 1, 50, 9, 0.25, 0.15, 0.81, 0.94, 8000, 5};
    g_params[4] = {900000, 60, 110, 17, 60000, 14.5, 0.5, 0.045, 0.55, 0.01, 1.1, 8, 5, 80, 4, 0.4, 0.0, 0.91, 0.95, 14000, 3};
    g_params[5] = {1100000, 80, 20, 6, 25000, 16.5, 0.8, 0.0, 1.2, 0.035, 0.2, 5, 1, 60, 3, 0.65, 0.25, 0.9, 0.93, 11000, 8};
}

static void load_params(const char* fname) {
    std::ifstream f(fname);
    if(!f.is_open()) return;
    std::string line;
    while(std::getline(f,line)) {
        std::istringstream iss(line);
        std::string key; double v;
        if(!(iss>>key>>v)) continue;
        int lv=0;
        if(key.size()>=2 && key[0]=='L') lv=key[1]-'0';
        if(lv<1||lv>5) continue;
        std::string field = key.substr(3);
        auto& p = g_params[lv];
        if(field=="dfs_limit") p.dfs_limit=(int)v;
        else if(field=="keep1") p.keep1=(int)v;
        else if(field=="keep2") p.keep2=(int)v;
        else if(field=="keep3") p.keep3=(int)v;
        else if(field=="cache_limit") p.cache_limit=(int)v;
        else if(field=="mxc_w") p.mxc_w=v;
        else if(field=="dq_w") p.dq_w=v;
        else if(field=="bh_w") p.bh_w=v;
        else if(field=="s3_dq_w") p.s3_dq_w=v;
        else if(field=="s3_bh_w") p.s3_bh_w=v;
        else if(field=="surv_mul") p.surv_mul=v;
        else if(field=="beam_w") p.beam_w=(int)v;
        else if(field=="beam_d") p.beam_d=(int)v;
        else if(field=="short_pen") p.short_pen=(int)v;
        else if(field=="surv_check") p.surv_check=(int)v;
        else if(field=="beam_bonus") p.beam_bonus=v;
        else if(field=="len_bonus") p.len_bonus=v;
        else if(field=="tfast") p.tfast_ratio=v;
        else if(field=="tdead") p.tdead_ratio=v;
        else if(field=="max_time") p.max_time_ms=(int)v;
        else if(field=="fallback_len") p.fallback_len_thresh=(int)v;
    }
}

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

    explicit Board(int n = 0) : N(n), grid(n, std::vector<Cell>(n)), drop_queue(std::make_shared<std::vector<std::vector<int>>>()) {}
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
            for (int i = 0; i < (int)remaining.size(); ++i) next_b.at(empty+i,c) = remaining[i];
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
};

constexpr int DR[]={-1,1,0,0};
constexpr int DC[]={0,0,-1,1};

int path_score(int k) { double t=std::sqrt((double)k)-1.0; return 10*k+18*(int)(t*t); }
int path_score(const Board& board, const std::vector<std::pair<int,int>>& path) {
    int k=(int)path.size(), s=path_score(k);
    std::vector<std::vector<bool>> in_path(board.N,std::vector<bool>(board.N));
    for (auto [r,c]:path) in_path[r][c]=true;
    std::vector<std::vector<bool>> exploded(board.N,std::vector<bool>(board.N));
    for (auto [r,c]:path) {
        if (!board.at(r,c).is_bomb()) continue;
        for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
            int nr=r+dr,nc=c+dc;
            if (board.in_bounds(nr,nc)&&!in_path[nr][nc]&&!exploded[nr][nc]) { exploded[nr][nc]=true; s+=10; }
        }
    }
    return s;
}

static bool compatible_color(int target, int color) { return target==0||color==0||target==color; }
static std::uint64_t bit(int r, int c, int N) { return 1ULL << (r * N + c); }
static std::uint64_t board_hash(const Board& b) {
    std::uint64_t h=1469598103934665603ULL; const std::uint64_t prime=1099511628211ULL;
    h^=(std::uint64_t)b.N; h*=prime; h^=(std::uint64_t)b.level; h*=prime;
    for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) { h^=(std::uint64_t)(b.at(r,c).value+17); h*=prime; }
    return h;
}

// ============================================================
// 统计学盘面特征提取器 (Statistical Board Feature Extractor)
// ============================================================
// 从盘面提取 ~50 维统计特征，用于神经网络评估
constexpr int NUM_BOARD_FEATURES = 52;

struct BoardFeatures {
    double f[NUM_BOARD_FEATURES];
    BoardFeatures() { for(int i=0;i<NUM_BOARD_FEATURES;++i) f[i]=0.0; }
    double& operator[](int i) { return f[i]; }
    const double& operator[](int i) const { return f[i]; }
};

static BoardFeatures extract_features(const Board& b) {
    BoardFeatures bf;
    int N = b.N, total_cells = N*N;
    int color_cnt[6] = {0}, bomb_cnt = 0, wc_cnt = 0;
    int adjacent_same = 0, adjacent_diff_compat = 0;
    double pos_sum_color[5] = {0}, pos_sum_row[5] = {0}, pos_sum_col[5] = {0};
    int color_even[5] = {0}, color_odd[5] = {0};

    // 连通分量分析
    std::vector<std::vector<bool>> vis(N, std::vector<bool>(N, false));
    std::vector<int> comp_sizes;
    int max_comp = 0;

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int val = b.at(r,c).value;
            if (val == 0) { wc_cnt++; }
            if (val < 0) { bomb_cnt++; color_cnt[std::abs(val)]++; }
            else { color_cnt[val]++; }

            if (val > 0) {
                pos_sum_color[val-1] += r*N + c;
                pos_sum_row[val-1] += r;
                pos_sum_col[val-1] += c;
                if ((r+c)%2 == 0) color_even[val-1]++;
                else color_odd[val-1]++;
            }

            if (vis[r][c]) continue;
            int co = b.at(r,c).color();
            int sz = 0;
            std::deque<std::pair<int,int>> q;
            q.push_back({r,c}); vis[r][c] = true;
            while (!q.empty()) {
                auto [cr,cc] = q.front(); q.pop_front(); sz++;
                int cco = b.at(cr,cc).color();
                for (int d = 0; d < 4; ++d) {
                    int nr = cr + DR[d], nc = cc + DC[d];
                    if (!b.in_bounds(nr,nc) || vis[nr][nc]) continue;
                    if (!compatible_color(cco, b.at(nr,nc).color())) continue;
                    vis[nr][nc] = true;
                    q.push_back({nr,nc});
                }
            }
            comp_sizes.push_back(sz);
            max_comp = std::max(max_comp, sz);
        }
    }

    // 邻接对统计
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int ac = b.at(r,c).color();
            if (c+1 < N) { int bc2 = b.at(r,c+1).color(); if (compatible_color(ac,bc2)) { adjacent_diff_compat++; if (ac==bc2&&ac!=0) adjacent_same++; } }
            if (r+1 < N) { int bc3 = b.at(r+1,c).color(); if (compatible_color(ac,bc3)) { adjacent_diff_compat++; if (ac==bc3&&ac!=0) adjacent_same++; } }
        }
    }

    // 掉落队列预测质量
    double dq_compat = 0, dq_wc_expect = 0, dq_bomb_expect = 0;
    if (b.drop_queue) {
        auto& dq = *b.drop_queue;
        for (int c = 0; c < N; ++c) {
            int qp = b.queue_ptr[c];
            for (int i = 0; i < 10; ++i) {
                if (qp + i >= 1000) break;
                int v = dq[c][qp+i];
                if (v == 0) { dq_wc_expect += 1.0/N/10; continue; }
                if (v < 0) { dq_bomb_expect += 1.0/N/10; continue; }
                int col = std::abs(v);
                for (int d = 0; d < 4; ++d) {
                    int nr = DR[d], nc = c + DC[d];
                    if (b.in_bounds(nr,nc) && compatible_color(col, b.at(nr,nc).color()))
                        dq_compat += 0.1 / N / 10;
                }
            }
        }
    }

    // 平均连通分量大小和方差
    double avg_comp = 0, var_comp = 0;
    if (!comp_sizes.empty()) {
        double sum_comp = 0;
        for (int sz : comp_sizes) sum_comp += sz;
        avg_comp = sum_comp / (int)comp_sizes.size();
        for (int sz : comp_sizes) var_comp += (sz - avg_comp) * (sz - avg_comp);
        var_comp /= (double)comp_sizes.size();
    }

    // 颜色熵
    double entropy = 0;
    int num_colored = total_cells - wc_cnt;
    for (int i = 1; i <= 5; ++i) {
        if (color_cnt[i] > 0 && num_colored > 0) {
            double p = (double)color_cnt[i] / num_colored;
            entropy -= p * std::log(p);
        }
    }

    // 空间分散性：每种颜色的位置方差
    double spatial_var[5] = {0};
    for (int ci = 0; ci < 5; ++ci) {
        if (color_cnt[ci+1] <= 1) continue;
        double avg_pos = pos_sum_color[ci] / color_cnt[ci+1];
        // 近似空间方差
        spatial_var[ci] = 1.0; // 简化
    }

    // 连通分量大小分布区间: 1, 2-3, 4-6, 7-10, 11+
    int comp_bin[5] = {0};
    for (int sz : comp_sizes) {
        if (sz == 1) comp_bin[0]++;
        else if (sz <= 3) comp_bin[1]++;
        else if (sz <= 6) comp_bin[2]++;
        else if (sz <= 10) comp_bin[3]++;
        else comp_bin[4]++;
    }

    // 行/列颜色多样性
    double row_diversity = 0, col_diversity = 0;
    for (int r = 0; r < N; ++r) {
        std::set<int> s;
        for (int c = 0; c < N; ++c) if (b.at(r,c).color() > 0) s.insert(b.at(r,c).color());
        row_diversity += (double)s.size() / N;
    }
    for (int c = 0; c < N; ++c) {
        std::set<int> s;
        for (int r = 0; r < N; ++r) if (b.at(r,c).color() > 0) s.insert(b.at(r,c).color());
        col_diversity += (double)s.size() / N;
    }

    // 连通分量中最大分量占总盘面比例
    double max_comp_ratio = total_cells > 0 ? (double)max_comp / total_cells : 0;

    // 黑/白格子连通性（棋盘格染色分析）
    int black_comp = 0, white_comp = 0;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if (compatible_color(1, b.at(r,c).color())) { // 粗略估计
                if ((r+c)%2 == 0) black_comp++;
                else white_comp++;
            }

    // 填充特征数组 (52维)
    int idx = 0;
    // f[0-5]: 各颜色计数比例
    for (int i = 0; i <= 5; ++i) bf[idx++] = total_cells > 0 ? (double)color_cnt[i] / total_cells : 0;
    // f[6-7]: 万能块、炸弹比例
    bf[idx++] = total_cells > 0 ? (double)wc_cnt / total_cells : 0;
    bf[idx++] = total_cells > 0 ? (double)bomb_cnt / total_cells : 0;
    // f[8-11]: 邻接对
    bf[idx++] = (double)adjacent_same / (2.0*N*(N-1)); // 归一化到~[0,1]
    bf[idx++] = (double)adjacent_diff_compat / (2.0*N*(N-1));
    bf[idx++] = total_cells > 0 ? (double)adjacent_same / total_cells : 0;
    bf[idx++] = total_cells > 0 ? (double)adjacent_diff_compat / total_cells : 0;
    // f[12-15]: 连通分量统计
    bf[idx++] = (double)max_comp / total_cells;
    bf[idx++] = avg_comp / total_cells;
    bf[idx++] = std::sqrt(var_comp) / total_cells; // 标准差归一化
    bf[idx++] = (double)comp_sizes.size() / total_cells; // 分量密度
    // f[16-20]: 连通分量大小分布
    for (int i = 0; i < 5; ++i) bf[idx++] = (double)comp_bin[i] / std::max(1, (int)comp_sizes.size());
    // f[21]: 熵
    bf[idx++] = entropy;
    // f[22-26]: 各颜色一阶邻接对数（均匀配对潜力）
    for (int ci = 0; ci < 5; ++ci) bf[idx++] = (double)color_cnt[ci+1] / std::max(1, total_cells);
    // f[27-31]: 各颜色空间分散性
    for (int ci = 0; ci < 5; ++ci) bf[idx++] = spatial_var[ci];
    // f[32-33]: 行列多样性
    bf[idx++] = row_diversity;
    bf[idx++] = col_diversity;
    // f[34]: 死锁标志
    bf[idx++] = b.is_deadlocked() ? 1.0 : 0.0;
    // f[35-37]: 掉落队列预测
    bf[idx++] = dq_compat;
    bf[idx++] = dq_wc_expect;
    bf[idx++] = dq_bomb_expect;
    // f[38-42]: 棋盘格分析
    bf[idx++] = (double)black_comp / total_cells;
    bf[idx++] = (double)white_comp / total_cells;
    bf[idx++] = total_cells > 0 ? (double)(black_comp + white_comp) / total_cells : 0;
    bf[idx++] = black_comp + white_comp > 0 ? (double)black_comp / (black_comp + white_comp) : 0.5;
    // f[41]: 平均颜色位置 = 盘面有序度
    double pos_entropy = 0;
    for (int ci = 0; ci < 5; ++ci)
        if (color_cnt[ci+1] > 0)
            pos_entropy += (double)color_cnt[ci+1] / std::max(1, total_cells);
    bf[idx++] = pos_entropy;
    // f[42-46]: 各颜色二部图分析
    for (int ci = 0; ci < 5; ++ci) {
        int total_ci = color_even[ci] + color_odd[ci];
        bf[idx++] = total_ci > 0 ? (double)color_even[ci] / total_ci : 0.5;
    }
    // f[47]: 盘面级数
    bf[idx++] = (double)b.level / 5.0;
    // f[48]: 盘面大小
    bf[idx++] = (double)b.N / 12.0;
    // f[49-51]: 压缩盘面特征（行/列主导颜色统计）
    int row_dominant = 0, col_dominant = 0;
    for (int r = 0; r < N; ++r) {
        int row_colors[6] = {};
        for (int c = 0; c < N; ++c) row_colors[b.at(r,c).color()]++;
        int dom = 0;
        for (int i = 1; i <= 5; ++i) if (row_colors[i] > row_colors[dom]) dom = i;
        row_dominant += row_colors[dom];
    }
    for (int c = 0; c < N; ++c) {
        int col_colors[6] = {};
        for (int r = 0; r < N; ++r) col_colors[b.at(r,c).color()]++;
        int dom = 0;
        for (int i = 1; i <= 5; ++i) if (col_colors[i] > col_colors[dom]) dom = i;
        col_dominant += col_colors[dom];
    }
    bf[idx++] = (double)row_dominant / total_cells / N;
    bf[idx++] = (double)col_dominant / total_cells / N;
    bf[idx++] = total_cells > 0 ? (double)max_comp / (1.0 + color_cnt[1] + color_cnt[2] + color_cnt[3] + color_cnt[4] + color_cnt[5]) : 0;

    return bf;
}

// ============================================================
// 轻量级神经网络实现 (From-scratch MLP)
// ============================================================
struct NNLayer {
    std::vector<std::vector<double>> W; // weights [output_neurons][input_neurons]
    std::vector<double> b;              // biases [output_neurons]
    std::vector<double> z;              // pre-activation
    std::vector<double> a;              // post-activation
    int input_dim, output_dim;
    bool use_relu;

    NNLayer(int in, int out, bool relu = true) : input_dim(in), output_dim(out), use_relu(relu) {
        W.assign(out, std::vector<double>(in, 0));
        b.assign(out, 0);
        z.assign(out, 0);
        a.assign(out, 0);
        init_xavier();
    }

    void init_xavier() {
        double scale = std::sqrt(2.0 / std::max(1.0, (double)(input_dim + output_dim)));
        static std::mt19937 rng(114514);
        std::normal_distribution<double> dist(0, scale);
        for (int i = 0; i < output_dim; ++i) {
            for (int j = 0; j < input_dim; ++j) W[i][j] = dist(rng);
            b[i] = 0;
        }
    }

    void forward(const std::vector<double>& input, std::vector<double>& output) {
        for (int i = 0; i < output_dim; ++i) {
            double sum = b[i];
            const auto& w_row = W[i];
            for (int j = 0; j < input_dim; ++j) sum += w_row[j] * input[j];
            if (sum != sum) sum = 0; // NaN guard
            if (sum > 50.0) sum = 50.0;
            if (sum < -50.0) sum = -50.0;
            z[i] = sum;
            if (use_relu) a[i] = std::max(0.0, sum);
            else a[i] = std::tanh(sum);
        }
        output = a;
    }

    // 返回梯度 (对 W, b, input)
    void backward(const std::vector<double>& delta, const std::vector<double>& prev_a,
                  std::vector<double>& dW_sum, std::vector<double>& db_sum,
                  std::vector<double>& prev_delta) {
        for (int i = 0; i < output_dim; ++i) {
            double d_act;
            if (use_relu) d_act = (z[i] > 0) ? delta[i] : 0;
            else d_act = delta[i] * (1 - a[i]*a[i]); // tanh derivative

            db_sum[i] += d_act;
            for (int j = 0; j < input_dim; ++j) {
                dW_sum[i * input_dim + j] += d_act * prev_a[j];
                prev_delta[j] += d_act * W[i][j];
            }
        }
    }
};

class DeepPredictor {
    std::vector<NNLayer> layers_;
    int input_dim_, output_dim_;
    double learning_rate_;
    double scale_factor_ = 1000.0; // 将NN输出[−1,1]映射到约[−1000,1000]

    // Adam optimizer state
    struct AdamState {
        std::vector<std::vector<double>> mW; // first moment (same shape as W)
        std::vector<std::vector<double>> vW; // second moment
        std::vector<double> mb;
        std::vector<double> vb;
        double beta1_t = 1.0, beta2_t = 1.0;
    };
    std::vector<AdamState> adam_states_;
    int batch_count_ = 0;

public:
    DeepPredictor() : input_dim_(NUM_BOARD_FEATURES), output_dim_(1), learning_rate_(0.001) {
        init_network();
    }

    DeepPredictor(const std::vector<int>& hidden_sizes) : input_dim_(NUM_BOARD_FEATURES), output_dim_(1), learning_rate_(0.001) {
        init_network(hidden_sizes);
    }

    void init_network(const std::vector<int>& hidden_sizes = {64, 32}) {
        layers_.clear();
        adam_states_.clear();
        int prev = input_dim_;
        for (int size : hidden_sizes) {
            layers_.emplace_back(prev, size, true); // ReLU hidden layers
            prev = size;
        }
        layers_.emplace_back(prev, output_dim_, false); // tanh output layer
        init_adam();
    }

    void init_adam() {
        adam_states_.clear();
        for (auto& layer : layers_) {
            AdamState as;
            as.mW.assign(layer.output_dim, std::vector<double>(layer.input_dim, 0));
            as.vW.assign(layer.output_dim, std::vector<double>(layer.input_dim, 0));
            as.mb.assign(layer.output_dim, 0);
            as.vb.assign(layer.output_dim, 0);
            adam_states_.push_back(as);
        }
    }

    double predict(const BoardFeatures& features) {
        std::vector<double> input(features.f, features.f + NUM_BOARD_FEATURES);
        std::vector<double> output;
        for (auto& layer : layers_) {
            output.clear();
            layer.forward(input, output);
            input = output;
        }
        return input[0] * scale_factor_; // 反归一化
    }

    double predict_board(const Board& b) {
        return predict(extract_features(b));
    }

    void set_scale_factor(double s) { scale_factor_ = s; }
    double scale_factor() const { return scale_factor_; }

    double predict_raw(const BoardFeatures& features) {
        std::vector<double> input(features.f, features.f + NUM_BOARD_FEATURES);
        std::vector<double> output;
        for (auto& layer : layers_) {
            output.clear();
            layer.forward(input, output);
            input = output;
        }
        return input[0];
    }

    // Mini-batch training on one sample
    double train_one(const BoardFeatures& features, double target) {
        return train_batch(&features, &target, 1);
    }

    // Batch training
    double train_batch(const BoardFeatures* features, const double* targets, int batch_size) {
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        int L = (int)layers_.size();
        batch_count_++;

        // 收集中间激活值
        std::vector<std::vector<double>> layer_inputs(L + 1);
        layer_inputs[0].assign(features[0].f, features[0].f + NUM_BOARD_FEATURES);

        gradients_.resize(L);
        for (int l = 0; l < L; ++l) {
            gradients_[l].assign(layers_[l].output_dim * (layers_[l].input_dim + 1), 0.0);
        }

        double total_loss = 0;

        for (int bi = 0; bi < batch_size; ++bi) {
            // Forward pass
            std::vector<double> input(features[bi].f, features[bi].f + NUM_BOARD_FEATURES);
            std::vector<std::vector<double>> activations(L + 1);
            activations[0] = input;

            for (int l = 0; l < L; ++l) {
                std::vector<double> out;
                layers_[l].forward(activations[l], out);
                activations[l + 1] = out;
            }

            double predicted = activations[L][0];
            double error = predicted - targets[bi];
            total_loss += error * error;

            // Backward pass
            std::vector<double> delta = { 2.0 * error / batch_size }; // output delta (MSE gradient)

            for (int l = L - 1; l >= 0; --l) {
                int out_dim = layers_[l].output_dim;
                int in_dim = layers_[l].input_dim;
                std::vector<double> prev_delta(in_dim, 0);

                double d_act_scale = 1.0 / batch_size;
                for (int i = 0; i < out_dim; ++i) {
                    double d_act;
                    if (layers_[l].use_relu)
                        d_act = (layers_[l].z[i] > 0) ? delta[i] : 0;
                    else
                        d_act = delta[i] * (1 - layers_[l].a[i] * layers_[l].a[i]);

                    for (int j = 0; j < in_dim; ++j) {
                        layers_[l].W[i][j] = layers_[l].W[i][j]; // will accumulate in grad
                        prev_delta[j] += d_act * layers_[l].W[i][j];
                    }
                }
                delta = prev_delta;
            }
        }

        return (double)batch_size > 0 ? total_loss / batch_size : 0;
    }

    // 随机梯度下降更新（每样本），带梯度裁剪
    void sgd_update(const BoardFeatures& features, double target) {
        if (target != target) return;
        const double lr = learning_rate_;
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        int L = (int)layers_.size();

        // 归一化目标值到[-1,1]
        double normalized_target = target / std::max(1.0, scale_factor_);
        if (normalized_target > 1.0) normalized_target = 1.0;
        if (normalized_target < -1.0) normalized_target = -1.0;

        std::vector<double> input(features.f, features.f + NUM_BOARD_FEATURES);
        std::vector<std::vector<double>> activations(L + 1);
        activations[0] = input;

        // Forward
        for (int l = 0; l < L; ++l) {
            std::vector<double> out;
            layers_[l].forward(activations[l], out);
            activations[l + 1] = out;
        }

        double predicted = activations[L][0];
        double error = predicted - normalized_target;

        // Output delta: dE/dy = 2*(y-t); for tanh scaled
        std::vector<double> delta_out = { 2.0 * error };
        // scale by tanh derivative if output layer uses tanh
        // (already handled by backward, but for clarity do it here)
        if (!layers_.back().use_relu) {
            double y = layers_.back().a[0];
            delta_out[0] *= (1 - y * y);
        }

        std::vector<double> delta = delta_out;

        // Backward and update
        for (int l = L - 1; l >= 0; --l) {
            int out_dim = layers_[l].output_dim;
            int in_dim = layers_[l].input_dim;
            std::vector<double> prev_delta(in_dim, 0);

            // 使用 Adam 更新（先更新衰减因子再计算修正）
            adam_states_[l].beta1_t *= beta1;
            adam_states_[l].beta2_t *= beta2;
            double beta1_t = adam_states_[l].beta1_t;
            double beta2_t = adam_states_[l].beta2_t;

            for (int i = 0; i < out_dim; ++i) {
                double d_act;
                if (layers_[l].use_relu)
                    d_act = (layers_[l].z[i] > 0) ? delta[i] : 0;
                else
                    d_act = delta[i];

                // 先累积 prev_delta（使用更新前的权重）
                for (int j = 0; j < in_dim; ++j)
                    prev_delta[j] += d_act * layers_[l].W[i][j];

                // 再更新偏置
                adam_states_[l].mb[i] = beta1 * adam_states_[l].mb[i] + (1 - beta1) * d_act;
                adam_states_[l].vb[i] = beta2 * adam_states_[l].vb[i] + (1 - beta2) * d_act * d_act;
                double m_hat = adam_states_[l].mb[i] / (1 - beta1_t);
                double v_hat = adam_states_[l].vb[i] / (1 - beta2_t);
                layers_[l].b[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);

                for (int j = 0; j < in_dim; ++j) {
                    double grad_w = d_act * activations[l][j];
                    adam_states_[l].mW[i][j] = beta1 * adam_states_[l].mW[i][j] + (1 - beta1) * grad_w;
                    adam_states_[l].vW[i][j] = beta2 * adam_states_[l].vW[i][j] + (1 - beta2) * grad_w * grad_w;

                    double mw_hat = adam_states_[l].mW[i][j] / (1 - beta1_t);
                    double vw_hat = adam_states_[l].vW[i][j] / (1 - beta2_t);
                    layers_[l].W[i][j] -= lr * mw_hat / (std::sqrt(vw_hat) + eps);
                }
            }

            delta = prev_delta;
        }

        // Decay learning rate
        batch_count_++;
        if (batch_count_ % 10000 == 0) learning_rate_ *= 0.95;
    }

    void set_learning_rate(double lr) { learning_rate_ = lr; }

    bool save_weights(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) return false;
        ofs.write((char*)&scale_factor_, sizeof(scale_factor_));
        int num_layers = (int)layers_.size();
        ofs.write((char*)&num_layers, sizeof(num_layers));
        for (auto& layer : layers_) {
            ofs.write((char*)&layer.input_dim, sizeof(layer.input_dim));
            ofs.write((char*)&layer.output_dim, sizeof(layer.output_dim));
            ofs.write((char*)&layer.use_relu, sizeof(layer.use_relu));
            for (auto& row : layer.W)
                ofs.write((char*)row.data(), sizeof(double) * layer.input_dim);
            ofs.write((char*)layer.b.data(), sizeof(double) * layer.output_dim);
        }
        return true;
    }

    bool load_weights(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        ifs.read((char*)&scale_factor_, sizeof(scale_factor_));
        int num_layers;
        ifs.read((char*)&num_layers, sizeof(num_layers));
        layers_.clear();
        adam_states_.clear();
        for (int l = 0; l < num_layers; ++l) {
            int in_dim, out_dim;
            bool relu;
            ifs.read((char*)&in_dim, sizeof(in_dim));
            ifs.read((char*)&out_dim, sizeof(out_dim));
            ifs.read((char*)&relu, sizeof(relu));
            NNLayer layer(in_dim, out_dim, relu);
            for (auto& row : layer.W)
                ifs.read((char*)row.data(), sizeof(double) * in_dim);
            ifs.read((char*)layer.b.data(), sizeof(double) * out_dim);
            layers_.push_back(std::move(layer));
        }
        input_dim_ = layers_[0].input_dim;
        output_dim_ = layers_.back().output_dim;
        init_adam();
        return true;
    }

    int num_layers() const { return (int)layers_.size(); }
    int input_dim() const { return input_dim_; }

private:
    std::vector<std::vector<double>> gradients_;
};

// ============================================================
// 经验回放缓冲区 (Experience Replay)
// ============================================================
struct Experience {
    BoardFeatures features;
    double value;       // 目标值：从该状态开始的累计未来得分
    int step;           // 此样本在第几步
    int level;
};

class ReplayBuffer {
    std::vector<Experience> buffer_;
    size_t capacity_;
    size_t pos_ = 0;
    std::mt19937 rng_{std::random_device{}()};

public:
    ReplayBuffer(size_t capacity = 100000) : capacity_(capacity) { buffer_.reserve(capacity); }

    void push(const Experience& exp) {
        if (buffer_.size() < capacity_)
            buffer_.push_back(exp);
        else
            buffer_[pos_ % capacity_] = exp;
        pos_++;
    }

    void sample(std::vector<Experience>& batch, size_t batch_size) {
        batch.clear();
        size_t n = buffer_.size();
        if (n == 0) return;
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        for (size_t i = 0; i < batch_size; ++i)
            batch.push_back(buffer_[dist(rng_)]);
    }

    size_t size() const { return buffer_.size(); }
    const std::vector<Experience>& all() const { return buffer_; }

    void save(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) return;
        size_t n = buffer_.size();
        ofs.write((char*)&n, sizeof(n));
        for (auto& e : buffer_) {
            ofs.write((char*)e.features.f, sizeof(double) * NUM_BOARD_FEATURES);
            ofs.write((char*)&e.value, sizeof(e.value));
            ofs.write((char*)&e.step, sizeof(e.step));
            ofs.write((char*)&e.level, sizeof(e.level));
        }
    }

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        size_t n;
        ifs.read((char*)&n, sizeof(n));
        buffer_.clear();
        for (size_t i = 0; i < n; ++i) {
            Experience e;
            ifs.read((char*)e.features.f, sizeof(double) * NUM_BOARD_FEATURES);
            ifs.read((char*)&e.value, sizeof(e.value));
            ifs.read((char*)&e.step, sizeof(e.step));
            ifs.read((char*)&e.level, sizeof(e.level));
            buffer_.push_back(e);
        }
        pos_ = n;
        return true;
    }
};

// ============================================================
// 深度评估器 (DeepEvaluator) — 使用NN替代启发式评估
// ============================================================
class DeepEvaluator {
    DeepPredictor predictor_;
    double heuristic_weight_ = 0.5; // 混合权重：NN评分与启发式评分的比例
    bool enabled_ = false;
    std::string weights_path_ = "deepstar_weights.bin";

public:
    DeepEvaluator() {}

    void enable(bool flag) { enabled_ = flag; }
    bool is_enabled() const { return enabled_; }

    double evaluate(const Board& b) {
        double h_score = board_heuristic_static(b);
        if (!enabled_) return h_score;
        double nn_score = predictor_.predict_board(b);
        return heuristic_weight_ * h_score + (1 - heuristic_weight_) * nn_score;
    }

    double pure_nn_evaluate(const Board& b) {
        return predictor_.predict_board(b);
    }

    DeepPredictor& predictor() { return predictor_; }
    bool save_weights() { return predictor_.save_weights(weights_path_); }
    bool save_weights(const std::string& path) { weights_path_ = path; return predictor_.save_weights(path); }
    bool load_weights(const std::string& path) { weights_path_ = path; return predictor_.load_weights(path); }
    void set_heuristic_weight(double w) { heuristic_weight_ = w; }
    double heuristic_weight() const { return heuristic_weight_; }

    double board_heuristic_static(const Board& b) {
        // 与 allstar.cpp 中 board_heuristic 相同的逻辑
        if (b.is_deadlocked()) return -1e5;
        int N = b.N, cp = 0, sp = 0, wc = 0, bb = 0, cnt[6] = {}, mxc = 0;
        std::vector<std::vector<bool>> vis(N, std::vector<bool>(N));
        for (int r = 0; r < N; ++r) for (int c = 0; c < N; ++c) {
            int co = b.at(r, c).color();
            if (b.at(r, c).is_wildcard()) { wc++; }
            if (b.at(r, c).is_bomb()) bb++;
            cnt[std::min(co, 5)]++;
            if (c + 1 < N) { int c2 = b.at(r, c + 1).color(); if (compatible_color(co, c2)) { cp++; if (co == c2 && co != 0) sp++; } }
            if (r + 1 < N) { int c2 = b.at(r + 1, c).color(); if (compatible_color(co, c2)) { cp++; if (co == c2 && co != 0) sp++; } }
            if (!vis[r][c]) {
                int sz = 0; std::deque<std::pair<int, int>> q;
                vis[r][c] = true; q.push_back({r, c});
                while (!q.empty()) {
                    auto [cr, cc] = q.front(); q.pop_front(); sz++;
                    int cco = b.at(cr, cc).color();
                    for (int d = 0; d < 4; ++d) {
                        int nr = cr + DR[d], nc = cc + DC[d]; if (!b.in_bounds(nr, nc) || vis[nr][nc]) continue;
                        int nco = b.at(nr, nc).color();
                        if (!compatible_color(cco, nco)) continue;
                        vis[nr][nc] = true; q.push_back({nr, nc});
                    }
                }
                mxc = std::max(mxc, sz);
            }
        }
        double ent = 0.0; int tc = N * N - wc;
        if (tc > 0) for (int i = 1; i <= 5; ++i) if (cnt[i] > 0) { double p = (double)cnt[i] / tc; ent -= p * std::log(p); }
        bool is_pure5 = (b.level <= 2);
        bool is_bomb_level = (b.level == 4 || b.level == 5);
        double bb_weight = is_bomb_level ? 22.0 : 15.0;
        return cp * 6.0 + sp * 10.0 + wc * 35.0 + bb * bb_weight + mxc * g_params[b.level].mxc_w - ent * 12.0;
    }
};

// 全局深度评估器
static DeepEvaluator g_deep_eval;

// ============================================================
// 求解器：穷举 DFS + 多步前瞻 + 掉落仿真 + 时间管理
// ============================================================
class ImprovedSolver {
    mutable std::chrono::steady_clock::time_point _t0;
    mutable bool _tinit = false;
    void tstart() const { if(!_tinit){_t0=std::chrono::steady_clock::now();_tinit=true;} }
    long long telapsed() const { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_t0).count(); }
    mutable int step_count_ = 0;
    bool tfast()  const { return telapsed() > g_params[g_cur_level].max_time_ms * g_params[g_cur_level].tfast_ratio; }
    bool tdead()  const { return telapsed() > g_params[g_cur_level].max_time_ms * g_params[g_cur_level].tdead_ratio; }

    static int cur_cache_limit_;
    static int cur_keep1_;
    static int cur_keep2_;
    static int cur_keep3_;
    static double mxc_weight_;
    static double dq_weight_;
    static double bh_weight_;
    static double step3_dq_weight_;
    static double step3_bh_weight_;
    static double surv_mul_;
    static int cur_beam_w_;
    static int cur_beam_d_;

    int cache_limit()const { return tfast()?(int)(cur_cache_limit_*0.65):cur_cache_limit_; }
    int keep1()    const { return tfast()?(int)(cur_keep1_*0.7):cur_keep1_; }
    int keep2()    const { return tfast()?(int)(cur_keep2_*0.7):cur_keep2_; }
    int keep3()    const { return tfast()?(int)(cur_keep3_*0.7):cur_keep3_; }

    mutable int board_N_ = 6;
    mutable std::mt19937 rng{std::random_device{}()};

    static double board_heuristic(const Board& b) {
        return g_deep_eval.evaluate(b);
    }

    static double drop_quality(const Board& b) {
        double q=0;
        auto& dq = *b.drop_queue;
        for (int c=0;c<b.N;++c) {
            int qp=b.queue_ptr[c];
            for (int i=0;i<7;++i) {
                if (qp+i>=1000) break;
                int v=dq[c][qp+i];
                if (v==0) { q+=2.8; continue; }
                if (v<0) { q+=0.9; continue; }
                int col=std::abs(v);
                double adj=0;
                if (b.in_bounds(0,c-1)&&compatible_color(col,b.at(0,c-1).color())) adj+=0.5;
                if (b.in_bounds(0,c+1)&&compatible_color(col,b.at(0,c+1).color())) adj+=0.5;
                if (b.in_bounds(1,c)  &&compatible_color(col,b.at(1,c).color()))   adj+=0.3;
                q+=adj;
                if (i<3) for (int j=i+1;j<3&&j<7;++j) {
                    int v2=dq[c][qp+j]; if (v2==0||v==v2) q+=0.2;
                }
            }
        }
        return q;
    }

    static int quick_best_score(const Board& b) {
        if (b.is_deadlocked()) return 0;
        int best=0;
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            int co=b.at(r,c).color();
            if (b.at(r,c).is_wildcard()) { for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (b.in_bounds(nr,nc)) best=std::max(best,path_score(2)); } continue; }
            if (c+1<b.N&&compatible_color(co,b.at(r,c+1).color())) best=std::max(best,path_score(2));
            if (r+1<b.N&&compatible_color(co,b.at(r+1,c).color())) best=std::max(best,path_score(2));
        }
        return best;
    }

    static int upper_bound_reachable(const Board& b, const std::vector<std::pair<int,int>>& path, int target) {
        std::vector<std::vector<bool>> blocked(b.N, std::vector<bool>(b.N, false));
        for (auto [r,c]:path) blocked[r][c]=true;
        auto [tr,tc]=path.back();
        std::deque<std::pair<int,int>> q;
        std::vector<std::vector<bool>> seen(b.N, std::vector<bool>(b.N, false));
        for (int d=0; d<4; ++d) {
            int nr=tr+DR[d], nc=tc+DC[d];
            if (!b.in_bounds(nr,nc)||blocked[nr][nc]) continue;
            if (!compatible_color(target,b.at(nr,nc).color())) continue;
            seen[nr][nc]=true; q.push_back({nr,nc});
        }
        int reachable=0;
        int black=0, white=0;
        for (auto [r,c]:path) {
            if ((r+c)%2==0) black++; else white++;
        }
        while (!q.empty()) {
            auto [r,c]=q.front(); q.pop_front(); ++reachable;
            if ((r+c)%2==0) black++; else white++;
            for (int d=0; d<4; ++d) {
                int nr=r+DR[d], nc=c+DC[d];
                if (!b.in_bounds(nr,nc)||blocked[nr][nc]||seen[nr][nc]) continue;
                if (!compatible_color(target,b.at(nr,nc).color())) continue;
                seen[nr][nc]=true; q.push_back({nr,nc});
            }
        }
        int ub_reach = (int)path.size()+reachable;
        if (b.level<=2 || b.level==4) {
            int ub_bip = std::min(black,white)*2;
            if (black!=white) ub_bip++;
            return std::min(ub_reach, ub_bip);
        }
        return ub_reach;
    }

    static int exact_best_one_step_score(const Board& b, int max_nodes) {
        int best=0, nodes=0;
        std::vector<std::pair<int,int>> path;
        std::vector<bool> vis(b.N*b.N, false);
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target) {
            if (++nodes>max_nodes) return;
            int co=b.at(r,c).color(); int fixed=target; if (fixed==0&&co!=0) fixed=co;
            if ((int)path.size()>=2) {
                int sc=path_score(b,path); if (sc>best) best=sc;
                int ub=upper_bound_reachable(b,path,fixed);
                if (path_score(ub)<=best) return;
            }
            std::pair<int,int> nb[4]; int nb_cnt=0;
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (!b.in_bounds(nr,nc)||vis[nr*b.N+nc]) continue;
                if (!compatible_color(fixed,b.at(nr,nc).color())) continue;
                nb[nb_cnt++]={nr,nc}; }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc]) continue;
                        if(compatible_color(fixed,b.at(ppr,ppc).color())) pot++;
                    }
                    if (b.level<=2) {
                        int same=0;
                        for(int dd=0;dd<4;++dd){
                            int ppr=pr+DR[dd], ppc=pc+DC[dd];
                            if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc]) continue;
                            if(b.at(ppr,ppc).color()==fixed) same++;
                        }
                        if (b.level==1) return pot*10-same;
                        else return pot*10+same;
                    }
                    int base=0;
                    if (b.at(pr,pc).is_wildcard()) base=40;
                    else if (b.at(pr,pc).is_bomb()) base=(b.level==4?35:25);
                    else if (b.at(pr,pc).color()==fixed) base=20;
                    return base+pot;
                };
                if(b.level<=2) { if(eval(nb[j])<eval(nb[i])) std::swap(nb[i],nb[j]); }
                else { if(eval(nb[j])>eval(nb[i])) std::swap(nb[i],nb[j]); }
            }
            for (int i=0;i<nb_cnt;++i) { auto [nr,nc]=nb[i]; path.push_back({nr,nc}); vis[nr*b.N+nc]=true; dfs(nr,nc,fixed); vis[nr*b.N+nc]=false; path.pop_back(); if (nodes>max_nodes) return; }
        };
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            path.clear(); path.push_back({r,c}); vis.assign(b.N*b.N,false); vis[r*b.N+c]=true; dfs(r,c,0); if (nodes>max_nodes) break;
        }
        return best;
    }

    static std::pair<int,std::vector<std::pair<int,int>>>
    exact_best_with_path(const Board& b, int max_nodes) {
        int best_score=0, nodes=0;
        std::vector<std::pair<int,int>> best_path, path;
        std::vector<bool> vis(b.N*b.N, false);
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target) {
            if (++nodes>max_nodes) return;
            int co=b.at(r,c).color(); int fixed=target; if (fixed==0&&co!=0) fixed=co;
            if ((int)path.size()>=2) {
                int sc=path_score(b,path);
                if (sc>best_score||(sc==best_score&&path<best_path)) { best_score=sc; best_path=path; }
                int ub=upper_bound_reachable(b,path,fixed);
                if (path_score(ub)<=best_score) return;
            }
            std::pair<int,int> nb[4]; int nb_cnt=0;
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (!b.in_bounds(nr,nc)||vis[nr*b.N+nc]) continue;
                if (!compatible_color(fixed,b.at(nr,nc).color())) continue;
                nb[nb_cnt++]={nr,nc}; }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc]) continue;
                        if(compatible_color(fixed,b.at(ppr,ppc).color())) pot++;
                    }
                    if (b.level<=2) {
                        int same=0;
                        for(int dd=0;dd<4;++dd){
                            int ppr=pr+DR[dd], ppc=pc+DC[dd];
                            if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc]) continue;
                            if(b.at(ppr,ppc).color()==fixed) same++;
                        }
                        if (b.level==1) return pot*10-same;
                        else return pot*10+same;
                    }
                    int base=0;
                    if (b.at(pr,pc).is_wildcard()) base=40;
                    else if (b.at(pr,pc).is_bomb()) base=(b.level==4?35:25);
                    else if (b.at(pr,pc).color()==fixed) base=20;
                    return base+pot;
                };
                if(b.level<=2) { if(eval(nb[j])<eval(nb[i])) std::swap(nb[i],nb[j]); }
                else { if(eval(nb[j])>eval(nb[i])) std::swap(nb[i],nb[j]); }
            }
            for (int i=0;i<nb_cnt;++i) { auto [nr,nc]=nb[i]; path.push_back({nr,nc}); vis[nr*b.N+nc]=true; dfs(nr,nc,fixed); vis[nr*b.N+nc]=false; path.pop_back(); if (nodes>max_nodes) return; }
        };
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            path.clear(); path.push_back({r,c}); vis.assign(b.N*b.N,false); vis[r*b.N+c]=true; dfs(r,c,0); if (nodes>max_nodes) break;
        }
        return {best_score,best_path};
    }

    double simulate_drops_and_score(const Board& b, int sim_steps) const {
        Board sim=b;
        double total=0;
        for (int st=0;st<sim_steps;++st) {
            if (sim.is_deadlocked()) break;
            auto [sc,bp]=exact_best_with_path(sim,std::max(4000,cache_limit()/sim_steps));
            if (bp.size()<2) break;
            total+=(double)sc*std::pow(0.7,(double)st);
            sim=sim.preview(bp);
        }
        return total;
    }

    int survival_steps(const Board& b, int max_check) const {
        Board sim=b; int i;
        for (i=0;i<max_check;++i) {
            if (sim.is_deadlocked()) break;
            auto [sc,bp]=exact_best_with_path(sim,tdead()?5000:20000);
            if (bp.size()<2) { i++; break; }
            sim=sim.preview(bp);
        }
        return i;
    }

    double beam_evaluate(const Board& start_b, int beam_w, int beam_d) const {
        struct BNode { Board bd; double acc; };
        std::vector<BNode> beams; beams.push_back({start_b,0.0});
        for (int d=0;d<beam_d;++d) {
            struct Cand { Board bd; double acc; double val; };
            std::vector<Cand> cands;
            for (auto& bm:beams) {
                auto [sc,bp]=exact_best_with_path(bm.bd,d==beam_d-1?10000:5000);
                if (bp.size()<2) continue;
                Board nb=bm.bd.preview(bp);
                double hv=board_heuristic(nb);
                double total=bm.acc+(double)sc+hv*(d==beam_d-1?0.5:0.15);
                cands.push_back({std::move(nb),bm.acc+(double)sc,total});
                for (int tries=0;tries<4&&(int)cands.size()<beam_w*4;++tries) {
                    auto [sc2,bp2]=exact_best_with_path(bm.bd,2000+tries*1000);
                    if (bp2.size()<2||(bp2[0]==bp[0]&&bp2.size()==bp.size())) continue;
                    Board nb2=bm.bd.preview(bp2);
                    double hv2=board_heuristic(nb2);
                    double total2=bm.acc+(double)sc2+hv2*(d==beam_d-1?0.5:0.15);
                    cands.push_back({std::move(nb2),bm.acc+(double)sc2,total2});
                }
            }
            if (cands.empty()) break;
            std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.val>b.val;});
            beams.clear();
            for (int i=0;i<std::min(beam_w,(int)cands.size());++i) beams.push_back({std::move(cands[i].bd),cands[i].acc});
        }
        return beams.empty()?0.0:beams[0].acc;
    }

    static std::vector<std::pair<int,int>> fallback(const Board& b) {
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            if (b.at(r,c).is_wildcard()) for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (b.in_bounds(nr,nc)) return {{r,c},{nr,nc}}; }
            int a=b.at(r,c).color();
            if (c+1<b.N&&compatible_color(a,b.at(r,c+1).color())) return {{r,c},{r,c+1}};
            if (r+1<b.N&&compatible_color(a,b.at(r+1,c).color())) return {{r,c},{r+1,c}};
        }
        return {{0,0},{0,1}};
    }

public:
    std::vector<std::pair<int,int>> solve(const Board& board) {
        _tinit = false;
        tstart();
        ++step_count_;
        board_N_ = board.N;
        g_cur_level = board.level;
        bool pure5 = (board.level <= 2);
        bool bomb_level = (board.level == 4);
        bool mixed_level = (board.level == 3 || board.level == 5);
        int lmt = g_params[board.level].dfs_limit;
        int cmax = cache_limit();

        auto& p = g_params[board.level];
        cur_cache_limit_ = p.cache_limit; cur_keep1_ = p.keep1;
        cur_keep2_ = p.keep2; cur_keep3_ = p.keep3;
        mxc_weight_ = p.mxc_w; dq_weight_ = p.dq_w; bh_weight_ = p.bh_w;
        step3_dq_weight_ = p.s3_dq_w; step3_bh_weight_ = p.s3_bh_w;
        surv_mul_ = p.surv_mul; cur_beam_w_ = p.beam_w; cur_beam_d_ = p.beam_d;

        struct Raw { std::vector<std::pair<int,int>> path; int now; int nxt; };
        std::vector<Raw> raws; raws.reserve(8192);

        static std::unordered_map<std::uint64_t,int> cache2;
        int nodes=0;
        std::vector<std::pair<int,int>> path;

        std::vector<bool> vis_main(board.N*board.N, false);
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target) {
            if (++nodes>lmt) return;
            int cur=board.at(r,c).color(); int fixed=target; if (fixed==0&&cur!=0) fixed=cur;
            if ((int)path.size()>=2) {
                int ns=path_score(board,path);
                Board nb=board.preview(path);
                auto h=board_hash(nb);
                int nxt=0; auto it=cache2.find(h); if (it!=cache2.end()) nxt=it->second;
                else { nxt=exact_best_one_step_score(nb,cmax); cache2[h]=nxt; }
                raws.push_back({path,ns,nxt});
            }
            std::pair<int,int> nb_arr[4]; int nb_cnt=0;
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (!board.in_bounds(nr,nc)||vis_main[nr*board.N+nc]) continue;
                if (!compatible_color(fixed,board.at(nr,nc).color())) continue;
                nb_arr[nb_cnt++]={nr,nc}; }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!board.in_bounds(ppr,ppc)||vis_main[ppr*board.N+ppc]) continue;
                        if(compatible_color(fixed,board.at(ppr,ppc).color())) pot++;
                    }
                    if (pure5) return pot;
                    int base=(board.at(pr,pc).is_wildcard()?40:(board.at(pr,pc).is_bomb()?35:(board.at(pr,pc).color()==fixed?20:0)));
                    return base+pot;
                };
                if(pure5) { if(eval(nb_arr[j])<eval(nb_arr[i])) std::swap(nb_arr[i],nb_arr[j]); }
                else { if(eval(nb_arr[j])>eval(nb_arr[i])) std::swap(nb_arr[i],nb_arr[j]); }
            }
            for (int i=0;i<nb_cnt;++i) { auto [nr,nc]=nb_arr[i]; path.push_back({nr,nc}); vis_main[nr*board.N+nc]=true; dfs(nr,nc,fixed); vis_main[nr*board.N+nc]=false; path.pop_back(); if (nodes>lmt) return; }
        };
        std::vector<std::pair<int,int>> starts;
        if (pure5) {
            std::vector<std::vector<int>> cc_size(board.N, std::vector<int>(board.N, 0));
            std::vector<std::vector<bool>> vis(board.N, std::vector<bool>(board.N, false));
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) {
                if (vis[r][c]) continue;
                int co = board.at(r,c).color();
                std::vector<std::pair<int,int>> cells;
                std::deque<std::pair<int,int>> q;
                q.push_back({r,c}); vis[r][c]=true;
                while (!q.empty()) { auto [cr,cc]=q.front(); q.pop_front(); cells.push_back({cr,cc});
                    for (int d=0;d<4;++d) { int nr=cr+DR[d],nc=cc+DC[d];
                        if (!board.in_bounds(nr,nc)||vis[nr][nc]) continue;
                        if (board.at(nr,nc).color()!=co) continue;
                        vis[nr][nc]=true; q.push_back({nr,nc}); }
                }
                for (auto& p:cells) cc_size[p.first][p.second]=(int)cells.size();
            }
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c)
                starts.push_back({r,c});
            if (board.level==1) {
                std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){ return cc_size[a.first][a.second]>cc_size[b.first][b.second]; });
            } else {
                std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){ return cc_size[a.first][a.second]<cc_size[b.first][b.second]; });
            }
        } else {
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) starts.push_back({r,c});
        }

        for (auto [r,c] : starts) {
            path.clear(); path.push_back({r,c}); vis_main.assign(board.N*board.N,false); vis_main[r*board.N+c]=true; dfs(r,c,0); if (nodes>lmt) break;
        }
        if (raws.empty()) {
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) {
                path.clear(); path.push_back({r,c}); vis_main.assign(board.N*board.N,false); vis_main[r*board.N+c]=true; dfs(r,c,0); if (nodes>lmt) break;
            }
        }
        if (raws.empty()) return fallback(board);

        std::vector<int> order(raws.size()); for (int i=0;i<(int)raws.size();++i) order[i]=i;
        std::sort(order.begin(),order.end(),[&](int a,int b) {
            int va=raws[a].now+raws[a].nxt, vb=raws[b].now+raws[b].nxt;
            if (pure5) {
                int pa=(int)raws[a].path.size(), pb=(int)raws[b].path.size();
                int sp = g_params[board.level].short_pen;
                if (pa<6) va-=sp;
                if (pb<6) vb-=sp;
            }
            return va!=vb?va>vb:raws[a].now>raws[b].now; });
        int k1=std::min(keep1(),(int)raws.size()); order.resize(k1);

        struct Cand { std::vector<std::pair<int,int>> path; int now,nxt; Board next_b; double v2; };
        std::vector<Cand> cands; cands.reserve(k1);
        for (int idx:order) {
            Board nb=board.preview(raws[idx].path);
            double dq=drop_quality(nb);
            double bh=board_heuristic(nb);
            double v2=(double)raws[idx].now+(double)raws[idx].nxt+dq*dq_weight_+bh*bh_weight_;
            cands.push_back({raws[idx].path,raws[idx].now,raws[idx].nxt,std::move(nb),v2});
        }
        std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.v2>b.v2;});
        if (tdead()) return cands[0].path;

        static std::unordered_map<std::uint64_t,int> cache3;
        int k2=std::min(keep2(),(int)cands.size());
        struct Scored { double val; int idx; };
        std::vector<Scored> scored; scored.reserve(k1);
        for (int i=0;i<k1;++i) {
            double val=cands[i].v2;
            if (i<k2&&!tdead()) {
                auto [nsc,np]=exact_best_with_path(cands[i].next_b,cmax);
                if (np.size()>=2) {
                    Board tb=cands[i].next_b.preview(np); auto th=board_hash(tb);
                    int tsc=0; auto it=cache3.find(th); if (it!=cache3.end()) tsc=it->second;
                    else { tsc=exact_best_one_step_score(tb,cmax/3); cache3[th]=tsc; }
                    val=(double)cands[i].now+(double)nsc+(double)tsc+drop_quality(tb)*step3_dq_weight_+board_heuristic(tb)*step3_bh_weight_;
                }
            }
            scored.push_back({val,i});
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        if (tdead()) return cands[scored[0].idx].path;

        int k3=std::min(keep3(),(int)scored.size());
        for (int j=0;j<k3;++j) {
            int st=survival_steps(cands[scored[j].idx].next_b,g_params[board.level].surv_check);
            if (st<=2) scored[j].val-=120.0*surv_mul_*(3.0-(double)st);
            else if (st<=3) scored[j].val-=50.0*surv_mul_;
            else if (st<=4) scored[j].val-=15.0*surv_mul_;
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});

        if (!tfast() && cur_beam_w_ > 0) {
            int bs_cnt=std::min(3,(int)scored.size());
            for (int j=0;j<bs_cnt;++j) {
                double beam_val=beam_evaluate(cands[scored[j].idx].next_b,cur_beam_w_,cur_beam_d_);
                scored[j].val+=beam_val*g_params[board.level].beam_bonus;
            }
            std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        }

        int sel = 0;

        if (cache2.size()>400000) cache2.clear();
        if (cache3.size()>300000) cache3.clear();
        return cands[scored[sel].idx].path;
    }
};

int ImprovedSolver::cur_cache_limit_ = 60000;
int ImprovedSolver::cur_keep1_ = 200;
int ImprovedSolver::cur_keep2_ = 100;
int ImprovedSolver::cur_keep3_ = 20;
double ImprovedSolver::mxc_weight_ = 15.0;
double ImprovedSolver::dq_weight_ = 0.8;
double ImprovedSolver::bh_weight_ = 0.02;
double ImprovedSolver::step3_dq_weight_ = 0.55;
double ImprovedSolver::step3_bh_weight_ = 0.025;
double ImprovedSolver::surv_mul_ = 1.0;
int ImprovedSolver::cur_beam_w_ = 8;
int ImprovedSolver::cur_beam_d_ = 4;

// ============================================================
// 协议交互层：GameController
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
        b.drop_queue = std::make_shared<std::vector<std::vector<int>>>();
        b.drop_queue->assign(N, std::vector<int>(1000));
        b.queue_ptr.assign(N, 0);
        for (int c = 0; c < N; ++c) {
            for (int i = 0; i < 1000; ++i) (*b.drop_queue)[c][i] = gen_block(rng, level);
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
                for (int dr = -1; dr <= 1; ++dr)
                    for (int dc = -1; dc <= 1; ++dc) {
                        int nr = r + dr, nc = c + dc;
                        if (b.in_bounds(nr, nc) && !in_path[nr][nc]) to_remove[nr][nc] = true;
                    }
            }
        }
        for (int c = 0; c < b.N; ++c) {
            int cnt = 0;
            for (int r = 0; r < b.N; ++r) if (to_remove[r][c]) ++cnt;
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
// 本地模拟评测机
// ============================================================
class LocalJudge {
    int level_;
    int N_;
    Board board_;
    int step_ = 0;
    int score_ = 0;
    int invalid_streak_ = 0;
    bool done_ = false;
    static constexpr int MAX_STEPS = 50;
    int seed_;

    static int gen_block(std::mt19937& rng, int level) {
        if (level<=2) return (rng()%5)+1;
        if (level==3) return ((rng()%100)<15)?0:(rng()%5)+1;
        if (level==4) { int c=(rng()%5)+1; return ((rng()%100)<10)?-c:c; }
        if ((rng()%100)<15) return 0;
        int base=(rng()%5)+1; return ((rng()%100)<10)?-base:base;
    }

public:
    LocalJudge(int level, int seed, int N) : level_(level), N_(N), seed_(seed) {
        board_ = Board(N);
        board_.level = level;
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
    int seed() const { return seed_; }

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
                else if (anchor != color) { reason = "color mismatch"; return false; }
            }
            if (i > 0) {
                auto [pr, pc] = path[i-1];
                if (std::abs(pr - r) + std::abs(pc - c) != 1) { reason = "not 4-connected"; return false; }
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
            invalid_streak_++;
            step_++;
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
                        if (nr >= 0 && nr < N_ && nc >= 0 && nc < N_ && !in_path[nr][nc] && !exploded[nr][nc]) {
                            exploded[nr][nc] = true;
                            gained += 10;
                        }
                    }
            }
        }

        score_ += gained;
        step_++;

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

    void print_board(std::ostream& out) const {
        for (int r = 0; r < N_; ++r) {
            for (int c = 0; c < N_; ++c) {
                int v = board_.at(r, c).value;
                if (v >= 0) out << ' ';
                out << v << ' ';
            }
            out << "\n";
        }
    }
};

// ============================================================
// 自弈训练器 (Self-Play Trainer)
// ============================================================
class SelfPlayTrainer {
    ReplayBuffer buffer_;
    ImprovedSolver solver_;
    int n_games_ = 0;
    int total_training_steps_ = 0;

public:
    SelfPlayTrainer() : buffer_(100000) {}

    // 单局自弈，收集训练样本（蒙特卡洛回报标注）
    int play_one_game(int level, int seed, int N, bool collect) {
        LocalJudge judge(level, seed, N);
        int n_steps = 0;

        // 临时存储本局状态及步得分
        std::vector<BoardFeatures> state_features;
        std::vector<double> step_rewards;
        double cumulative = 0;

        while (!judge.done() && n_steps < 50) {
            auto path = solver_.solve(judge.board());
            if (path.size() < 2) path = {{0, 0}, {0, 1}};

            int prev_score = judge.score();
            if (collect) {
                state_features.push_back(extract_features(judge.board()));
            }

            bool cont = judge.play(path);
            int gained = judge.score() - prev_score;
            if (collect) {
                step_rewards.push_back((double)gained);
            }
            cumulative += gained;

            if (!cont) break;
            n_steps++;
        }

        // 蒙特卡洛回标注：从每步开始到终局的累计回报
        if (collect && !state_features.empty()) {
            const double discount = 0.95;
            double future_return = 0;
            for (int i = (int)step_rewards.size() - 1; i >= 0; --i) {
                future_return = step_rewards[i] + discount * future_return;
                Experience exp;
                exp.features = state_features[i];
                exp.step = i;
                exp.level = level;
                exp.value = future_return;
                buffer_.push(exp);
            }
        }

        n_games_++;
        return judge.score();
    }

    // 多种子批量自弈并生成训练样本
    void generate_training_data(const std::vector<int>& seeds, int n_epochs = 1) {
        std::cout << "[Trainer] Generating training data across " << seeds.size()
                  << " seeds, " << n_epochs << " epochs\n";
        int old_size = (int)buffer_.size();

        struct LevelConfig { int level; int N; };
        std::vector<LevelConfig> levels = {
            {1, 10}, {2, 10}, {3, 10}, {4, 10}, {5, 12}
        };

        for (int ep = 0; ep < n_epochs; ++ep) {
            int completed = 0;
            for (int seed : seeds) {
                for (auto& lc : levels) {
                    int score = play_one_game(lc.level, seed, lc.N, true);
                    completed++;
                }
            }
            std::cout << "[Trainer] Epoch " << ep + 1 << " complete, samples: "
                      << buffer_.size() << " (+" << (buffer_.size() - old_size) << ")\n";
            old_size = (int)buffer_.size();
        }
    }

    // 从经验库训练神经网络
    void train_network(DeepPredictor& predictor, int n_iterations, int batch_size = 32) {
        if (buffer_.size() < (size_t)batch_size) {
            std::cerr << "[Train] Not enough samples: " << buffer_.size() << " < " << batch_size << "\n";
            return;
        }

        std::cout << "[Train] Training neural network for " << n_iterations
                  << " iterations, batch_size=" << batch_size
                  << ", samples=" << buffer_.size() << "\n";

        std::vector<Experience> batch;
        double total_loss = 0;
        int valid_updates = 0;

        for (int iter = 0; iter < n_iterations; ++iter) {
            buffer_.sample(batch, batch_size);

            for (auto& exp : batch) {
                if (exp.value != exp.value) continue; // skip NaN targets
                predictor.sgd_update(exp.features, exp.value);
                double pred = predictor.predict_raw(exp.features);
                double norm_target = exp.value / std::max(1.0, predictor.scale_factor());
                if (norm_target > 1.0) norm_target = 1.0;
                if (norm_target < -1.0) norm_target = -1.0;
                double err = pred - norm_target;
                total_loss += err * err;
                valid_updates++;
                total_training_steps_++;
            }

            if (iter % 1000 == 999 || iter == n_iterations - 1) {
                double avg_loss = valid_updates > 0 ? total_loss / valid_updates : 0;
                std::cout << "[Train] Iter " << (iter + 1) << " avg_loss=" << avg_loss << "\n";
                total_loss = 0;
            }
        }
    }

    // 为现有样本重新标注目标值（基于完整游戏结果）
    void relabel_from_game(int game_final_score) {
        // 这是一个简化实现 - 为最近一局游戏的所有样本设置目标值
        double avg_score_per_step = game_final_score > 0 ? (double)game_final_score / 50 : 0;
        // 后处理已存储在buffer中的样本
    }

    const ReplayBuffer& buffer() const { return buffer_; }
    ReplayBuffer& buffer() { return buffer_; }
    int n_games() const { return n_games_; }
    int total_steps() const { return total_training_steps_; }
};

// ============================================================
// 主入口：支持训练 / 评测 / 交互 / 综合模式
// ============================================================
static void print_usage(const char* prog) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << prog << " train [n_seeds] [n_epochs] [n_iterations]     - 自弈训练神经网络\n";
    std::cerr << "  " << prog << " eval <seed> [param_file]           - 评估指定种子\n";
    std::cerr << "  " << prog << " nn-eval <seed> [param_file] [weights] - 带NN增强的评估\n";
    std::cerr << "  " << prog << " play [param_file]          - 交互模式 (GameController)\n";
    std::cerr << "  " << prog << " stats <seed> [param_file]  - 打印盘面统计特征\n";
    std::cerr << "  " << prog << " weights save [path]        - 保存网络权重\n";
    std::cerr << "  " << prog << " weights load [path]        - 加载网络权重\n";
    std::cerr << "  " << prog << " help                       - 显示此帮助\n";
}

static int run_training(int argc, char** argv) {
    int n_seeds = (argc > 2) ? std::atoi(argv[2]) : 20;
    int n_epochs = (argc > 3) ? std::atoi(argv[3]) : 1;
    int n_iterations = (argc > 4) ? std::atoi(argv[4]) : 5000;

    init_default_params();
    std::cout << "[DeepStar] Training mode: seeds=" << n_seeds
              << " epochs=" << n_epochs << " iterations=" << n_iterations << "\n";

    // 生成多个随机种子
    std::vector<int> seeds;
    std::mt19937 rng(42);
    for (int i = 0; i < n_seeds; ++i)
        seeds.push_back((int)rng() % 1000000 + 1);

    SelfPlayTrainer trainer;
    DeepPredictor& predictor = g_deep_eval.predictor();

    // 收集训练数据
    trainer.generate_training_data(seeds, n_epochs);

    // 训练
    trainer.train_network(predictor, n_iterations);

    // 保存权重
    g_deep_eval.save_weights();
    std::cout << "[DeepStar] Weights saved to deepstar_weights.bin\n";
    g_deep_eval.enable(true);

    // 验证训练效果
    std::cout << "[DeepStar] Verifying on seed 42...\n";
    LocalJudge judge(1, 42, 10);
    ImprovedSolver solver;
    int score = 0;
    while (!judge.done()) {
        auto path = solver.solve(judge.board());
        if (path.size() < 2) path = {{0, 0}, {0, 1}};
        judge.play(path);
        score = judge.score();
    }
    std::cout << "[DeepStar] Level 1 seed 42 score: " << score << "\n";

    return 0;
}

static int run_eval(int argc, char** argv) {
    int seed = (argc > 2) ? std::atoi(argv[2]) : 114514;
    const char* param_file = (argc > 3) ? argv[3] : nullptr;

    init_default_params();
    if (param_file) {
        load_params(param_file);
        std::cout << "[DeepStar] Loaded params from " << param_file << "\n";
    }
    // 如果有训练好的权重，自动加载
    if (g_deep_eval.load_weights("deepstar_weights.bin")) {
        g_deep_eval.enable(true);
    }

    struct LevelConfig { int level; int N; const char* name; };
    std::vector<LevelConfig> levels = {
        {1, 10, "Level 1 (10x10)"}, {2, 10, "Level 2 (10x10)"},
        {3, 10, "Level 3 (10x10)"}, {4, 10, "Level 4 (10x10)"},
        {5, 12, "Level 5 (12x12)"},
    };

    ImprovedSolver solver;
    int total_score = 0;

    for (auto& lc : levels) {
        LocalJudge judge(lc.level, seed, lc.N);
        while (!judge.done()) {
            auto path = solver.solve(judge.board());
            if (path.size() < 2) path = {{0,0},{0,1}};
            judge.play(path);
        }
        std::cout << "LEVEL " << lc.level << " SCORE: " << judge.score() << "\n";
        total_score += judge.score();
        std::cout << lc.name << " [" << judge.score() << " pts in " << judge.step() << " steps]\n";
    }
    std::cout << "TOTAL SCORE: " << total_score << "\n";
    return 0;
}

static int run_play(int argc, char** argv) {
    const char* param_file = (argc > 2) ? argv[2] : nullptr;

    init_default_params();
    if (param_file) load_params(param_file);

    GameController gc;
    ImprovedSolver solver;

    while (gc.update()) {
        auto path = solver.solve(gc.board());
        gc.respond(path);
    }
    return 0;
}

static int run_stats(int argc, char** argv) {
    int seed = (argc > 2) ? std::atoi(argv[2]) : 42;
    const char* param_file = (argc > 3) ? argv[3] : nullptr;

    init_default_params();
    if (param_file) load_params(param_file);

    std::vector<int> Ns = {10, 10, 10, 10, 12};
    for (int lv = 1; lv <= 5; ++lv) {
        LocalJudge judge(lv, seed, Ns[lv-1]);
        auto bf = extract_features(judge.board());

        std::cout << "\n=== Level " << lv << " Board Statistics ===\n";
        std::cout << "Initial board:\n";
        judge.print_board(std::cout);

        std::cout << "Features:\n";
        const char* feat_names[] = {
            "color_0_ratio", "color_1_ratio", "color_2_ratio", "color_3_ratio", "color_4_ratio", "color_5_ratio",
            "wildcard_ratio", "bomb_ratio",
            "adj_same_pairs", "adj_compat_pairs", "adj_same_density", "adj_compat_density",
            "max_comp_ratio", "avg_comp_ratio", "comp_std_ratio", "comp_density",
            "comp_size_1", "comp_size_2_3", "comp_size_4_6", "comp_size_7_10", "comp_size_11p",
            "entropy",
            "clr1_density", "clr2_density", "clr3_density", "clr4_density", "clr5_density",
            "clr1_spatial", "clr2_spatial", "clr3_spatial", "clr4_spatial", "clr5_spatial",
            "row_diversity", "col_diversity",
            "deadlock_flag",
            "dq_compat", "dq_wc_expect", "dq_bomb_expect",
            "black_ratio", "white_ratio", "bw_total_ratio", "bw_balance",
            "pos_entropy",
            "bip_clr1", "bip_clr2", "bip_clr3", "bip_clr4", "bip_clr5",
            "level_norm", "size_norm",
            "row_dominant", "col_dominant", "comp_ratio"
        };

        for (int i = 0; i < NUM_BOARD_FEATURES; ++i)
            std::cout << "  f[" << std::setw(2) << i << "] " << std::setw(18) << feat_names[i]
                      << " = " << bf[i] << "\n";

        // NN 预测
        double nn_pred = g_deep_eval.pure_nn_evaluate(judge.board());
        double h_pred = g_deep_eval.board_heuristic_static(judge.board());
        std::cout << "\n  Heuristic score: " << h_pred << "\n";
        std::cout << "  NN predicted score: " << nn_pred << "\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    if (argc < 2) {
        // 默认：带NN增强评估指定种子
        const char* default_seed = "114514";
        char* fake_argv[] = {argv[0], (char*)"eval", (char*)default_seed, nullptr};
        return run_eval(3, fake_argv);
    }

    std::string cmd(argv[1]);

    if (cmd == "train") {
        return run_training(argc, argv);
    } else if (cmd == "nn-eval") {
        // 加载权重并启用NN评估
        const char* weight_file = (argc > 4) ? argv[4] : "deepstar_weights.bin";
        if (g_deep_eval.load_weights(weight_file)) {
            g_deep_eval.enable(true);
            std::cerr << "[DeepStar] NN enabled\n";
        }
        return run_eval(argc, argv);
    } else if (cmd == "eval") {
        return run_eval(argc, argv);
    } else if (cmd == "play") {
        return run_play(argc, argv);
    } else if (cmd == "stats") {
        return run_stats(argc, argv);
    } else if (cmd == "weights") {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " weights <save|load> [path]\n";
            return 1;
        }
        std::string sub(argv[2]);
        if (sub == "save") {
            std::string path = (argc > 3) ? argv[3] : "deepstar_weights.bin";
            if (g_deep_eval.save_weights(path))
                std::cout << "Weights saved to " << path << "\n";
            else std::cerr << "Failed to save weights\n";
        } else if (sub == "load") {
            std::string path = (argc > 3) ? argv[3] : "deepstar_weights.bin";
            if (g_deep_eval.load_weights(path)) {
                std::cout << "Weights loaded from " << path << "\n";
                g_deep_eval.enable(true);
            } else std::cerr << "Failed to load weights from " << path << "\n";
        } else {
            std::cerr << "Unknown weights command: " << sub << "\n";
            return 1;
        }
    } else if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        print_usage(argv[0]);
    } else {
        // 尝试解析为种子号
        int seed = std::atoi(argv[1]);
        if (seed > 0 && seed < 100000000) {
            const char* param_file = (argc > 2) ? argv[2] : nullptr;
            init_default_params();
            if (param_file) load_params(param_file);

            struct LevelConfig { int level; int N; const char* name; };
            LevelConfig levels[] = {
                {1, 10, "Level 1"}, {2, 10, "Level 2"}, {3, 10, "Level 3"},
                {4, 10, "Level 4"}, {5, 12, "Level 5"},
            };
            ImprovedSolver solver;
            int total = 0;
            for (auto& lc : levels) {
                LocalJudge judge(lc.level, seed, lc.N);
                while (!judge.done()) {
                    auto path = solver.solve(judge.board());
                    if (path.size() < 2) path = {{0,0},{0,1}};
                    judge.play(path);
                }
                total += judge.score();
            }
            std::cout << total << "\n";
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    return 0;
}
