#ifndef BOARD_FEATURES_H
#define BOARD_FEATURES_H

#include <vector>
#include <deque>
#include <set>
#include <cmath>
#include <cstdint>

constexpr int NUM_FEATURES = 52;

struct BoardFeatures {
    double f[NUM_FEATURES];
    BoardFeatures() { for(int i=0;i<NUM_FEATURES;++i) f[i]=0.0; }
    double operator[](int i) const { return f[i]; }
    double& operator[](int i) { return f[i]; }
};

// 前向声明
static bool bf_compat(int a, int b) { return a==0||b==0||a==b; }

template<typename BoardT>
static BoardFeatures extract_features(const BoardT& b) {
    constexpr int DR[]={-1,1,0,0};
    constexpr int DC[]={0,0,-1,1};

    BoardFeatures bf;
    int N = b.N, total = N*N;
    int color_cnt[6] = {0}, bomb_cnt = 0, wc_cnt = 0;
    int adj_same = 0, adj_compat = 0;
    double pos_sum[5] = {0};

    std::vector<std::vector<bool>> vis(N, std::vector<bool>(N, false));
    std::vector<int> comp_sizes;
    int max_comp = 0;
    int comp_bin[5] = {0};

    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            int val = b.at(r,c).value;
            int co = std::abs(val);
            if (val == 0) { wc_cnt++; }
            if (val < 0) { bomb_cnt++; color_cnt[std::abs(val)]++; }
            else if (val > 0) {
                color_cnt[val]++;
                pos_sum[val-1] += r * N + c;
            }

            if (vis[r][c]) continue;
            int sz = 0;
            std::deque<std::pair<int,int>> q;
            q.push_back({r,c}); vis[r][c] = true;
            while (!q.empty()) {
                auto p = q.front(); q.pop_front(); sz++;
                int cr = p.first, cc = p.second;
                int cco = std::abs(b.at(cr,cc).value);
                for (int d = 0; d < 4; ++d) {
                    int nr = cr + DR[d], nc = cc + DC[d];
                    if (nr<0||nr>=N||nc<0||nc>=N||vis[nr][nc]) continue;
                    if (!bf_compat(cco, std::abs(b.at(nr,nc).value))) continue;
                    vis[nr][nc] = true;
                    q.push_back(std::make_pair(nr,nc));
                }
            }
            comp_sizes.push_back(sz);
            max_comp = std::max(max_comp, sz);
            if (sz == 1) comp_bin[0]++;
            else if (sz <= 3) comp_bin[1]++;
            else if (sz <= 6) comp_bin[2]++;
            else if (sz <= 10) comp_bin[3]++;
            else comp_bin[4]++;
        }
    }

    // 邻接对
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            int ac = std::abs(b.at(r,c).value);
            if (c+1 < N) { int bc = std::abs(b.at(r,c+1).value); if (bf_compat(ac,bc)) { adj_compat++; if (ac==bc&&ac!=0) adj_same++; } }
            if (r+1 < N) { int bc = std::abs(b.at(r+1,c).value); if (bf_compat(ac,bc)) { adj_compat++; if (ac==bc&&ac!=0) adj_same++; } }
        }

    // 颜色熵
    double ent = 0;
    int tc = total - wc_cnt;
    for (int i = 1; i <= 5; ++i)
        if (color_cnt[i] > 0 && tc > 0) {
            double p = (double)color_cnt[i] / tc;
            ent -= p * std::log(p);
        }

    // 分量统计
    double avg_comp = 0;
    if (!comp_sizes.empty()) {
        double sum = 0;
        for (int sz : comp_sizes) sum += sz;
        avg_comp = sum / comp_sizes.size();
    }

    double var_comp = 0;
    if (!comp_sizes.empty()) {
        for (int sz : comp_sizes) var_comp += (sz - avg_comp) * (sz - avg_comp);
        var_comp /= comp_sizes.size();
    }

    // 掉落队列质量
    double dq_compat = 0, dq_wc = 0, dq_bomb = 0;
    if (b.drop_queue) {
        auto& dq = *b.drop_queue;
        for (int c = 0; c < N; ++c) {
            int qp = b.queue_ptr[c];
            for (int i = 0; i < 10; ++i) {
                if (qp + i >= 1000) break;
                int v = dq[c][qp+i];
                if (v == 0) { dq_wc += 1.0/N/10; continue; }
                if (v < 0) { dq_bomb += 1.0/N/10; continue; }
                int col = std::abs(v);
                for (int d = 0; d < 4; ++d) {
                    int nr = DR[d], nc = c + DC[d];
                    if (b.in_bounds(nr,nc) && bf_compat(col, std::abs(b.at(nr,nc).value)))
                        dq_compat += 0.1 / N / 10;
                }
            }
        }
    }

    // 黑白格分析
    int black = 0, white = 0;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c)
            if (std::abs(b.at(r,c).value) > 0) {
                if ((r+c)%2 == 0) black++; else white++;
            }

    // 行列多样性
    double row_div = 0, col_div = 0;
    for (int r = 0; r < N; ++r) {
        std::set<int> s;
        for (int c = 0; c < N; ++c) { int co = std::abs(b.at(r,c).value); if (co>0) s.insert(co); }
        row_div += (double)s.size() / N;
    }
    for (int c = 0; c < N; ++c) {
        std::set<int> s;
        for (int r = 0; r < N; ++r) { int co = std::abs(b.at(r,c).value); if (co>0) s.insert(co); }
        col_div += (double)s.size() / N;
    }

    // 颜色奇偶分布
    int color_even[5] = {0}, color_odd[5] = {0};
    for (int r = 0; r < N; ++r)
        for (int c = 0; c < N; ++c) {
            int val = b.at(r,c).value;
            if (val <= 0) continue;
            int ci = val - 1;
            if ((r+c)%2 == 0) color_even[ci]++; else color_odd[ci]++;
        }

    int idx = 0;
    // f[0-5]: 颜色分布
    for (int i = 0; i <= 5; ++i) bf[idx++] = total>0 ? (double)color_cnt[i]/total : 0;
    // f[6-7]: 万能块/炸弹比例
    bf[idx++] = total>0 ? (double)wc_cnt/total : 0;
    bf[idx++] = total>0 ? (double)bomb_cnt/total : 0;
    // f[8-11]: 邻接对
    bf[idx++] = (double)adj_same / std::max(1, 2*N*(N-1));
    bf[idx++] = (double)adj_compat / std::max(1, 2*N*(N-1));
    bf[idx++] = total>0 ? (double)adj_same/total : 0;
    bf[idx++] = total>0 ? (double)adj_compat/total : 0;
    // f[12-15]: 分量统计
    bf[idx++] = total>0 ? (double)max_comp/total : 0;
    bf[idx++] = total>0 ? avg_comp/total : 0;
    bf[idx++] = total>0 ? std::sqrt(std::max(0.0,var_comp))/total : 0;
    bf[idx++] = total>0 ? (double)comp_sizes.size()/total : 0;
    // f[16-20]: 分量分布
    int nc = std::max(1, (int)comp_sizes.size());
    for (int i = 0; i < 5; ++i) bf[idx++] = (double)comp_bin[i]/nc;
    // f[21]: 熵
    bf[idx++] = ent;
    // f[22-26]: 各颜色归一化计数
    for (int i = 1; i <= 5; ++i) bf[idx++] = total>0 ? (double)color_cnt[i]/total : 0;
    // f[27-31]: 各颜色空间分散性(简化)
    for (int ci = 0; ci < 5; ++ci)
        bf[idx++] = color_cnt[ci+1]>1 ? 1.0 : 0.0;
    // f[32-33]: 行列多样性
    bf[idx++] = row_div;
    bf[idx++] = col_div;
    // f[34]: 死锁
    bf[idx++] = b.is_deadlocked() ? 1.0 : 0.0;
    // f[35-37]: 掉落队列
    bf[idx++] = dq_compat;
    bf[idx++] = dq_wc;
    bf[idx++] = dq_bomb;
    // f[38-42]: 棋盘格分析
    bf[idx++] = total>0 ? (double)black/total : 0;
    bf[idx++] = total>0 ? (double)white/total : 0;
    bf[idx++] = total>0 ? (double)(black+white)/total : 0;
    double bp = black+white;
    bf[idx++] = bp>0 ? (double)black/bp : 0.5;
    // f[42]: 有序度
    double pos_ent = 0;
    for (int ci = 0; ci < 5; ++ci)
        if (color_cnt[ci+1] > 0)
            pos_ent += (double)color_cnt[ci+1]/total;
    bf[idx++] = pos_ent;
    // f[43-47]: 二部图
    for (int ci = 0; ci < 5; ++ci) {
        int tci = color_even[ci] + color_odd[ci];
        bf[idx++] = tci>0 ? (double)color_even[ci]/tci : 0.5;
    }
    // f[48-49]: level和size
    bf[idx++] = (double)b.level/5.0;
    bf[idx++] = (double)N/12.0;
    // f[50-51]
    int row_dom = 0, col_dom = 0;
    for (int r = 0; r < N; ++r) {
        int rc[6]={};
        for (int c = 0; c < N; ++c) rc[std::abs(b.at(r,c).value)]++;
        int dom = 0;
        for (int i=1;i<=5;++i) if(rc[i]>rc[dom]) dom=i;
        row_dom += rc[dom];
    }
    for (int c = 0; c < N; ++c) {
        int cc[6]={};
        for (int r = 0; r < N; ++r) cc[std::abs(b.at(r,c).value)]++;
        int dom = 0;
        for (int i=1;i<=5;++i) if(cc[i]>cc[dom]) dom=i;
        col_dom += cc[dom];
    }
    bf[idx++] = total>0 ? (double)row_dom/total/N : 0;
    bf[idx++] = total>0 ? (double)col_dom/total/N : 0;

    return bf;
}

#endif
