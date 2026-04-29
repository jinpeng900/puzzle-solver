/**
 * 2.cpp — 改进版求解器
 * 基于 1.cpp，增加：
 *   - 神经网络混合评估器 (nn_solver)
 *   - 扩展自适应反馈（mxc_weight, step3权重）
 *   - 改进DFS邻居排序（1-ply lookahead）
 *   - 改进生存分析（累积分数）
 *   - L3/L5 启用 beam search
 *   - L5 dfs_limit 提升
 */
#if __cplusplus < 201402L
#error "C++14 required"
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
#include <memory>

#include "nn_solver/board_features.h"
#include "nn_solver/nn_model.h"
#include "nn_solver/nn_evaluator.h"

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
        for (auto& p : path) in_path[p.first][p.second] = true;
        std::vector<std::vector<bool>> to_remove = in_path;
        if (level >= 4)
            for (auto& p : path) {
                int r = p.first, c = p.second;
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
    for (auto& p : path) in_path[p.first][p.second]=true;
    std::vector<std::vector<bool>> exploded(board.N,std::vector<bool>(board.N));
    for (auto& p : path) {
        int r=p.first, c=p.second;
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
// 全局 NN 评估器
// ============================================================
static NNEvaluator<Board> g_nn_eval;

// ============================================================
// 求解器：DFS + 多步前瞻 + NN评估 + 自适应学习
// ============================================================
class ImprovedSolver {
    static constexpr int MAX_TIME_MS = 10000;
    mutable std::chrono::steady_clock::time_point _t0;
    mutable bool _tinit = false;
    void tstart() const { if(!_tinit){_t0=std::chrono::steady_clock::now();_tinit=true;} }
    long long telapsed() const { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_t0).count(); }
    mutable int step_count_ = 0;
    bool tfast()  const { return telapsed() > MAX_TIME_MS * 0.90; }
    bool tdead()  const { return telapsed() > MAX_TIME_MS * 0.97; }

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

    // === 自适应参数追踪（扩展版）===
    mutable double last_pred_dq_ = 0;
    mutable double last_pred_bh_ = 0;
    mutable int    last_pred_nxt_ = 0;
    mutable int    last_pred_now_ = 0;
    mutable int    last_pred_level_ = -1;
    mutable bool   pending_feedback_ = false;

    // 新增：追踪 mxc 和 step3 预测值用于反馈
    mutable double last_pred_mxc_contrib_ = 0;  // mxc * mxc_weight
    mutable double last_step3_score_ = 0;       // 第三步的总分

    void apply_feedback(const Board& actual_board) {
        if (!pending_feedback_) return;

        double actual_dq = drop_quality(actual_board);
        double actual_bh = board_heuristic(actual_board);
        int actual_nxt = exact_best_one_step_score(actual_board, std::min(20000, cache_limit()));

        // dq 权重调整
        double dq_delta = actual_dq - last_pred_dq_;
        double predicted = last_pred_nxt_ + last_pred_dq_ * dq_weight_ + last_pred_bh_ * bh_weight_;
        double actual_val = actual_nxt + actual_dq * dq_weight_ + actual_bh * bh_weight_;
        double norm = std::max(1.0, (std::abs(predicted) + std::abs(actual_val)) / 2.0);
        double rel_err = (actual_val - predicted) / norm;

        dq_weight_ += 0.002 * rel_err * dq_delta;
        dq_weight_ = std::max(0.1, std::min(2.5, dq_weight_));

        double bh_delta = actual_bh - last_pred_bh_;
        bh_weight_ += 0.0005 * rel_err * bh_delta;
        bh_weight_ = std::max(-0.05, std::min(0.15, bh_weight_));

        // ★新增：mxc_weight 自适应
        if (last_pred_level_ >= 1 && last_pred_level_ <= 5) {
            // 用实际 nxt 值反馈调整 mxc_weight
            // 如果实际 nxt 显著低于预测，说明高估了盘面 → 降低 mxc_weight
            if (actual_nxt < last_pred_nxt_ * 0.5) {
                mxc_weight_ -= 0.3;
                mxc_weight_ = std::max(8.0, mxc_weight_);
            } else if (actual_nxt > last_pred_nxt_ * 1.8) {
                mxc_weight_ += 0.2;
                mxc_weight_ = std::min(35.0, mxc_weight_);
            }
        }

        // ★新增：step3 权重自适应
        // 如果第三步分数很低，降低 step3 权重
        if (last_step3_score_ < 0 && actual_nxt > 0) {
            step3_dq_weight_ -= 0.01;
            step3_dq_weight_ = std::max(0.2, step3_dq_weight_);
            step3_bh_weight_ -= 0.002;
            step3_bh_weight_ = std::max(0.0, step3_bh_weight_);
        }

        // surv_mul 调整（保持原有逻辑，增强幅度）
        int actual_surv = survival_steps(actual_board, std::min(4, 6));
        if (actual_surv <= 2) {
            surv_mul_ += 0.05 * (3.0 - actual_surv); // 增强调整幅度
            surv_mul_ = std::max(0.1, std::min(3.0, surv_mul_));
        } else if (actual_surv >= 5) {
            surv_mul_ -= 0.02;
            surv_mul_ = std::max(0.1, std::min(3.0, surv_mul_));
        }

        pending_feedback_ = false;
    }

    void store_prediction(const Board& next_b, int now, int nxt, int level, double step3_val = 0) {
        last_pred_dq_ = drop_quality(next_b);
        last_pred_bh_ = board_heuristic(next_b);
        last_pred_nxt_ = nxt;
        last_pred_now_ = now;
        last_pred_level_ = level;
        last_step3_score_ = step3_val;
        pending_feedback_ = true;
    }

public:
    void give_feedback(const Board& actual_board) {
        if (!pending_feedback_) return;
        apply_feedback(actual_board);
    }

private:
    // === 盘面启发式评估（混合 NN + 传统启发式）===
    static double board_heuristic(const Board& b) {
        if (b.is_deadlocked()) return -1e5;
        int N=b.N, cp=0, sp=0, wc=0, bb=0, cnt[6]={}, mxc=0;
        std::vector<std::vector<bool>> vis(N,std::vector<bool>(N));
        for (int r=0;r<N;++r) for (int c=0;c<N;++c) {
            int co=b.at(r,c).color();
            if (b.at(r,c).is_wildcard()) { wc++; }
            if (b.at(r,c).is_bomb()) bb++;
            cnt[std::min(co,5)]++;
            if (c+1<N) { int c2=b.at(r,c+1).color(); if (compatible_color(co,c2)){cp++;if(co==c2&&co!=0)sp++;} }
            if (r+1<N) { int c2=b.at(r+1,c).color(); if (compatible_color(co,c2)){cp++;if(co==c2&&co!=0)sp++;} }
            if (!vis[r][c]) {
                int sz=0; std::deque<std::pair<int,int>> q;
                vis[r][c]=true; q.push_back(std::make_pair(r,c));
                while (!q.empty()) { auto p=q.front(); q.pop_front(); sz++;
                    int cr=p.first, cc=p.second;
                    int cco=b.at(cr,cc).color();
                    for (int d=0;d<4;++d) { int nr=cr+DR[d],nc=cc+DC[d]; if (!b.in_bounds(nr,nc)||vis[nr][nc]) continue;
                        int nco=b.at(nr,nc).color();
                        if (!compatible_color(cco,nco)) continue;
                        vis[nr][nc]=true; q.push_back(std::make_pair(nr,nc)); }
                }
                mxc=std::max(mxc,sz);
            }
        }
        double ent=0.0; int tc=N*N-wc;
        if (tc>0) for (int i=1;i<=5;++i) if (cnt[i]>0) { double p=(double)cnt[i]/tc; ent-=p*std::log(p); }
        bool is_pure5 = (b.level <= 2);
        bool is_bomb_level = (b.level == 4 || b.level == 5);
        double bb_weight = is_bomb_level ? 22.0 : 15.0;
        double h_score = cp*6.0+sp*10.0+wc*35.0+bb*bb_weight+mxc*(is_pure5?25.0:15.0)-ent*12.0;

        // ★NN 混合评估
        if (g_nn_eval.is_enabled()) {
            double nn = g_nn_eval.pure_nn(b);
            return g_nn_eval.nn_weight() * nn + (1.0 - g_nn_eval.nn_weight()) * h_score;
        }
        return h_score;
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
        for (auto& p : path) blocked[p.first][p.second]=true;
        int tr=path.back().first, tc=path.back().second;
        std::deque<std::pair<int,int>> q;
        std::vector<std::vector<bool>> seen(b.N, std::vector<bool>(b.N, false));
        for (int d=0; d<4; ++d) {
            int nr=tr+DR[d], nc=tc+DC[d];
            if (!b.in_bounds(nr,nc)||blocked[nr][nc]) continue;
            if (!compatible_color(target,b.at(nr,nc).color())) continue;
            seen[nr][nc]=true; q.push_back(std::make_pair(nr,nc));
        }
        int reachable=0;
        int black=0, white=0;
        for (auto& p : path) {
            if ((p.first+p.second)%2==0) black++; else white++;
        }
        while (!q.empty()) {
            auto p=q.front(); q.pop_front(); ++reachable;
            int r=p.first, c=p.second;
            if ((r+c)%2==0) black++; else white++;
            for (int d=0; d<4; ++d) {
                int nr=r+DR[d], nc=c+DC[d];
                if (!b.in_bounds(nr,nc)||blocked[nr][nc]||seen[nr][nc]) continue;
                if (!compatible_color(target,b.at(nr,nc).color())) continue;
                seen[nr][nc]=true; q.push_back(std::make_pair(nr,nc));
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
        std::function<void(std::uint64_t,int,int,int)> dfs=[&](std::uint64_t mask,int r,int c,int target) {
            if (++nodes>max_nodes) return;
            int co=b.at(r,c).color(); int fixed=target; if (fixed==0&&co!=0) fixed=co;
            if ((int)path.size()>=2) {
                int sc=path_score(b,path); if (sc>best) best=sc;
                int ub=upper_bound_reachable(b,path,fixed);
                if (path_score(ub)<=best) return;
            }
            std::pair<int,int> nb[4]; int nb_cnt=0;
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (!b.in_bounds(nr,nc)||(mask&bit(nr,nc,b.N))) continue;
                if (!compatible_color(fixed,b.at(nr,nc).color())) continue;
                nb[nb_cnt++]=std::make_pair(nr,nc); }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N))) continue;
                        if(compatible_color(fixed,b.at(ppr,ppc).color())) pot++;
                    }
                    if (b.level<=2) {
                        int same=0;
                        for(int dd=0;dd<4;++dd){
                            int ppr=pr+DR[dd], ppc=pc+DC[dd];
                            if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N))) continue;
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
            for (int i=0;i<nb_cnt;++i) { int nr=nb[i].first, nc=nb[i].second; path.push_back(nb[i]); dfs(mask|bit(nr,nc,b.N),nr,nc,fixed); path.pop_back(); if (nodes>max_nodes) return; }
        };
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            path.clear(); path.push_back(std::make_pair(r,c)); dfs(bit(r,c,b.N),r,c,0); if (nodes>max_nodes) break;
        }
        return best;
    }

    static std::pair<int,std::vector<std::pair<int,int>>>
    exact_best_with_path(const Board& b, int max_nodes) {
        int best_score=0, nodes=0;
        std::vector<std::pair<int,int>> best_path, path;
        std::function<void(std::uint64_t,int,int,int)> dfs=[&](std::uint64_t mask,int r,int c,int target) {
            if (++nodes>max_nodes) return;
            int co=b.at(r,c).color(); int fixed=target; if (fixed==0&&co!=0) fixed=co;
            if ((int)path.size()>=2) {
                int sc=path_score(b,path);
                if (sc>best_score||(sc==best_score&&path<best_path)) { best_score=sc; best_path=path; }
                int ub=upper_bound_reachable(b,path,fixed);
                if (path_score(ub)<=best_score) return;
            }
            std::pair<int,int> nb[4]; int nb_cnt=0;
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (!b.in_bounds(nr,nc)||(mask&bit(nr,nc,b.N))) continue;
                if (!compatible_color(fixed,b.at(nr,nc).color())) continue;
                nb[nb_cnt++]=std::make_pair(nr,nc); }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N))) continue;
                        if(compatible_color(fixed,b.at(ppr,ppc).color())) pot++;
                    }
                    if (b.level<=2) {
                        int same=0;
                        for(int dd=0;dd<4;++dd){
                            int ppr=pr+DR[dd], ppc=pc+DC[dd];
                            if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N))) continue;
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
            for (int i=0;i<nb_cnt;++i) { path.push_back(nb[i]); dfs(mask|bit(nb[i].first,nb[i].second,b.N),nb[i].first,nb[i].second,fixed); path.pop_back(); if (nodes>max_nodes) return; }
        };
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            path.clear(); path.push_back(std::make_pair(r,c)); dfs(bit(r,c,b.N),r,c,0); if (nodes>max_nodes) break;
        }
        return {best_score,best_path};
    }

    double simulate_drops_and_score(const Board& b, int sim_steps) const {
        Board sim=b;
        double total=0;
        for (int st=0;st<sim_steps;++st) {
            if (sim.is_deadlocked()) break;
            auto scbp = exact_best_with_path(sim,std::max(4000,cache_limit()/sim_steps));
            int sc = scbp.first; auto& bp = scbp.second;
            if (bp.size()<2) break;
            total+=(double)sc*std::pow(0.7,(double)st);
            sim=sim.preview(bp);
        }
        return total;
    }

    // ★改进：生存分析现在累积实际分数
    int survival_steps(const Board& b, int max_check) const {
        Board sim=b; int i;
        int acc_score = 0;
        for (i=0;i<max_check;++i) {
            if (sim.is_deadlocked()) break;
            auto scbp = exact_best_with_path(sim,tdead()?5000:20000);
            int sc = scbp.first; auto& bp = scbp.second;
            if (bp.size()<2) { i++; break; }
            acc_score += sc;
            sim=sim.preview(bp);
        }
        return acc_score; // ★返回累积分数而非步数
    }

    double beam_evaluate(const Board& start_b, int beam_w, int beam_d) const {
        struct BNode { Board bd; double acc; };
        std::vector<BNode> beams; beams.push_back({start_b,0.0});
        for (int d=0;d<beam_d;++d) {
            struct Cand { Board bd; double acc; double val; };
            std::vector<Cand> cands;
            for (auto& bm:beams) {
                auto scbp = exact_best_with_path(bm.bd,d==beam_d-1?10000:5000);
                int sc = scbp.first; auto& bp = scbp.second;
                if (bp.size()<2) continue;
                Board nb=bm.bd.preview(bp);
                double hv=board_heuristic(nb);
                double total=bm.acc+(double)sc+hv*(d==beam_d-1?0.5:0.15);
                cands.push_back({std::move(nb),bm.acc+(double)sc,total});
                for (int tries=0;tries<4&&(int)cands.size()<beam_w*4;++tries) {
                    auto scbp2 = exact_best_with_path(bm.bd,2000+tries*1000);
                    int sc2 = scbp2.first; auto& bp2 = scbp2.second;
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
            if (b.at(r,c).is_wildcard()) for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (b.in_bounds(nr,nc)) return {std::make_pair(r,c),std::make_pair(nr,nc)}; }
            int a=b.at(r,c).color();
            if (c+1<b.N&&compatible_color(a,b.at(r,c+1).color())) return {std::make_pair(r,c),std::make_pair(r,c+1)};
            if (r+1<b.N&&compatible_color(a,b.at(r+1,c).color())) return {std::make_pair(r,c),std::make_pair(r+1,c)};
        }
        return {std::make_pair(0,0),std::make_pair(0,1)};
    }

public:
    std::vector<std::pair<int,int>> solve(const Board& board) {
        _tinit = false;
        tstart();
        ++step_count_;

        // 应用上一步反馈
        if (pending_feedback_) {
            if (board.level == last_pred_level_) apply_feedback(board);
            else pending_feedback_ = false;
        }

        board_N_ = board.N;
        bool pure5 = (board.level <= 2);
        // ★改进：L5 dfs_limit 从 800K → 1200K
        int lmt = (board_N_<=5?800000:(pure5?1200000:(board.level==5?1200000:(board.level==3?700000:600000))));
        int cmax = cache_limit();

        // === 参数初始化 ===
        switch(board.level) {
            case 1: cur_cache_limit_ = 80000; cur_keep1_ = 200; cur_keep2_ = 100; cur_keep3_ = 20; cur_beam_w_ = 8; cur_beam_d_ = 4; break;
            case 2: cur_cache_limit_ = 80000; cur_keep1_ = 200; cur_keep2_ = 100; cur_keep3_ = 20; cur_beam_w_ = 8; cur_beam_d_ = 4; break;
            case 3: cur_cache_limit_ = 60000; cur_keep1_ = 120; cur_keep2_ = 60; cur_keep3_ = 12; cur_beam_w_ = 2; cur_beam_d_ = 2; break; // ★L3 启用 beam
            case 4: cur_cache_limit_ = 80000; cur_keep1_ = 120; cur_keep2_ = 60; cur_keep3_ = 12; cur_beam_w_ = 5; cur_beam_d_ = 3; break;
            case 5: cur_cache_limit_ = 50000; cur_keep1_ = 60; cur_keep2_ = 30; cur_keep3_ = 6; cur_beam_w_ = 2; cur_beam_d_ = 2; break; // ★L5 启用 beam
            default: cur_cache_limit_ = 60000; cur_keep1_ = 200; cur_keep2_ = 100; cur_keep3_ = 20; cur_beam_w_ = 8; cur_beam_d_ = 4; break;
        }

        static int last_weight_level = -1;
        if (board.level != last_weight_level) {
            switch(board.level) {
                case 1: mxc_weight_ = 25.0; dq_weight_ = 0.8; bh_weight_ = 0.02; step3_dq_weight_ = 0.55; step3_bh_weight_ = 0.025; surv_mul_ = 0.5; break;
                case 2: mxc_weight_ = 25.0; dq_weight_ = 0.8; bh_weight_ = 0.02; step3_dq_weight_ = 0.55; step3_bh_weight_ = 0.025; surv_mul_ = 1.0; break;
                case 3: mxc_weight_ = 18.0; dq_weight_ = 0.9; bh_weight_ = 0.02; step3_dq_weight_ = 0.6; step3_bh_weight_ = 0.025; surv_mul_ = 1.0; break;
                case 4: mxc_weight_ = 15.0; dq_weight_ = 0.8; bh_weight_ = 0.02; step3_dq_weight_ = 0.55; step3_bh_weight_ = 0.025; surv_mul_ = 1.0; break;
                case 5: mxc_weight_ = 15.0; dq_weight_ = 0.9; bh_weight_ = 0.02; step3_dq_weight_ = 0.6; step3_bh_weight_ = 0.025; surv_mul_ = 1.0; break;
                default: mxc_weight_ = 15.0; dq_weight_ = 0.8; bh_weight_ = 0.02; step3_dq_weight_ = 0.55; step3_bh_weight_ = 0.025; surv_mul_ = 1.0; break;
            }
            last_weight_level = board.level;
        }

        struct Raw { std::vector<std::pair<int,int>> path; int now; int nxt; };
        std::vector<Raw> raws; raws.reserve(8192);

        static std::unordered_map<std::uint64_t,int> cache2;
        int nodes=0;
        std::vector<std::pair<int,int>> path;

        std::function<void(std::uint64_t,int,int,int)> dfs=[&](std::uint64_t mask,int r,int c,int target) {
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
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (!board.in_bounds(nr,nc)||(mask&bit(nr,nc,board.N))) continue;
                if (!compatible_color(fixed,board.at(nr,nc).color())) continue;
                nb_arr[nb_cnt++]=std::make_pair(nr,nc); }
            // ★改进：邻居排序使用 1-ply lookahead
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval_1ply=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!board.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,board.N))) continue;
                        if(compatible_color(fixed,board.at(ppr,ppc).color())) pot++;
                    }
                    if (pure5) {
                        // ★1-ply: 再数下一层的邻居
                        int pot2=0;
                        for(int dd=0;dd<4;++dd){
                            int ppr=pr+DR[dd], ppc=pc+DC[dd];
                            if(!board.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,board.N))) continue;
                            if(!compatible_color(fixed,board.at(ppr,ppc).color())) continue;
                            for(int ee=0;ee<4;++ee){
                                int ppr2=ppr+DR[ee], ppc2=ppc+DC[ee];
                                if(!board.in_bounds(ppr2,ppc2)) continue;
                                if(ppr2==pr&&ppc2==pc) continue;
                                if((mask&bit(ppr2,ppc2,board.N))) continue;
                                if(compatible_color(fixed,board.at(ppr2,ppc2).color())) pot2++;
                            }
                        }
                        return pot*10+pot2*2;
                    }
                    int base=(board.at(pr,pc).is_wildcard()?40:(board.at(pr,pc).is_bomb()?35:(board.at(pr,pc).color()==fixed?20:0)));
                    return base+pot;
                };
                if(eval_1ply(nb_arr[j])>eval_1ply(nb_arr[i])) std::swap(nb_arr[i],nb_arr[j]);
            }
            for (int i=0;i<nb_cnt;++i) { path.push_back(nb_arr[i]); dfs(mask|bit(nb_arr[i].first,nb_arr[i].second,board.N),nb_arr[i].first,nb_arr[i].second,fixed); path.pop_back(); if (nodes>lmt) return; }
        };

        // 起点排序（与 1.cpp 相同）
        std::vector<std::pair<int,int>> starts;
        if (pure5) {
            std::vector<std::vector<int>> cc_size(board.N, std::vector<int>(board.N, 0));
            std::vector<std::vector<bool>> vis(board.N, std::vector<bool>(board.N, false));
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) {
                if (vis[r][c]) continue;
                int co = board.at(r,c).color();
                std::vector<std::pair<int,int>> cells;
                std::deque<std::pair<int,int>> q;
                q.push_back(std::make_pair(r,c)); vis[r][c]=true;
                while (!q.empty()) { auto p=q.front(); q.pop_front(); cells.push_back(p);
                    int cr=p.first, cc=p.second;
                    for (int d=0;d<4;++d) { int nr=cr+DR[d],nc=cc+DC[d];
                        if (!board.in_bounds(nr,nc)||vis[nr][nc]) continue;
                        if (board.at(nr,nc).color()!=co) continue;
                        vis[nr][nc]=true; q.push_back(std::make_pair(nr,nc)); }
                }
                for (auto& p:cells) cc_size[p.first][p.second]=(int)cells.size();
            }
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c)
                starts.push_back(std::make_pair(r,c));
            if (board.level==1) {
                std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){ return cc_size[a.first][a.second]>cc_size[b.first][b.second]; });
            } else {
                std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){ return cc_size[a.first][a.second]<cc_size[b.first][b.second]; });
            }
        } else {
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) starts.push_back(std::make_pair(r,c));
        }

        for (auto& p : starts) {
            path.clear(); path.push_back(p); dfs(bit(p.first,p.second,board.N),p.first,p.second,0); if (nodes>lmt) break;
        }
        if (raws.empty()) {
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) {
                path.clear(); path.push_back(std::make_pair(r,c)); dfs(bit(r,c,board.N),r,c,0); if (nodes>lmt) break;
            }
        }
        if (raws.empty()) return fallback(board);

        std::vector<int> order(raws.size()); for (int i=0;i<(int)raws.size();++i) order[i]=i;
        std::sort(order.begin(),order.end(),[&](int a,int b) {
            int va=raws[a].now+raws[a].nxt, vb=raws[b].now+raws[b].nxt;
            if (pure5) {
                int pa=(int)raws[a].path.size(), pb=(int)raws[b].path.size();
                if (board.level==1) {
                    if (pa<6) va-=55;
                    if (pb<6) vb-=55;
                } else {
                    if (pa<6) va-=25;
                    if (pb<6) vb-=25;
                }
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
                auto scbp = exact_best_with_path(cands[i].next_b,cmax);
                int nsc = scbp.first; auto& np = scbp.second;
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

        // ★改进：生存分析用累积分数
        int k3=std::min(keep3(),(int)scored.size());
        for (int j=0;j<k3;++j) {
            int surv_score = survival_steps(cands[scored[j].idx].next_b,6);
            // 生存分数高 → 奖励
            scored[j].val += (double)surv_score * surv_mul_ * 0.15;
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});

        if (!tfast() && cur_beam_w_ > 0) {
            int bs_cnt=std::min(3,(int)scored.size());
            for (int j=0;j<bs_cnt;++j) {
                double beam_val=beam_evaluate(cands[scored[j].idx].next_b,cur_beam_w_,cur_beam_d_);
                scored[j].val+=beam_val*0.3;
            }
            std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        }

        int sel = 0;
        int chosen_idx = scored[sel].idx;

        // 计算 step3_val 用于反馈
        double step3_val = 0;
        {
            auto scbp = exact_best_with_path(cands[chosen_idx].next_b, cmax/2);
            int nsc = scbp.first; auto& np = scbp.second;
            if (np.size() >= 2) {
                Board tb = cands[chosen_idx].next_b.preview(np);
                int tsc = exact_best_one_step_score(tb, cmax/3);
                step3_val = (double)nsc + (double)tsc;
            }
        }
        store_prediction(cands[chosen_idx].next_b, cands[chosen_idx].now, cands[chosen_idx].nxt, board.level, step3_val);

        if (cache2.size()>400000) cache2.clear();
        if (cache3.size()>300000) cache3.clear();
        return cands[chosen_idx].path;
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
// 协议交互层：GameController（与 1.cpp 相同，适配 shared_ptr Board）
// ============================================================
class GameController {
public:
    struct DropObservation { int col=0; int value=0; };
private:
    Board _board; int _level=0, _step=0, _score=0; bool _done=false, _has_feedback=false, _last_valid=true;
    int _last_reward=0; std::string _pending_line;
    std::vector<std::pair<int,int>> _last_path; std::vector<DropObservation> _drop_obs;

    static int try_parse_level(const std::string& line, int& level, int& seed) {
        int lv,sd,N,steps;
        if (std::sscanf(line.c_str(),"LEVEL %d SEED %d SIZE %d STEPS %d",&lv,&sd,&N,&steps)==4) { level=lv; seed=sd; return N; }
        return 0;
    }
    static bool try_parse_step(const std::string& line, int& step, int& score, bool& valid) {
        char buf[16]={};
        if (std::sscanf(line.c_str(),"STEP %d SCORE %d %15s",&step,&score,buf)>=3) { valid=(std::string(buf)=="VALID"); return true; }
        return false;
    }
    static int gen_block(std::mt19937& rng, int level) {
        if (level<=2) return (rng()%5)+1;
        if (level==3) return ((rng()%100)<15)?0:(rng()%5)+1;
        if (level==4) { int c=(rng()%5)+1; return ((rng()%100)<10)?-c:c; }
        if ((rng()%100)<15) return 0;
        int base=(rng()%5)+1; return ((rng()%100)<10)?-base:base;
    }
    static void init_queues(Board& b, int seed, int N, int level) {
        b.level=level; std::mt19937 rng(seed);
        b.drop_queue->clear(); b.drop_queue->assign(N, std::vector<int>(1000));
        b.queue_ptr.assign(N,0);
        for (int c=0;c<N;++c) for (int i=0;i<1000;++i) (*b.drop_queue)[c][i]=gen_block(rng,level);
    }
    static std::vector<int> removed_per_column(const Board& b, const std::vector<std::pair<int,int>>& path) {
        std::vector<int> removed(b.N,0);
        if (path.size()<2) return removed;
        std::vector<std::vector<bool>> in_path(b.N,std::vector<bool>(b.N)), to_remove(b.N,std::vector<bool>(b.N));
        for (auto& p : path) in_path[p.first][p.second]=to_remove[p.first][p.second]=true;
        if (b.level>=4) for (auto& p : path) {
            int r=p.first, c=p.second;
            if (!b.at(r,c).is_bomb()) continue;
            for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
                int nr=r+dr,nc=c+dc;
                if (b.in_bounds(nr,nc)&&!in_path[nr][nc]) to_remove[nr][nc]=true;
            }
        }
        for (int c=0;c<b.N;++c) { int cnt=0; for (int r=0;r<b.N;++r) if (to_remove[r][c]) ++cnt; removed[c]=cnt; }
        return removed;
    }
    bool read_line(std::string& line) {
        if (!_pending_line.empty()) { line=std::move(_pending_line); _pending_line.clear(); return true; }
        return (bool)std::getline(std::cin,line);
    }
    Board read_board(int N) {
        Board board(N);
        for (int row=0;row<N;++row) { std::string line; read_line(line); std::istringstream ls(line); for (int c=0;c<N;++c) ls>>board.at(row,c).value; }
        return board;
    }
    void drain_trailing() {
        std::string line;
        while (std::cin.rdbuf()->in_avail()>0) {
            if (!read_line(line)) break;
            if (line.empty()||line.find("LEVEL_END")!=std::string::npos) continue;
            if (line.find("FINAL_SCORE")!=std::string::npos) { _done=true; continue; }
            _pending_line=std::move(line); break;
        }
    }
public:
    const Board& board() const { return _board; }
    int level() const { return _level; }
    int step()  const { return _step; }
    int score() const { return _score; }
    bool done() const { return _done; }

    bool update() {
        std::string first_line;
        while (true) { if (!read_line(first_line)) { _done=true; return false; } if (!first_line.empty()) break; }
        if (first_line.find("LEVEL_END")!=std::string::npos||first_line.find("FINAL_SCORE")!=std::string::npos) { _done=true; return false; }
        int seed, new_N=try_parse_level(first_line,_level,seed);
        if (new_N>0) { Board new_board=read_board(new_N); init_queues(new_board,seed,new_N,_level); _board=std::move(new_board); _step=0; _score=0; _has_feedback=false; _drop_obs.clear(); drain_trailing(); return true; }
        int step,score; bool valid;
        if (try_parse_step(first_line,step,score,valid)) {
            int prev_score=_score; _step=step; _score=score; _last_valid=valid; _last_reward=valid?(_score-prev_score):-30; _has_feedback=true; _drop_obs.clear();
            std::vector<int> removed_cols;
            if (valid&&!_last_path.empty()) removed_cols=removed_per_column(_board,_last_path);
            Board predicted=(valid&&!_last_path.empty())?_board.preview(_last_path):_board;
            Board new_board=read_board(_board.N);
            if (!removed_cols.empty()) for (int c=0;c<_board.N;++c) { int drops=removed_cols[c]; for (int r=0;r<drops&&r<_board.N;++r) _drop_obs.push_back({c,new_board.at(r,c).value}); }
            new_board.level=_level; new_board.drop_queue=std::move(predicted.drop_queue); new_board.queue_ptr=std::move(predicted.queue_ptr);
            _board=std::move(new_board); _last_path.clear(); drain_trailing();
            if (!_pending_line.empty()) { int nl,ns; int nN=try_parse_level(_pending_line,nl,ns); if (nN>0) { _level=nl; _pending_line.clear(); Board nb=read_board(nN); init_queues(nb,ns,nN,nl); _board=std::move(nb); _step=0; _score=0; _has_feedback=false; _drop_obs.clear(); drain_trailing(); } }
            return true;
        }
        _done=true; return false;
    }
    void respond(const std::vector<std::pair<int,int>>& path) { _last_path=path; std::cout<<path.size(); for (auto& p : path) std::cout<<' '<<p.first<<' '<<p.second; std::cout<<'\n'; std::cout.flush(); }
};

// ============================================================
// 主入口（在线协议）
// ============================================================
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    // 尝试加载预训练权重
    g_nn_eval.enable(false);
    if (g_nn_eval.load_weights("nn_solver/weights/nn_weights.bin")) {
        g_nn_eval.enable(true);
        g_nn_eval.set_nn_weight(0.3); // NN 权重 30%，启发式 70%
    }

    GameController ctl;
    ImprovedSolver solver;
    while (ctl.update()) { auto path=solver.solve(ctl.board()); ctl.respond(path); }
    return 0;
}
