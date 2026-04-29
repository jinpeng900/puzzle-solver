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
            for (int i = 0; i < empty; ++i) {
                next_b.at(i,c).value = dq[c][next_b.queue_ptr[c]++];
            }
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
                vis[r][c]=true; q.push_back({r,c});
                while (!q.empty()) { auto [cr,cc]=q.front(); q.pop_front(); sz++;
                    int cco=b.at(cr,cc).color();
                    for (int d=0;d<4;++d) { int nr=cr+DR[d],nc=cc+DC[d]; if (!b.in_bounds(nr,nc)||vis[nr][nc]) continue;
                        int nco=b.at(nr,nc).color();
                        if (!compatible_color(cco,nco)) continue;
                        vis[nr][nc]=true; q.push_back({nr,nc}); }
                }
                mxc=std::max(mxc,sz);
            }
        }
        double ent=0.0; int tc=N*N-wc;
        if (tc>0) for (int i=1;i<=5;++i) if (cnt[i]>0) { double p=(double)cnt[i]/tc; ent-=p*std::log(p); }
        bool is_pure5 = (b.level <= 2);
        bool is_bomb_level = (b.level == 4 || b.level == 5);
        double bb_weight = is_bomb_level ? 22.0 : 15.0;
        return cp*6.0+sp*10.0+wc*35.0+bb*bb_weight+mxc*g_params[b.level].mxc_w-ent*12.0;
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
        // 统计 path 中的黑白格
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

        // ========== 关卡特化参数 ==========
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
                // L2: 从小连通块开始，先消小的保留大的
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

    static int gen_block(std::mt19937& rng, int level) {
        if (level<=2) return (rng()%5)+1;
        if (level==3) return ((rng()%100)<15)?0:(rng()%5)+1;
        if (level==4) { int c=(rng()%5)+1; return ((rng()%100)<10)?-c:c; }
        if ((rng()%100)<15) return 0;
        int base=(rng()%5)+1; return ((rng()%100)<10)?-base:base;
    }

public:
    LocalJudge(int level, int seed, int N) : level_(level), N_(N) {
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

    bool validate(const std::vector<std::pair<int,int>>& path, std::string& reason) const {
        if (path.size() < 2) { reason = "长度<2"; return false; }
        std::vector<std::vector<bool>> used(N_, std::vector<bool>(N_));
        int anchor = 0;
        for (size_t i = 0; i < path.size(); ++i) {
            auto [r, c] = path[i];
            if (r < 0 || r >= N_ || c < 0 || c >= N_) { reason = "坐标越界"; return false; }
            if (used[r][c]) { reason = "重复坐标"; return false; }
            used[r][c] = true;
            int color = board_.at(r, c).color();
            if (color != 0) {
                if (anchor == 0) anchor = color;
                else if (anchor != color) { reason = "颜色不一致"; return false; }
            }
            if (i > 0) {
                auto [pr, pc] = path[i-1];
                if (std::abs(pr - r) + std::abs(pc - c) != 1) { reason = "非四联通"; return false; }
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
// 测试入口
// ============================================================
int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int target_seed = (argc > 1) ? std::atoi(argv[1]) : 114514;
    const char* param_file = (argc > 2) ? argv[2] : nullptr;

    init_default_params();
    if (param_file) load_params(param_file);

    struct LevelConfig {
        int level;
        int seed;
        int N;
        const char* name;
    };

    int target_level = (argc > 3) ? std::atoi(argv[3]) : 0;
    std::vector<LevelConfig> all_levels = {
        {1, target_seed, 10, "Level 1 (10x10, 纯5色)"},
        {2, target_seed, 10, "Level 2 (10x10, 纯5色)"},
        {3, target_seed, 10, "Level 3 (10x10, 万能块15%)"},
        {4, target_seed, 10, "Level 4 (10x10, 炸弹10%)"},
        {5, target_seed, 12, "Level 5 (12x12, 万能块15%+炸弹10%)"},
    };
    std::vector<LevelConfig> levels;
    if (target_level >= 1 && target_level <= 5) {
        levels.push_back(all_levels[target_level - 1]);
    } else {
        levels = all_levels;
    }

    ImprovedSolver solver;
    int total_score = 0;

    for (const auto& lc : levels) {
        LocalJudge judge(lc.level, lc.seed, lc.N);
        std::cout << "\n========================================\n";
        std::cout << "  " << lc.name << "\n";
        std::cout << "========================================\n";
        std::cout << "Initial board:\n";
        judge.print_board(std::cout);

        int prev_score = 0;
        while (!judge.done()) {
            auto path = solver.solve(judge.board());
            if (path.size() < 2) path = {{0,0},{0,1}};

            std::string reason;
            bool was_valid = judge.validate(path, reason);
            int old_step = judge.step();
            bool cont = judge.play(path);
            int gained = judge.score() - prev_score;
            prev_score = judge.score();

            std::cout << "Step " << old_step + 1 << ": len=" << path.size();
            if (!was_valid) std::cout << " [INVALID: " << reason << "]";
            else std::cout << " [+" << gained << "]";
            std::cout << " | path:";
            for (auto [r,c] : path) std::cout << " (" << r << "," << c << ")";
            std::cout << "\n";

            if (!cont) break;
        }

        std::cout << "\nResult: " << judge.score() << " pts in " << judge.step() << " steps\n";
        total_score += judge.score();
    }

    std::cout << "\n========================================\n";
    std::cout << "  TOTAL SCORE: " << total_score << "\n";
    std::cout << "========================================\n";
    return 0;
}
