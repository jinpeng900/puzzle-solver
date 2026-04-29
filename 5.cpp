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
    std::vector<std::vector<int>> drop_queue;
    std::vector<int> queue_ptr;

    explicit Board(int n = 0) : N(n), grid(n, std::vector<Cell>(n)) {}
    Cell&       at(int r, int c)       { return grid[r][c]; }
    const Cell& at(int r, int c) const { return grid[r][c]; }
    bool in_bounds(int r, int c) const { return r >= 0 && r < N && c >= 0 && c < N; }

    Board preview(const std::vector<std::pair<int,int>>& path) const {
        Board next_b = *this;
        if (path.size() < 2) return next_b;
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
                next_b.at(i,c).value = next_b.drop_queue[c][next_b.queue_ptr[c]++];
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
    static constexpr int MAX_TIME_MS = 10000;
    mutable std::chrono::steady_clock::time_point _t0;
    mutable bool _tinit = false;
    void tstart() const { if(!_tinit){_t0=std::chrono::steady_clock::now();_tinit=true;} }
    long long telapsed() const { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_t0).count(); }
    bool tfast()  const { return telapsed()>(long long)(MAX_TIME_MS*0.80); }
    bool tdead()  const { return telapsed()>(long long)(MAX_TIME_MS*0.95); }

    int cache_limit()const { return tfast()?20000:60000; }
    int keep1()    const { return tfast()?80:200; }
    int keep2()    const { return tfast()?35:100; }
    int keep3()    const { return tfast()?5:20; }

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
        return cp*6.0+sp*10.0+wc*35.0+bb*15.0+mxc*(is_pure5?25.0:15.0)-ent*12.0;
    }

    static double drop_quality(const Board& b) {
        double q=0;
        for (int c=0;c<b.N;++c) {
            int qp=b.queue_ptr[c];
            for (int i=0;i<7;++i) {
                if (qp+i>=1000) break;
                int v=b.drop_queue[c][qp+i];
                if (v==0) { q+=2.8; continue; }
                if (v<0) { q+=0.9; continue; }
                int col=std::abs(v);
                double adj=0;
                if (b.in_bounds(0,c-1)&&compatible_color(col,b.at(0,c-1).color())) adj+=0.5;
                if (b.in_bounds(0,c+1)&&compatible_color(col,b.at(0,c+1).color())) adj+=0.5;
                if (b.in_bounds(1,c)  &&compatible_color(col,b.at(1,c).color()))   adj+=0.3;
                q+=adj;
                if (i<3) for (int j=i+1;j<3&&j<7;++j) {
                    int v2=b.drop_queue[c][qp+j]; if (v2==0||v==v2) q+=0.2;
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
        while (!q.empty()) {
            auto [r,c]=q.front(); q.pop_front(); ++reachable;
            for (int d=0; d<4; ++d) {
                int nr=r+DR[d], nc=c+DC[d];
                if (!b.in_bounds(nr,nc)||blocked[nr][nc]||seen[nr][nc]) continue;
                if (!compatible_color(target,b.at(nr,nc).color())) continue;
                seen[nr][nc]=true; q.push_back({nr,nc});
            }
        }
        return (int)path.size()+reachable;
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
                nb[nb_cnt++]={nr,nc}; }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N))) continue;
                        if(compatible_color(fixed,b.at(ppr,ppc).color())) pot++;
                    }
                    if (b.level<=2) return pot;
                    int base=(b.at(pr,pc).is_wildcard()?40:(b.at(pr,pc).color()==fixed?20:0));
                    return base+pot;
                };
                if(b.level<=2) { if(eval(nb[j])<eval(nb[i])) std::swap(nb[i],nb[j]); }
                else { if(eval(nb[j])>eval(nb[i])) std::swap(nb[i],nb[j]); }
            }
            for (int i=0;i<nb_cnt;++i) { auto [nr,nc]=nb[i]; path.push_back({nr,nc}); dfs(mask|bit(nr,nc,b.N),nr,nc,fixed); path.pop_back(); if (nodes>max_nodes) return; }
        };
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            path.clear(); path.push_back({r,c}); dfs(bit(r,c,b.N),r,c,0); if (nodes>max_nodes) break;
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
                nb[nb_cnt++]={nr,nc}; }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N))) continue;
                        if(compatible_color(fixed,b.at(ppr,ppc).color())) pot++;
                    }
                    if (b.level<=2) return pot;
                    int base=(b.at(pr,pc).is_wildcard()?40:(b.at(pr,pc).color()==fixed?20:0));
                    return base+pot;
                };
                if(b.level<=2) { if(eval(nb[j])<eval(nb[i])) std::swap(nb[i],nb[j]); }
                else { if(eval(nb[j])>eval(nb[i])) std::swap(nb[i],nb[j]); }
            }
            for (int i=0;i<nb_cnt;++i) { auto [nr,nc]=nb[i]; path.push_back({nr,nc}); dfs(mask|bit(nr,nc,b.N),nr,nc,fixed); path.pop_back(); if (nodes>max_nodes) return; }
        };
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            path.clear(); path.push_back({r,c}); dfs(bit(r,c,b.N),r,c,0); if (nodes>max_nodes) break;
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
            auto [sc,bp]=exact_best_with_path(sim,tdead()?3000:12000);
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
                auto [sc,bp]=exact_best_with_path(bm.bd,d==beam_d-1?6000:3000);
                if (bp.size()<2) continue;
                Board nb=bm.bd.preview(bp);
                double hv=board_heuristic(nb);
                double total=bm.acc+(double)sc+hv*(d==beam_d-1?0.5:0.15);
                cands.push_back({std::move(nb),bm.acc+(double)sc,total});
                for (int tries=0;tries<4&&(int)cands.size()<beam_w*4;++tries) {
                    auto [sc2,bp2]=exact_best_with_path(bm.bd,1000+tries*500);
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
        tstart();
        board_N_ = board.N;
        bool pure5 = (board.level <= 2);
        int lmt = tfast()?120000:(board_N_<=5?800000:(pure5?1200000:500000));
        int cmax = cache_limit();

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
                nb_arr[nb_cnt++]={nr,nc}; }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                auto eval=[&](const std::pair<int,int>& p)->int{
                    int pr=p.first, pc=p.second, pot=0;
                    for(int dd=0;dd<4;++dd){
                        int ppr=pr+DR[dd], ppc=pc+DC[dd];
                        if(!board.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,board.N))) continue;
                        if(compatible_color(fixed,board.at(ppr,ppc).color())) pot++;
                    }
                    if (pure5) return pot;
                    int base=(board.at(pr,pc).is_wildcard()?40:(board.at(pr,pc).color()==fixed?20:0));
                    return base+pot;
                };
                if(pure5) { if(eval(nb_arr[j])<eval(nb_arr[i])) std::swap(nb_arr[i],nb_arr[j]); }
                else { if(eval(nb_arr[j])>eval(nb_arr[i])) std::swap(nb_arr[i],nb_arr[j]); }
            }
            for (int i=0;i<nb_cnt;++i) { auto [nr,nc]=nb_arr[i]; path.push_back({nr,nc}); dfs(mask|bit(nr,nc,board.N),nr,nc,fixed); path.pop_back(); if (nodes>lmt) return; }
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
            std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){ return cc_size[a.first][a.second]>cc_size[b.first][b.second]; });
        } else {
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) starts.push_back({r,c});
        }

        for (auto [r,c] : starts) {
            path.clear(); path.push_back({r,c}); dfs(bit(r,c,board.N),r,c,0); if (nodes>lmt) break;
        }
        if (raws.empty()) {
            for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) {
                path.clear(); path.push_back({r,c}); dfs(bit(r,c,board.N),r,c,0); if (nodes>lmt) break;
            }
        }
        if (raws.empty()) return fallback(board);

        std::vector<int> order(raws.size()); for (int i=0;i<(int)raws.size();++i) order[i]=i;
        std::sort(order.begin(),order.end(),[&](int a,int b) {
            int va=raws[a].now+raws[a].nxt, vb=raws[b].now+raws[b].nxt;
            if (pure5) {
                if ((int)raws[a].path.size()<6) va-=55;
                if ((int)raws[b].path.size()<6) vb-=55;
            }
            return va!=vb?va>vb:raws[a].now>raws[b].now; });
        int k1=std::min(keep1(),(int)raws.size()); order.resize(k1);

        struct Cand { std::vector<std::pair<int,int>> path; int now,nxt; Board next_b; double v2; };
        std::vector<Cand> cands; cands.reserve(k1);
        for (int idx:order) {
            Board nb=board.preview(raws[idx].path);
            double dq=drop_quality(nb);
            double bh=board_heuristic(nb);
            double v2=(double)raws[idx].now+(double)raws[idx].nxt+dq*0.8+bh*0.02;
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
                    val=(double)cands[i].now+(double)nsc+(double)tsc+drop_quality(tb)*0.55+board_heuristic(tb)*0.025;
                }
            }
            scored.push_back({val,i});
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        if (tdead()) return cands[scored[0].idx].path;

        int k3=std::min(keep3(),(int)scored.size());
        for (int j=0;j<k3;++j) {
            int st=survival_steps(cands[scored[j].idx].next_b,6);
            if (st<=2) scored[j].val-=120.0*(3.0-(double)st);
            else if (st<=3) scored[j].val-=50.0;
            else if (st<=4) scored[j].val-=15.0;
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});

        if (!tfast()) {
            int bs_cnt=std::min(3,(int)scored.size());
            for (int j=0;j<bs_cnt;++j) {
                double beam_val=beam_evaluate(cands[scored[j].idx].next_b,8,4);
                scored[j].val+=beam_val*0.3;
            }
            std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        }

        int sel = 0;

        if (cache2.size()>400000) cache2.clear();
        if (cache3.size()>300000) cache3.clear();
        return cands[scored[sel].idx].path;
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
        board_.drop_queue.assign(N, std::vector<int>(1000));
        board_.queue_ptr.assign(N, 0);

        // 生成 drop_queue
        std::mt19937 rng(seed);
        for (int c = 0; c < N; ++c)
            for (int i = 0; i < 1000; ++i)
                board_.drop_queue[c][i] = gen_block(rng, level);

        // 生成初始棋盘（使用独立但确定性的随机源）
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

    // 执行一步，返回是否还能继续
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

        // 炸弹额外得分
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

        // 消除与下落
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
                board_.at(i, c).value = board_.drop_queue[c][board_.queue_ptr[c]++];
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
int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    struct LevelConfig {
        int level;
        int seed;
        int N;
        const char* name;
    };

    std::vector<LevelConfig> levels = {
        {1, 42, 10, "Level 1 (10x10, 纯5色)"},
        {2, 42, 10, "Level 2 (10x10, 纯5色)"},
        {3, 42, 10, "Level 3 (10x10, 万能块15%)"},
        {4, 42, 10, "Level 4 (10x10, 炸弹10%)"},
        {5, 42, 12, "Level 5 (12x12, 万能块15%+炸弹10%)"},
    };

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
