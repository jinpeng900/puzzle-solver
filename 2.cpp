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
        b.drop_queue.assign(N,std::vector<int>(1000)); b.queue_ptr.assign(N,0);
        for (int c=0;c<N;++c) for (int i=0;i<1000;++i) b.drop_queue[c][i]=gen_block(rng,level);
    }
    static std::vector<int> removed_per_column(const Board& b, const std::vector<std::pair<int,int>>& path) {
        std::vector<int> removed(b.N,0);
        if (path.size()<2) return removed;
        std::vector<std::vector<bool>> in_path(b.N,std::vector<bool>(b.N)), to_remove(b.N,std::vector<bool>(b.N));
        for (auto [r,c]:path) in_path[r][c]=to_remove[r][c]=true;
        if (b.level>=4) for (auto [r,c]:path) {
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
    void respond(const std::vector<std::pair<int,int>>& path) { _last_path=path; std::cout<<path.size(); for (auto [r,c]:path) std::cout<<' '<<r<<' '<<c; std::cout<<'\n'; std::cout.flush(); }
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
// 求解器 vX：穷举 DFS + 多步前瞻 + 掉落仿真 + 时间管理
// ============================================================
class ImprovedSolver {
    static constexpr int MAX_TIME_MS = 10000;
    mutable std::chrono::steady_clock::time_point _t0;
    mutable bool _tinit = false;
    void tstart() const { if(!_tinit){_t0=std::chrono::steady_clock::now();_tinit=true;} }
    long long telapsed() const { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_t0).count(); }
    bool tfast()  const { return telapsed()>(long long)(MAX_TIME_MS*0.75); }
    bool tdead()  const { return telapsed()>(long long)(MAX_TIME_MS*0.92); }

    int dfs_limit() const { return tfast()?50000:(board_N_<=5?400000:200000); }
    int cache_limit()const { return tfast()?8000:30000; }
    int keep1()    const { return tfast()?50:120; }
    int keep2()    const { return tfast()?20:60; }
    int keep3()    const { return tfast()?3:10; }

    mutable int board_N_ = 6;
    mutable std::mt19937 rng{std::random_device{}()};

    // ======================== 盘面启发 ========================
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
                int sz=0, ac=co; std::deque<std::pair<int,int>> q;
                vis[r][c]=true; q.push_back({r,c});
                while (!q.empty()) { auto [cr,cc]=q.front(); q.pop_front(); sz++;
                    for (int d=0;d<4;++d) { int nr=cr+DR[d],nc=cc+DC[d]; if (!b.in_bounds(nr,nc)||vis[nr][nc]) continue;
                        int nco=b.at(nr,nc).color(); if (ac==0&&nco!=0) ac=nco;
                        if (nco!=0&&nco!=ac&&ac!=0) continue;
                        vis[nr][nc]=true; q.push_back({nr,nc}); }
                }
                mxc=std::max(mxc,sz);
            }
        }
        double ent=0.0; int tc=N*N-wc;
        if (tc>0) for (int i=1;i<=5;++i) if (cnt[i]>0) { double p=(double)cnt[i]/tc; ent-=p*std::log(p); }
        return cp*5.0+sp*8.0+wc*40.0+bb*12.0+mxc*12.0-ent*14.0;
    }

    // ======================== 掉落质量（向前看 7 个 + 兼容性评估） ========================
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

    // ======================== 快速最佳分（长度 2 检测 + BFS 短路径） ========================
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

    // ======================== 穷举最优单步分（无深度限制，带节点上限） ========================
    static int exact_best_one_step_score(const Board& b, int max_nodes) {
        int best=0, nodes=0;
        std::vector<std::pair<int,int>> path;
        std::function<void(std::uint64_t,int,int,int)> dfs=[&](std::uint64_t mask,int r,int c,int target) {
            if (++nodes>max_nodes) return;
            int co=b.at(r,c).color(); int fixed=target; if (fixed==0&&co!=0) fixed=co;
            if ((int)path.size()>=2) { int sc=path_score(b,path); if (sc>best) best=sc; }
            std::pair<int,int> nb[4]; int nb_cnt=0;
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (!b.in_bounds(nr,nc)||(mask&bit(nr,nc,b.N))) continue;
                if (!compatible_color(fixed,b.at(nr,nc).color())) continue;
                nb[nb_cnt++]={nr,nc}; }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                int wi=(b.at(nb[i].first,nb[i].second).is_wildcard()?2:b.at(nb[i].first,nb[i].second).color()==fixed?1:0);
                int wj=(b.at(nb[j].first,nb[j].second).is_wildcard()?2:b.at(nb[j].first,nb[j].second).color()==fixed?1:0);
                if (wj>wi) std::swap(nb[i],nb[j]);
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
            if ((int)path.size()>=2) { int sc=path_score(b,path); if (sc>best_score||(sc==best_score&&path<best_path)) { best_score=sc; best_path=path; } }
            std::pair<int,int> nb[4]; int nb_cnt=0;
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d]; if (!b.in_bounds(nr,nc)||(mask&bit(nr,nc,b.N))) continue;
                if (!compatible_color(fixed,b.at(nr,nc).color())) continue;
                nb[nb_cnt++]={nr,nc}; }
            for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
                int wi=(b.at(nb[i].first,nb[i].second).is_wildcard()?2:b.at(nb[i].first,nb[i].second).color()==fixed?1:0);
                int wj=(b.at(nb[j].first,nb[j].second).is_wildcard()?2:b.at(nb[j].first,nb[j].second).color()==fixed?1:0);
                if (wj>wi) std::swap(nb[i],nb[j]);
            }
            for (int i=0;i<nb_cnt;++i) { auto [nr,nc]=nb[i]; path.push_back({nr,nc}); dfs(mask|bit(nr,nc,b.N),nr,nc,fixed); path.pop_back(); if (nodes>max_nodes) return; }
        };
        for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
            path.clear(); path.push_back({r,c}); dfs(bit(r,c,b.N),r,c,0); if (nodes>max_nodes) break;
        }
        return {best_score,best_path};
    }

    // ======================== 掉落仿真：模拟 N 步盲目消除后的盘面 ========================
    double simulate_drops_and_score(const Board& b, int sim_steps) const {
        Board sim=b;
        double total=0;
        for (int st=0;st<sim_steps;++st) {
            if (sim.is_deadlocked()) break;
            auto [sc,bp]=exact_best_with_path(sim,std::max(2000,cache_limit()/sim_steps));
            if (bp.size()<2) break;
            total+=(double)sc*std::pow(0.7,(double)st);
            sim=sim.preview(bp);
        }
        return total;
    }

    // ======================== 死局预测 ========================
    int survival_steps(const Board& b, int max_check) const {
        Board sim=b; int i;
        for (i=0;i<max_check;++i) {
            if (sim.is_deadlocked()) break;
            auto [sc,bp]=exact_best_with_path(sim,tdead()?2000:8000);
            if (bp.size()<2) { i++; break; }
            sim=sim.preview(bp);
        }
        return i;
    }

    // ======================== Beam Search 多步前瞻 ========================
    double beam_evaluate(const Board& start_b, int beam_w, int beam_d) const {
        struct BNode { Board bd; double acc; };
        std::vector<BNode> beams; beams.push_back({start_b,0.0});
        for (int d=0;d<beam_d;++d) {
            struct Cand { Board bd; double acc; double val; };
            std::vector<Cand> cands;
            for (auto& bm:beams) {
                // 对每个 beam 找 top-15 路径
                auto [sc,bp]=exact_best_with_path(bm.bd,d==beam_d-1?4000:2000);
                if (bp.size()<2) continue;
                Board nb=bm.bd.preview(bp);
                double hv=board_heuristic(nb);
                double total=bm.acc+(double)sc+hv*(d==beam_d-1?0.5:0.15);
                cands.push_back({std::move(nb),bm.acc+(double)sc,total});
                // 额外尝试几个变体路径
                for (int tries=0;tries<3&&(int)cands.size()<beam_w*3;++tries) {
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
        int lmt = dfs_limit();
        int cmax = cache_limit();

        // ====== Step 1: 穷举 DFS + 两步精确前瞻 ======
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
                int wi=(board.at(nb_arr[i].first,nb_arr[i].second).is_wildcard()?2:board.at(nb_arr[i].first,nb_arr[i].second).color()==fixed?1:0);
                int wj=(board.at(nb_arr[j].first,nb_arr[j].second).is_wildcard()?2:board.at(nb_arr[j].first,nb_arr[j].second).color()==fixed?1:0);
                if (wj>wi) std::swap(nb_arr[i],nb_arr[j]);
            }
            for (int i=0;i<nb_cnt;++i) { auto [nr,nc]=nb_arr[i]; path.push_back({nr,nc}); dfs(mask|bit(nr,nc,board.N),nr,nc,fixed); path.pop_back(); if (nodes>lmt) return; }
        };
        for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) {
            path.clear(); path.push_back({r,c}); dfs(bit(r,c,board.N),r,c,0); if (nodes>lmt) break;
        }
        if (raws.empty()) return fallback(board);

        // 两步分排序取 top
        std::vector<int> order(raws.size()); for (int i=0;i<(int)raws.size();++i) order[i]=i;
        std::sort(order.begin(),order.end(),[&](int a,int b) {
            int va=raws[a].now+raws[a].nxt, vb=raws[b].now+raws[b].nxt;
            return va!=vb?va>vb:raws[a].now>raws[b].now; });
        int k1=std::min(keep1(),(int)raws.size()); order.resize(k1);

        // 重建 next_b + 掉落加分 + 启发值
        struct Cand { std::vector<std::pair<int,int>> path; int now,nxt; Board next_b; double v2; };
        std::vector<Cand> cands; cands.reserve(k1);
        for (int idx:order) {
            Board nb=board.preview(raws[idx].path);
            double dq=drop_quality(nb);
            double bh=board_heuristic(nb);
            double v2=(double)raws[idx].now+(double)raws[idx].nxt+dq*0.6+bh*0.015;
            cands.push_back({raws[idx].path,raws[idx].now,raws[idx].nxt,std::move(nb),v2});
        }
        std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.v2>b.v2;});
        if (tdead()) return cands[0].path;

        // ====== Step 2: 三步精确前瞻 ======
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
                    val=(double)cands[i].now+(double)nsc+(double)tsc+drop_quality(tb)*0.45+board_heuristic(tb)*0.018;
                }
            }
            scored.push_back({val,i});
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        if (tdead()) return cands[scored[0].idx].path;

        // ====== Step 3: 死局预测 ======
        int k3=std::min(keep3(),(int)scored.size());
        for (int j=0;j<k3;++j) {
            int st=survival_steps(cands[scored[j].idx].next_b,6);
            if (st<=2) scored[j].val-=80.0*(3.0-(double)st);
            else if (st<=3) scored[j].val-=30.0;
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});

        // ====== Step 4: 如果有大量时间剩余，对 top 3 做 Beam Search 深化 ======
        if (!tfast()) {
            int bs_cnt=std::min(3,(int)scored.size());
            for (int j=0;j<bs_cnt;++j) {
                double beam_val=beam_evaluate(cands[scored[j].idx].next_b,5,3);
                scored[j].val+=beam_val*0.3;
            }
            std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        }

        // ====== Step 5: ε-greedy 选择 ======
        double best_val=scored[0].val; int sel=0;
        int lim=std::min(8,(int)scored.size());
        double epsilon=tfast()?0.02:0.06;
        for (int i=1;i<lim;++i) {
            if (scored[i].val>=best_val) { best_val=scored[i].val; sel=i; }
            else if (std::uniform_real_distribution<double>(0,1)(rng)<epsilon) { sel=i; break; }
        }

        if (cache2.size()>200000) cache2.clear();
        if (cache3.size()>150000) cache3.clear();
        return cands[scored[sel].idx].path;
    }
};

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    GameController ctl;
    ImprovedSolver solver;
    while (ctl.update()) { auto path=solver.solve(ctl.board()); ctl.respond(path); }
    return 0;
}
