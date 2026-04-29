#include "solver_core.h"
#include <fstream>

// Global Params

LevelParams g_params[6];
int g_cur_level = 1;
int g_cur_step = 0;

// ???????????inline?????olver_core.h??
// ============================================================
// ?????????best_stable_v2.txt?????????????????
// ============================================================

void init_best_params() {
    // ??? best_stable_v2.txt + best_all_v2.txt ?????????????????
    g_params[1] = {1550000,350,65,29,110000, 30.0,1.15,0.04, 0.55,0.015, 0.8, 12,7,70,5, 0.55,0.05, 0.88,0.97,18000,6};
    g_params[2] = {1350000,200,95,12,90000,  24.0,0.50,0.025, 0.75,0.015, 1.0, 5,5,50,5, 0.40,0.45, 0.90,0.93,10000,6};
    g_params[3] = {1300000,200,180,12,60000, 22.5,1.50,0.0, 0.85,0.045, 2.8, 7,8,60,8, 0.0,0.25, 0.92,0.90,12000,4};
    g_params[4] = { 700000,140,165,10,75000,  16.5,1.10,0.025, 0.45,0.035, 1.2, 7,7,95,7, 0.30,0.15, 0.84,0.93,10000,4};
    g_params[5] = {1200000,55,40,16,60000,  15.0,0.75,0.065, 1.05,0.045, 0.7, 0,4,65,4, 0.15,0.40, 0.80,0.98,10000,3};
}

// ============================================================
// ??????????????
// ============================================================

PhaseMultiplier get_phase_multiplier(Phase ph, int level) {
    switch (ph) {
        case Phase::EARLY:    return {1.25, 1.15, 1.2, 1.2, 1.2, 1.3, 0.8, 1.3};
        case Phase::MID:      return {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        case Phase::LATE:     return {0.7, 0.75, 0.8, 0.7, 0.6, 0.5, 1.3, 0.3};
        case Phase::URGENT:   return {0.5, 0.5, 0.5, 0.5, 0.4, 0.3, 1.5, 0.1};
        case Phase::SURVIVAL: return {1.3, 1.0, 0.0, 0.0, 0.3, 0.2, 4.0, -0.5};
    }
    return {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
}

Phase detect_phase(const Board& board, int step) {
    int adj = board.count_adjacent_pairs();
    if (adj <= 3) return Phase::SURVIVAL;
    double mr = (double)step / 50.0;
    if (mr < 0.28) return Phase::EARLY;
    if (mr < 0.68) return Phase::MID;
    if (mr < 0.90) return Phase::LATE;
    return Phase::URGENT;
}

// ============================================================
// ?????????
// ============================================================

void ImprovedSolver::apply_phase_params(int level, int step) {
    current_phase_ = detect_phase(*current_board_ptr_, step);
    auto pm = get_phase_multiplier(current_phase_, level);

    phase_dfs_mul_ = pm.dfs_mul; phase_keep_mul_ = pm.keep_mul;
    phase_beam_w_mul_ = pm.beam_w_mul; phase_beam_d_mul_ = pm.beam_d_mul;
    phase_dq_mul_ = pm.dq_mul; phase_bh_mul_ = pm.bh_mul;
    phase_surv_mul_ = pm.surv_mul; phase_sp_mul_ = pm.short_pen_mul;

    auto& p = g_params[level];
    cur_cache_limit_ = p.cache_limit;
    cur_keep1_ = (int)(p.keep1 * phase_keep_mul_);
    cur_keep2_ = (int)(p.keep2 * phase_keep_mul_);
    cur_keep3_ = (int)(p.keep3 * phase_keep_mul_);
    mxc_weight_ = p.mxc_w;
    dq_weight_ = p.dq_w * phase_dq_mul_;
    bh_weight_ = p.bh_w * phase_bh_mul_;
    step3_dq_weight_ = p.s3_dq_w * phase_dq_mul_;
    step3_bh_weight_ = p.s3_bh_w * phase_bh_mul_;
    surv_mul_ = p.surv_mul * phase_surv_mul_;
    cur_beam_w_ = (int)(p.beam_w * phase_beam_w_mul_);
    cur_beam_d_ = (int)(p.beam_d * phase_beam_d_mul_);
}

// ============================================================
// ?????????
// ============================================================

double ImprovedSolver::board_heuristic(const Board& b) {
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
                for (int d=0;d<4;++d) { int nr=cr+DR[d],nc=cc+DC[d];
                    if (!b.in_bounds(nr,nc)||vis[nr][nc]) continue;
                    int nco=b.at(nr,nc).color();
                    if (!compatible_color(cco,nco)) continue;
                    vis[nr][nc]=true; q.push_back({nr,nc}); }
            }
            mxc=std::max(mxc,sz);
        }
    }
    double ent=0.0; int tc=N*N-wc;
    if (tc>0) for (int i=1;i<=5;++i)
        if (cnt[i]>0) { double p=(double)cnt[i]/tc; ent-=p*std::log(p); }
    bool is_bomb_level = (b.level == 4 || b.level == 5);
    double bb_weight = is_bomb_level ? 22.0 : 14.0;
    double h = cp*6.0+sp*10.0+wc*35.0+bb*bb_weight+mxc*g_params[b.level].mxc_w-ent*12.0;

    // ????????????
    if (g_nn_eval.loaded()) {
        double nn_val = g_nn_eval.predict(nn_extract_features(b));
        return h * 0.7 + nn_val * 0.3;
    }
    return h;
}

double ImprovedSolver::drop_quality(const Board& b) {
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
            if (b.in_bounds(1,c)&&compatible_color(col,b.at(1,c).color())) adj+=0.3;
            q+=adj;
            if (i<3) for (int j=i+1;j<3&&j<7;++j) {
                int v2=dq[c][qp+j]; if (v2==0||v==v2) q+=0.2;
            }
        }
    }
    return q;
}

// ============================================================
// DFS??????????????????
// ============================================================

int ImprovedSolver::upper_bound_reachable(const Board& b, const std::vector<std::pair<int,int>>& path, int target) {
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
    int reachable=0, black=0, white=0;
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

int ImprovedSolver::exact_best_one_step_score(const Board& b, int max_nodes) {
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
        for (int d=0;d<4;++d) {
            int nr=r+DR[d],nc=c+DC[d];
            if (!b.in_bounds(nr,nc)||vis[nr*b.N+nc]) continue;
            if (!compatible_color(fixed,b.at(nr,nc).color())) continue;
            nb[nb_cnt++]={nr,nc};
        }
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
        for (int i=0;i<nb_cnt;++i) {
            auto [nr,nc]=nb[i];
            path.push_back({nr,nc}); vis[nr*b.N+nc]=true;
            dfs(nr,nc,fixed);
            vis[nr*b.N+nc]=false; path.pop_back();
            if (nodes>max_nodes) return;
        }
    };
    for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
        path.clear(); path.push_back({r,c});
        vis.assign(b.N*b.N,false); vis[r*b.N+c]=true;
        dfs(r,c,0);
        if (nodes>max_nodes) break;
    }
    return best;
}

std::pair<int,std::vector<std::pair<int,int>>>
ImprovedSolver::exact_best_with_path(const Board& b, int max_nodes) {
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
        for (int d=0;d<4;++d) {
            int nr=r+DR[d],nc=c+DC[d];
            if (!b.in_bounds(nr,nc)||vis[nr*b.N+nc]) continue;
            if (!compatible_color(fixed,b.at(nr,nc).color())) continue;
            nb[nb_cnt++]={nr,nc};
        }
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
        for (int i=0;i<nb_cnt;++i) {
            auto [nr,nc]=nb[i];
            path.push_back({nr,nc}); vis[nr*b.N+nc]=true;
            dfs(nr,nc,fixed);
            vis[nr*b.N+nc]=false; path.pop_back();
            if (nodes>max_nodes) return;
        }
    };
    for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
        path.clear(); path.push_back({r,c});
        vis.assign(b.N*b.N,false); vis[r*b.N+c]=true;
        dfs(r,c,0);
        if (nodes>max_nodes) break;
    }
    return {best_score,best_path};
}

// ============================================================
// Beam Search & ???????// ============================================================

int ImprovedSolver::survival_steps(const Board& b, int max_check) const {
    Board sim=b; int i;
    for (i=0;i<max_check;++i) {
        if (sim.is_deadlocked()) break;
        auto [sc,bp]=exact_best_with_path(sim,tdead()?5000:20000);
        if (bp.size()<2) { i++; break; }
        sim=sim.preview(bp);
    }
    return i;
}

double ImprovedSolver::beam_evaluate(const Board& start_b, int beam_w, int beam_d) const {
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
        for (int i=0;i<std::min(beam_w,(int)cands.size());++i)
            beams.push_back({std::move(cands[i].bd),cands[i].acc});
    }
    return beams.empty()?0.0:beams[0].acc;
}

std::vector<std::pair<int,int>> ImprovedSolver::fallback(const Board& b) {
    for (int r=0;r<b.N;++r) for (int c=0;c<b.N;++c) {
        if (b.at(r,c).is_wildcard())
            for (int d=0;d<4;++d) { int nr=r+DR[d],nc=c+DC[d];
                if (b.in_bounds(nr,nc)) return {{r,c},{nr,nc}}; }
        int a=b.at(r,c).color();
        if (c+1<b.N&&compatible_color(a,b.at(r,c+1).color()))
            return {{r,c},{r,c+1}};
        if (r+1<b.N&&compatible_color(a,b.at(r+1,c).color()))
            return {{r,c},{r+1,c}};
    }
    return {{0,0},{0,1}};
}

// ============================================================
// ????????// ============================================================

std::vector<std::pair<int,int>> ImprovedSolver::solve(const Board& board) {
    _tinit = false;
    tstart();
    ++step_count_;
    board_N_ = board.N;
    g_cur_level = board.level;
    current_board_ptr_ = &board;
    g_cur_step = step_count_;

    apply_phase_params(board.level, step_count_);

    bool pure5 = (board.level <= 2);
    int lmt = (int)(g_params[board.level].dfs_limit * phase_dfs_mul_);
    int cmax = cache_limit();

    struct Raw { std::vector<std::pair<int,int>> path; int now; int nxt; };
    std::vector<Raw> raws; raws.reserve(8192);

    static std::unordered_map<std::uint64_t,int> cache2;
    int nodes=0;
    std::vector<std::pair<int,int>> path;

    std::vector<bool> vis_main(board.N*board.N, false);
    std::function<void(int,int,int)> dfs=[&](int r,int c,int target) {
        if (++nodes>lmt) return;
        int cur=board.at(r,c).color(); int fixed=target;
        if (fixed==0&&cur!=0) fixed=cur;
        if ((int)path.size()>=2) {
            int ns=path_score(board,path);
            Board nb=board.preview(path);
            auto h=board_hash(nb);
            int nxt=0;
            auto it=cache2.find(h);
            if (it!=cache2.end()) nxt=it->second;
            else { nxt=exact_best_one_step_score(nb,cmax); cache2[h]=nxt; }
            raws.push_back({path,ns,nxt});
        }
        std::pair<int,int> nb_arr[4]; int nb_cnt=0;
        for (int d=0;d<4;++d) {
            int nr=r+DR[d],nc=c+DC[d];
            if (!board.in_bounds(nr,nc)||vis_main[nr*board.N+nc]) continue;
            if (!compatible_color(fixed,board.at(nr,nc).color())) continue;
            nb_arr[nb_cnt++]={nr,nc};
        }
        for (int i=0;i<nb_cnt;++i) for (int j=i+1;j<nb_cnt;++j) {
            auto eval=[&](const std::pair<int,int>& p)->int{
                int pr=p.first, pc=p.second, pot=0;
                for(int dd=0;dd<4;++dd){
                    int ppr=pr+DR[dd], ppc=pc+DC[dd];
                    if(!board.in_bounds(ppr,ppc)||vis_main[ppr*board.N+ppc]) continue;
                    if(compatible_color(fixed,board.at(ppr,ppc).color())) pot++;
                }
                if (pure5) return pot;
                int base=(board.at(pr,pc).is_wildcard()?40:
                         (board.at(pr,pc).is_bomb()?35:
                         (board.at(pr,pc).color()==fixed?20:0)));
                return base+pot;
            };
            if(pure5) { if(eval(nb_arr[j])<eval(nb_arr[i])) std::swap(nb_arr[i],nb_arr[j]); }
            else { if(eval(nb_arr[j])>eval(nb_arr[i])) std::swap(nb_arr[i],nb_arr[j]); }
        }
        for (int i=0;i<nb_cnt;++i) {
            auto [nr,nc]=nb_arr[i];
            path.push_back({nr,nc}); vis_main[nr*board.N+nc]=true;
            dfs(nr,nc,fixed);
            vis_main[nr*board.N+nc]=false; path.pop_back();
            if (nodes>lmt) return;
        }
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
            while (!q.empty()) {
                auto [cr,cc]=q.front(); q.pop_front(); cells.push_back({cr,cc});
                for (int d=0;d<4;++d) {
                    int nr=cr+DR[d],nc=cc+DC[d];
                    if (!board.in_bounds(nr,nc)||vis[nr][nc]) continue;
                    if (board.at(nr,nc).color()!=co) continue;
                    vis[nr][nc]=true; q.push_back({nr,nc});
                }
            }
            for (auto& p:cells) cc_size[p.first][p.second]=(int)cells.size();
        }
        for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c)
            starts.push_back({r,c});
        if (board.level==1) {
            std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){
                return cc_size[a.first][a.second]>cc_size[b.first][b.second]; });
        } else {
            std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){
                return cc_size[a.first][a.second]<cc_size[b.first][b.second]; });
        }
    } else {
        for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c)
            starts.push_back({r,c});
        std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){
            int va = board.at(a.first,a.second).is_wildcard() ? 2 :
                     board.at(a.first,a.second).is_bomb() ? 1 : 0;
            int vb = board.at(b.first,b.second).is_wildcard() ? 2 :
                     board.at(b.first,b.second).is_bomb() ? 1 : 0;
            return va > vb;
        });
    }

    for (auto [r,c] : starts) {
        path.clear(); path.push_back({r,c});
        vis_main.assign(board.N*board.N,false); vis_main[r*board.N+c]=true;
        dfs(r,c,0);
        if (nodes>lmt) break;
    }

    if (raws.empty()) {
        for (int r=0;r<board.N;++r) for (int c=0;c<board.N;++c) {
            path.clear(); path.push_back({r,c});
            vis_main.assign(board.N*board.N,false); vis_main[r*board.N+c]=true;
            dfs(r,c,0);
            if (nodes>lmt) break;
        }
    }

    if (raws.empty()) return fallback(board);

    std::vector<int> order(raws.size());
    for (int i=0;i<(int)raws.size();++i) order[i]=i;
    std::sort(order.begin(),order.end(),[&](int a,int b) {
        int va=raws[a].now+raws[a].nxt, vb=raws[b].now+raws[b].nxt;
        if (pure5) {
            int pa=(int)raws[a].path.size(), pb=(int)raws[b].path.size();
            int sp = (int)(g_params[board.level].short_pen * phase_sp_mul_);
            if (pa<6) va-=sp;
            if (pb<6) vb-=sp;
        }
        return va!=vb?va>vb:raws[a].now>raws[b].now;
    });
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
                Board tb=cands[i].next_b.preview(np);
                auto th=board_hash(tb);
                int tsc=0;
                auto it=cache3.find(th);
                if (it!=cache3.end()) tsc=it->second;
                else { tsc=exact_best_one_step_score(tb,cmax/3); cache3[th]=tsc; }
                val=(double)cands[i].now+(double)nsc+(double)tsc
                    +drop_quality(tb)*step3_dq_weight_
                    +board_heuristic(tb)*step3_bh_weight_;
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

    if (cache2.size()>400000) cache2.clear();
    if (cache3.size()>300000) cache3.clear();

    return cands[scored[0].idx].path;
}