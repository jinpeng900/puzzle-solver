/**
 * test_2.cpp — 本地测试 2.cpp 求解器性能
 * 编译: g++ -std=c++14 -O2 -o test_2.exe test_2.cpp
 * 用法: ./test_2.exe [num_seeds] [start_seed]
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
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
#include <iostream>
#include <iomanip>

// ============================================================
// 本地评测机 (LocalJudge)
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

    Board preview(const std::vector<std::pair<int,int>>& path) const {
        Board nb = *this; if (path.size()<2) return nb;
        auto& dq = *nb.drop_queue;
        std::vector<std::vector<bool>> in_path(N,std::vector<bool>(N));
        for (auto& p : path) in_path[p.first][p.second]=true;
        auto to_remove = in_path;
        if (level>=4) for (auto& p : path) { int r=p.first,c=p.second;
            if (!at(r,c).is_bomb()) continue;
            for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
                int nr=r+dr,nc=c+dc;
                if (in_bounds(nr,nc)&&!in_path[nr][nc]) to_remove[nr][nc]=true;
            }
        }
        for (int c=0;c<N;++c) {
            std::vector<Cell> rem;
            for (int r=0;r<N;++r) if(!to_remove[r][c]) rem.push_back(at(r,c));
            int empty=N-(int)rem.size();
            for (int i=0;i<empty;++i) nb.at(i,c).value=dq[c][nb.queue_ptr[c]++];
            for (int i=0;i<(int)rem.size();++i) nb.at(empty+i,c)=rem[i];
        }
        return nb;
    }

    bool is_deadlocked() const {
        for (int r=0;r<N;++r) for (int c=0;c<N;++c) {
            int ac=at(r,c).color();
            if (c+1<N){int c2=at(r,c+1).color(); if(ac==c2||ac==0||c2==0) return false;}
            if (r+1<N){int c2=at(r+1,c).color(); if(ac==c2||ac==0||c2==0) return false;}
        }
        return true;
    }
};

int gen_block(std::mt19937& rng, int level) {
    if (level<=2) return (rng()%5)+1;
    if (level==3) return ((rng()%100)<15)?0:(rng()%5)+1;
    if (level==4){int c=(rng()%5)+1;return ((rng()%100)<10)?-c:c;}
    if ((rng()%100)<15) return 0;
    int base=(rng()%5)+1;return ((rng()%100)<10)?-base:base;
}

Board init_board(int level, int seed, int N) {
    Board b(N); b.level=level;
    b.drop_queue->assign(N,std::vector<int>(1000));
    b.queue_ptr.assign(N,0);
    std::mt19937 rng(seed);
    for (int c=0;c<N;++c) for (int i=0;i<1000;++i) (*b.drop_queue)[c][i]=gen_block(rng,level);
    std::mt19937 rng_b(seed^0x9E3779B9);
    for (int r=0;r<N;++r) for (int c=0;c<N;++c) b.at(r,c).value=gen_block(rng_b,level);
    return b;
}

int path_score(int k) { double t=std::sqrt((double)k)-1.0; return 10*k+18*(int)(t*t); }
int path_score(const Board& b, const std::vector<std::pair<int,int>>& path) {
    int k=(int)path.size(),s=path_score(k);
    std::vector<std::vector<bool>> in_path(b.N,std::vector<bool>(b.N));
    for (auto& p : path) in_path[p.first][p.second]=true;
    std::vector<std::vector<bool>> expl(b.N,std::vector<bool>(b.N));
    for (auto& p : path) { int r=p.first,c=p.second;
        if (!b.at(r,c).is_bomb()) continue;
        for (int dr=-1;dr<=1;++dr) for (int dc=-1;dc<=1;++dc) {
            int nr=r+dr,nc=c+dc;
            if (b.in_bounds(nr,nc)&&!in_path[nr][nc]&&!expl[nr][nc]){expl[nr][nc]=true;s+=10;}
        }
    }
    return s;
}

// ============================================================
// 简化求解器（用于基线对比）
// 与 1.cpp 的核心逻辑一致
// ============================================================
constexpr int DR[]={-1,1,0,0}, DC[]={0,0,-1,1};
static bool comp(int a,int b){return a==0||b==0||a==b;}
static uint64_t bit2(int r,int c,int N){int idx=r*N+c;return (idx<64)?(1ULL<<idx):0;}
static uint64_t hash_board(const Board& b){
    uint64_t h=1469598103934665603ULL, prime=1099511628211ULL;
    h^=(uint64_t)b.N;h*=prime;h^=(uint64_t)b.level;h*=prime;
    for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){h^=(uint64_t)(b.at(r,c).value+17);h*=prime;}
    return h;
}

struct SimpleSolver {
    int cache_limit=60000, keep1=200, keep2=100, keep3=20;
    int beam_w=8, beam_d=4;
    double dq_w=0.8, bh_w=0.02, surv_m=1.0, s3dq=0.55, s3bh=0.025;

    int upper_bound(const Board& b, const std::vector<std::pair<int,int>>& path, int target) {
        std::vector<std::vector<bool>> blocked(b.N,std::vector<bool>(b.N));
        for(auto& p:path)blocked[p.first][p.second]=true;
        int tr=path.back().first,tc=path.back().second;
        std::deque<std::pair<int,int>> q;std::vector<std::vector<bool>> seen(b.N,std::vector<bool>(b.N));
        for(int d=0;d<4;++d){int nr=tr+DR[d],nc=tc+DC[d];if(!b.in_bounds(nr,nc)||blocked[nr][nc])continue;
            if(!comp(target,b.at(nr,nc).color()))continue;seen[nr][nc]=true;q.push_back({nr,nc});}
        int reach=0,black=0,white=0;
        for(auto&p:path){if((p.first+p.second)%2==0)black++;else white++;}
        while(!q.empty()){auto p=q.front();q.pop_front();++reach;
            if((p.first+p.second)%2==0)black++;else white++;
            for(int d=0;d<4;++d){int nr=p.first+DR[d],nc=p.second+DC[d];
                if(!b.in_bounds(nr,nc)||blocked[nr][nc]||seen[nr][nc])continue;
                if(!comp(target,b.at(nr,nc).color()))continue;seen[nr][nc]=true;q.push_back({nr,nc});}}
        int ub=(int)path.size()+reach;
        if(b.level<=2||b.level==4){int ub2=std::min(black,white)*2;if(black!=white)ub2++;return std::min(ub,ub2);}
        return ub;
    }

    int best_one_step(const Board& b, int maxn) {
        int best=0,nodes=0;std::vector<std::pair<int,int>> path;
        std::function<void(uint64_t,int,int,int)> dfs=[&](uint64_t mask,int r,int c,int target){
            if(++nodes>maxn)return;int co=b.at(r,c).color();int fixed=target;if(fixed==0&&co!=0)fixed=co;
            if(path.size()>=2){int sc=path_score(b,path);if(sc>best)best=sc;
                int ub=upper_bound(b,path,fixed);if(path_score(ub)<=best)return;}
            std::pair<int,int> nb[4];int nbc=0;
            for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!b.in_bounds(nr,nc)||(mask&bit2(nr,nc,b.N)))continue;
                if(!comp(fixed,b.at(nr,nc).color()))continue;nb[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){
                auto ev=[&](auto&p)->int{int pr=p.first,pc=p.second,pot=0;
                    for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||(mask&bit2(ppr,ppc,b.N)))continue;
                        if(comp(fixed,b.at(ppr,ppc).color()))pot++;}
                    int base=0;if(b.at(pr,pc).is_wildcard())base=40;
                    else if(b.at(pr,pc).is_bomb())base=(b.level==4?35:25);
                    else if(b.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(ev(nb[j])>ev(nb[i]))std::swap(nb[i],nb[j]);}
            for(int i=0;i<nbc;++i){int nr=nb[i].first,nc=nb[i].second;path.push_back({nr,nc});
                dfs(mask|bit2(nr,nc,b.N),nr,nc,fixed);path.pop_back();if(nodes>maxn)return;}
        };
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){
            path.clear();path.push_back({r,c});dfs(bit2(r,c,b.N),r,c,0);if(nodes>maxn)break;}
        return best;
    }

    std::pair<int,std::vector<std::pair<int,int>>> best_with_path(const Board& b, int maxn) {
        int bs=0,nodes=0;std::vector<std::pair<int,int>> bp,path;
        std::function<void(uint64_t,int,int,int)> dfs=[&](uint64_t mask,int r,int c,int target){
            if(++nodes>maxn)return;int co=b.at(r,c).color();int fixed=target;if(fixed==0&&co!=0)fixed=co;
            if(path.size()>=2){int sc=path_score(b,path);
                if(sc>bs||(sc==bs&&path<bp)){bs=sc;bp=path;}
                int ub=upper_bound(b,path,fixed);if(path_score(ub)<=bs)return;}
            std::pair<int,int> nb[4];int nbc=0;
            for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!b.in_bounds(nr,nc)||(mask&bit2(nr,nc,b.N)))continue;
                if(!comp(fixed,b.at(nr,nc).color()))continue;nb[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){
                auto ev=[&](auto&p)->int{int pr=p.first,pc=p.second,pot=0;
                    for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||(mask&bit2(ppr,ppc,b.N)))continue;
                        if(comp(fixed,b.at(ppr,ppc).color()))pot++;}
                    int base=0;if(b.at(pr,pc).is_wildcard())base=40;
                    else if(b.at(pr,pc).is_bomb())base=(b.level==4?35:25);
                    else if(b.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(ev(nb[j])>ev(nb[i]))std::swap(nb[i],nb[j]);}
            for(int i=0;i<nbc;++i){path.push_back(nb[i]);dfs(mask|bit2(nb[i].first,nb[i].second,b.N),nb[i].first,nb[i].second,fixed);path.pop_back();if(nodes>maxn)return;}
        };
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){
            path.clear();path.push_back({r,c});dfs(bit2(r,c,b.N),r,c,0);if(nodes>maxn)break;}
        return {bs,bp};
    }

    int survival_steps(const Board& b, int maxc) {
        Board sim=b;int i;for(i=0;i<maxc;++i){if(sim.is_deadlocked())break;
            auto sbp=best_with_path(sim,15000);if(sbp.second.size()<2){i++;break;}sim=sim.preview(sbp.second);}return i;
    }

    double beam_eval(const Board& sb, int bw, int bd) {
        struct BN{Board bd;double acc;};std::vector<BN> beams;beams.push_back({sb,0.0});
        for(int d=0;d<bd;++d){struct CN{Board bd;double acc,val;};std::vector<CN> cands;
            for(auto& bm:beams){auto sbp=best_with_path(bm.bd,d==bd-1?10000:5000);
                if(sbp.second.size()<2)continue;Board nb=bm.bd.preview(sbp.second);
                double hv=0;{if(nb.is_deadlocked())hv=-1e5;else{/* simplified heuristic */}}hv=sbp.first*0.5;
                double total=bm.acc+(double)sbp.first+hv*(d==bd-1?0.5:0.15);
                cands.push_back({std::move(nb),bm.acc+(double)sbp.first,total});}
            if(cands.empty())break;std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.val>b.val;});
            beams.clear();for(int i=0;i<std::min(bw,(int)cands.size());++i)beams.push_back({std::move(cands[i].bd),cands[i].acc});}
        return beams.empty()?0.0:beams[0].acc;
    }

    double drop_q(const Board& b) {
        double q=0;auto& dq=*b.drop_queue;
        for(int c=0;c<b.N;++c){int qp=b.queue_ptr[c];
            for(int i=0;i<7;++i){if(qp+i>=1000)break;int v=dq[c][qp+i];
                if(v==0){q+=2.8;continue;}if(v<0){q+=0.9;continue;}
                int col=std::abs(v);double adj=0;
                if(b.in_bounds(0,c-1)&&comp(col,b.at(0,c-1).color()))adj+=0.5;
                if(b.in_bounds(0,c+1)&&comp(col,b.at(0,c+1).color()))adj+=0.5;
                if(b.in_bounds(1,c)&&comp(col,b.at(1,c).color()))adj+=0.3;q+=adj;}}
        return q;
    }

    double board_h(const Board& b) {
        if(b.is_deadlocked())return -1e5;int N=b.N,cp=0,sp=0,wc=0,bb=0,cnt[6]={},mxc=0;
        std::vector<std::vector<bool>> vis(N,std::vector<bool>(N));
        for(int r=0;r<N;++r)for(int c=0;c<N;++c){int co=b.at(r,c).color();
            if(b.at(r,c).is_wildcard())wc++;if(b.at(r,c).is_bomb())bb++;cnt[std::min(co,5)]++;
            if(c+1<N){int c2=b.at(r,c+1).color();if(comp(co,c2)){cp++;if(co==c2&&co!=0)sp++;}}
            if(r+1<N){int c2=b.at(r+1,c).color();if(comp(co,c2)){cp++;if(co==c2&&co!=0)sp++;}}
            if(!vis[r][c]){int sz=0;std::deque<std::pair<int,int>> q;q.push_back({r,c});vis[r][c]=true;
                while(!q.empty()){auto p=q.front();q.pop_front();sz++;int cco=b.at(p.first,p.second).color();
                    for(int d=0;d<4;++d){int nr=p.first+DR[d],nc=p.second+DC[d];
                        if(!b.in_bounds(nr,nc)||vis[nr][nc])continue;
                        if(!comp(cco,b.at(nr,nc).color()))continue;vis[nr][nc]=true;q.push_back({nr,nc});}}
                mxc=std::max(mxc,sz);}}
        double ent=0;int tc=N*N-wc;if(tc>0)for(int i=1;i<=5;++i)if(cnt[i]>0){double p=(double)cnt[i]/tc;ent-=p*std::log(p);}
        double bbw=(b.level==4||b.level==5)?22.0:15.0;bool pp=(b.level<=2);
        return cp*6.0+sp*10.0+wc*35.0+bb*bbw+mxc*(pp?25.0:15.0)-ent*12.0;
    }

    std::vector<std::pair<int,int>> solve(const Board& board) {
        int lmt=(board.N<=5?800000:(board.level<=2?1200000:(board.level==5?1200000:700000)));
        int cmax=cache_limit;bool pure5=(board.level<=2);

        struct Raw{std::vector<std::pair<int,int>> path;int now,nxt;};
        std::vector<Raw> raws;raws.reserve(8192);
        static std::unordered_map<uint64_t,int> cache2;int nodes=0;
        std::vector<std::pair<int,int>> path;

        std::function<void(uint64_t,int,int,int)> dfs=[&](uint64_t mask,int r,int c,int target){
            if(++nodes>lmt)return;int cur=board.at(r,c).color();int fixed=target;if(fixed==0&&cur!=0)fixed=cur;
            if(path.size()>=2){int ns=path_score(board,path);Board nb=board.preview(path);
                auto h=hash_board(nb);int nxt=0;auto it=cache2.find(h);if(it!=cache2.end())nxt=it->second;
                else{nxt=best_one_step(nb,cmax);cache2[h]=nxt;}raws.push_back({path,ns,nxt});}
            std::pair<int,int> nb2[4];int nbc=0;
            for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!board.in_bounds(nr,nc)||(mask&bit2(nr,nc,board.N)))continue;
                if(!comp(fixed,board.at(nr,nc).color()))continue;nb2[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){
                auto ev=[&](auto&p)->int{int pr=p.first,pc=p.second,pot=0;
                    for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                        if(!board.in_bounds(ppr,ppc)||(mask&bit2(ppr,ppc,board.N)))continue;
                        if(comp(fixed,board.at(ppr,ppc).color()))pot++;}
                    if(pure5)return pot;int base=(board.at(pr,pc).is_wildcard()?40:(board.at(pr,pc).color()==fixed?20:0));return base+pot;};
                if(pure5){if(ev(nb2[j])<ev(nb2[i]))std::swap(nb2[i],nb2[j]);}
                else{if(ev(nb2[j])>ev(nb2[i]))std::swap(nb2[i],nb2[j]);}}
            for(int i=0;i<nbc;++i){path.push_back(nb2[i]);dfs(mask|bit2(nb2[i].first,nb2[i].second,board.N),nb2[i].first,nb2[i].second,fixed);path.pop_back();if(nodes>lmt)return;}
        };

        for(int r=0;r<board.N;++r)for(int c=0;c<board.N;++c){
            path.clear();path.push_back({r,c});dfs(bit2(r,c,board.N),r,c,0);if(nodes>lmt)break;}
        if(raws.empty()){return{{0,0},{0,1}};}

        std::vector<int> order(raws.size());for(int i=0;i<(int)raws.size();++i)order[i]=i;
        std::sort(order.begin(),order.end(),[&](int a,int b){int va=raws[a].now+raws[a].nxt,vb=raws[b].now+raws[b].nxt;
            if(pure5){if((int)raws[a].path.size()<6)va-=55;if((int)raws[b].path.size()<6)vb-=55;}
            return va!=vb?va>vb:raws[a].now>raws[b].now;});
        int k1=std::min(keep1,(int)raws.size());order.resize(k1);

        struct Cand{std::vector<std::pair<int,int>> path;int now,nxt;Board next_b;double v2;};
        std::vector<Cand> cands;cands.reserve(k1);
        for(int idx:order){Board nb=board.preview(raws[idx].path);
            double v2=(double)raws[idx].now+(double)raws[idx].nxt+drop_q(nb)*dq_w+board_h(nb)*bh_w;
            cands.push_back({raws[idx].path,raws[idx].now,raws[idx].nxt,std::move(nb),v2});}
        std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.v2>b.v2;});

        static std::unordered_map<uint64_t,int> cache3;int k2=std::min(keep2,(int)cands.size());
        struct Scored{double val;int idx;};std::vector<Scored> scored;scored.reserve(k1);
        for(int i=0;i<k1;++i){double val=cands[i].v2;
            if(i<k2){auto sbp=best_with_path(cands[i].next_b,cmax);
                if(sbp.second.size()>=2){Board tb=cands[i].next_b.preview(sbp.second);auto th=hash_board(tb);
                    int tsc=0;auto it=cache3.find(th);if(it!=cache3.end())tsc=it->second;
                    else{tsc=best_one_step(tb,cmax/3);cache3[th]=tsc;}
                    val=(double)cands[i].now+(double)sbp.first+(double)tsc+drop_q(tb)*s3dq+board_h(tb)*s3bh;}}
            scored.push_back({val,i});}
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});

        int k3=std::min(keep3,(int)scored.size());
        for(int j=0;j<k3;++j){int st=survival_steps(cands[scored[j].idx].next_b,6);
            scored[j].val+=(double)st*surv_m*0.15;}
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});

        if(beam_w>0){int bsc=std::min(3,(int)scored.size());
            for(int j=0;j<bsc;++j){scored[j].val+=beam_eval(cands[scored[j].idx].next_b,beam_w,beam_d)*0.3;}
            std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});}

        if(cache2.size()>400000)cache2.clear();if(cache3.size()>300000)cache3.clear();
        return cands[scored[0].idx].path;
    }
};

// ============================================================
// 主测试
// ============================================================
int main(int argc, char** argv) {
    int num_seeds = (argc>1)?std::atoi(argv[1]):20;
    int start_seed = (argc>2)?std::atoi(argv[2]):1;

    printf("=== Testing 2.cpp solver over %d seeds (start=%d) ===\n\n", num_seeds, start_seed);

    // Per-level params (matching 1.cpp)
    struct Lvl{int l,N;double dq,bh,surv,s3dq,s3bh;int cl,k1,k2,k3,bw,bd;};
    Lvl levels[]={
        {1,10,0.8,0.02,0.5,0.55,0.025,80000,200,100,20,8,4},
        {2,10,0.8,0.02,1.0,0.55,0.025,80000,200,100,20,8,4},
        {3,10,0.9,0.02,1.0,0.60,0.025,60000,120,60,12,2,2},   // ★L3 beam=2
        {4,10,0.8,0.02,1.0,0.55,0.025,80000,120,60,12,5,3},
        {5,12,0.9,0.02,1.0,0.60,0.025,50000,60,30,6,2,2},     // ★L5 beam=2
    };

    int grand_total = 0;
    int level_totals[5] = {0};

    for (int si=0; si<num_seeds; ++si) {
        int seed = start_seed + si;
        int seed_total = 0;
        for (auto& lc : levels) {
            Board board = init_board(lc.l, seed, lc.N);
            SimpleSolver solver;
            solver.cache_limit=lc.cl; solver.keep1=lc.k1; solver.keep2=lc.k2; solver.keep3=lc.k3;
            solver.dq_w=lc.dq; solver.bh_w=lc.bh; solver.surv_m=lc.surv;
            solver.s3dq=lc.s3dq; solver.s3bh=lc.s3bh;
            solver.beam_w=lc.bw; solver.beam_d=lc.bd;

            int step=0, score=0;
            while (step<50 && !board.is_deadlocked()) {
                auto path = solver.solve(board);
                if (path.size()<2) break;
                int gained = path_score(board, path);
                score += gained;
                step++;
                board = board.preview(path);
            }
            seed_total += score;
            level_totals[lc.l-1] += score;
        }
        grand_total += seed_total;
        if ((si+1)%10==0) printf("  seed %d: %d (avg %.1f)\n", seed, seed_total, (double)grand_total/(si+1));
    }

    printf("\n=== RESULTS (%d seeds) ===\n", num_seeds);
    for (int i=0;i<5;++i) printf("  L%d avg: %.1f\n", i+1, (double)level_totals[i]/num_seeds);
    printf("  TOTAL avg: %.1f\n", (double)grand_total/num_seeds);
    printf("  TOTAL sum: %d\n", grand_total);
    return 0;
}
