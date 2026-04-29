// 并行 nxt 评估测试版
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <deque>
#include <unordered_map>
#include <cstdint>
#include <set>
#include <memory>
#include <algorithm>
#include <cmath>
#include <array>
#include <limits>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>

struct Cell { int value=1; int color()const{return std::abs(value);} bool is_bomb()const{return value<0;} bool is_wildcard()const{return value==0;} };
struct Board {
    int N=0,level=1;
    std::vector<std::vector<Cell>> grid;
    std::shared_ptr<std::vector<std::vector<int>>> drop_queue;
    std::vector<int> queue_ptr;
    explicit Board(int n=0):N(n),grid(n,std::vector<Cell>(n)),drop_queue(std::make_shared<std::vector<std::vector<int>>>()){}
    Cell& at(int r,int c){return grid[r][c];}
    const Cell& at(int r,int c)const{return grid[r][c];}
    bool in_bounds(int r,int c)const{return r>=0&&r<N&&c>=0&&c<N;}
    Board preview(const std::vector<std::pair<int,int>>& path)const{
        Board nb=*this;
        if(path.size()<2)return nb;
        auto& dq=*nb.drop_queue;
        std::vector<std::vector<bool>> in_path(N,std::vector<bool>(N));
        for(auto p:path)in_path[p.first][p.second]=true;
        std::vector<std::vector<bool>> to_remove=in_path;
        if(level>=4)for(auto [r,c]:path){if(!at(r,c).is_bomb())continue;
            for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){int nr=r+dr,nc=c+dc;
                if(in_bounds(nr,nc)&&!in_path[nr][nc])to_remove[nr][nc]=true;}}
        for(int c=0;c<N;++c){
            std::vector<Cell> rem;
            for(int r=0;r<N;++r)if(!to_remove[r][c])rem.push_back(at(r,c));
            int empty=N-(int)rem.size();
            for(int i=0;i<empty;++i)nb.at(i,c).value=dq[c][nb.queue_ptr[c]++];
            for(int i=0;i<(int)rem.size();++i)nb.at(empty+i,c)=rem[i];
        }
        return nb;
    }
    bool is_deadlocked()const{
        for(int r=0;r<N;++r)for(int c=0;c<N;++c){int ac=at(r,c).color();
            if(c+1<N){int c2=at(r,c+1).color();if(ac==c2||ac==0||c2==0)return false;}
            if(r+1<N){int c2=at(r+1,c).color();if(ac==c2||ac==0||c2==0)return false;}}
        return true;
    }
};

constexpr int DR[]={-1,1,0,0},DC[]={0,0,-1,1};
int path_score(int k){double t=std::sqrt((double)k)-1.0;return 10*k+18*(int)(t*t);}
int path_score(const Board& b,const std::vector<std::pair<int,int>>& path){
    int k=(int)path.size(),s=path_score(k);
    std::vector<std::vector<bool>> ip(b.N,std::vector<bool>(b.N)),ex(b.N,std::vector<bool>(b.N));
    for(auto [r,c]:path)ip[r][c]=true;
    for(auto [r,c]:path){if(!b.at(r,c).is_bomb())continue;
        for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){int nr=r+dr,nc=c+dc;
            if(b.in_bounds(nr,nc)&&!ip[nr][nc]&&!ex[nr][nc]){ex[nr][nc]=true;s+=10;}}}
    return s;
}
static bool compatible_color(int t,int c){return t==0||c==0||t==c;}
static std::uint64_t board_hash(const Board& b){
    std::uint64_t h=1469598103934665603ULL;const std::uint64_t p=1099511628211ULL;
    h^=(std::uint64_t)b.N;h*=p;h^=(std::uint64_t)b.level;h*=p;
    for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){h^=(std::uint64_t)(b.at(r,c).value+17);h*=p;}
    return h;
}

class ImprovedSolver {
    static constexpr int MAX_TIME_MS=10000;
    mutable std::chrono::steady_clock::time_point _t0;mutable bool _tinit=false;
    void tstart()const{if(!_tinit){_t0=std::chrono::steady_clock::now();_tinit=true;}}
    long long telapsed()const{return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_t0).count();}
    mutable int step_count_=0;
    bool tfast()const{return telapsed()>MAX_TIME_MS*0.90;}
    bool tdead()const{return telapsed()>MAX_TIME_MS*0.97;}
    static int cur_cache_limit_,cur_keep1_,cur_keep2_,cur_keep3_;
    static double mxc_weight_,dq_weight_,bh_weight_,step3_dq_weight_,step3_bh_weight_,surv_mul_;
    static int cur_beam_w_,cur_beam_d_;
    int cache_limit()const{return tfast()?(int)(cur_cache_limit_*0.65):cur_cache_limit_;}
    int keep1()const{return tfast()?(int)(cur_keep1_*0.7):cur_keep1_;}
    int keep2()const{return tfast()?(int)(cur_keep2_*0.7):cur_keep2_;}
    int keep3()const{return tfast()?(int)(cur_keep3_*0.7):cur_keep3_;}
    mutable int board_N_=6;

    static double board_heuristic(const Board& b){
        if(b.is_deadlocked())return -1e5;
        int N=b.N,cp=0,sp=0,wc=0,bb=0,cnt[6]={},mxc=0;
        std::vector<std::vector<bool>> vis(N,std::vector<bool>(N));
        for(int r=0;r<N;++r)for(int c=0;c<N;++c){
            int co=b.at(r,c).color();
            if(b.at(r,c).is_wildcard())wc++;
            if(b.at(r,c).is_bomb())bb++;
            cnt[std::min(co,5)]++;
            if(c+1<N){int c2=b.at(r,c+1).color();if(compatible_color(co,c2)){cp++;if(co==c2&&co!=0)sp++;}}
            if(r+1<N){int c2=b.at(r+1,c).color();if(compatible_color(co,c2)){cp++;if(co==c2&&co!=0)sp++;}}
            if(!vis[r][c]){int sz=0;std::deque<std::pair<int,int>> q;vis[r][c]=true;q.push_back({r,c});
                while(!q.empty()){auto [cr,cc]=q.front();q.pop_front();sz++;
                    int cco=b.at(cr,cc).color();
                    for(int d=0;d<4;++d){int nr=cr+DR[d],nc=cc+DC[d];if(!b.in_bounds(nr,nc)||vis[nr][nc])continue;
                        if(!compatible_color(cco,b.at(nr,nc).color()))continue;vis[nr][nc]=true;q.push_back({nr,nc});}}
                mxc=std::max(mxc,sz);}
        }
        double ent=0.0;int tc=N*N-wc;
        if(tc>0)for(int i=1;i<=5;++i)if(cnt[i]>0){double p=(double)cnt[i]/tc;ent-=p*std::log(p);}
        bool is_pure5=(b.level<=2);bool is_bomb_level=(b.level==4||b.level==5);
        double bb_weight=is_bomb_level?22.0:15.0;
        return cp*6.0+sp*10.0+wc*35.0+bb*bb_weight+mxc*(is_pure5?25.0:15.0)-ent*12.0;
    }
    static double drop_quality(const Board& b){
        double q=0;auto& dq=*b.drop_queue;
        for(int c=0;c<b.N;++c){int qp=b.queue_ptr[c];
            for(int i=0;i<7;++i){if(qp+i>=1000)break;int v=dq[c][qp+i];
                if(v==0){q+=2.8;continue;}if(v<0){q+=0.9;continue;}
                int col=std::abs(v);double adj=0;
                if(b.in_bounds(0,c-1)&&compatible_color(col,b.at(0,c-1).color()))adj+=0.5;
                if(b.in_bounds(0,c+1)&&compatible_color(col,b.at(0,c+1).color()))adj+=0.5;
                if(b.in_bounds(1,c)&&compatible_color(col,b.at(1,c).color()))adj+=0.3;
                q+=adj;if(i<3)for(int j=i+1;j<3&&j<7;++j){int v2=dq[c][qp+j];if(v2==0||v==v2)q+=0.2;}}}
        return q;
    }
    static int upper_bound_reachable(const Board& b,const std::vector<std::pair<int,int>>& path,int target){
        std::vector<std::vector<bool>> blocked(b.N,std::vector<bool>(b.N));
        for(auto [r,c]:path)blocked[r][c]=true;
        auto [tr,tc]=path.back();
        std::deque<std::pair<int,int>> q;std::vector<std::vector<bool>> seen(b.N,std::vector<bool>(b.N));
        for(int d=0;d<4;++d){int nr=tr+DR[d],nc=tc+DC[d];if(!b.in_bounds(nr,nc)||blocked[nr][nc])continue;
            if(!compatible_color(target,b.at(nr,nc).color()))continue;seen[nr][nc]=true;q.push_back({nr,nc});}
        int reachable=0,black=0,white=0;
        for(auto [r,c]:path){if((r+c)%2==0)black++;else white++;}
        while(!q.empty()){auto [r,c]=q.front();q.pop_front();++reachable;
            if((r+c)%2==0)black++;else white++;
            for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];
                if(!b.in_bounds(nr,nc)||blocked[nr][nc]||seen[nr][nc])continue;
                if(!compatible_color(target,b.at(nr,nc).color()))continue;seen[nr][nc]=true;q.push_back({nr,nc});}}
        int ub_reach=(int)path.size()+reachable;
        if(b.level<=2||b.level==4){int ub_bip=std::min(black,white)*2;if(black!=white)ub_bip++;return std::min(ub_reach,ub_bip);}
        return ub_reach;
    }
    static int exact_best_one_step_score(const Board& b,int max_nodes){
        int best=0,nodes=0;std::vector<std::pair<int,int>> path;std::vector<bool> vis(b.N*b.N,false);
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target){
            if(++nodes>max_nodes)return;
            int co=b.at(r,c).color();int fixed=target;if(fixed==0&&co!=0)fixed=co;
            if((int)path.size()>=2){int sc=path_score(b,path);if(sc>best)best=sc;
                int ub=upper_bound_reachable(b,path,fixed);if(path_score(ub)<=best)return;}
            std::pair<int,int> nb[4];int nb_cnt=0;
            for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!b.in_bounds(nr,nc)||vis[nr*b.N+nc])continue;
                if(!compatible_color(fixed,b.at(nr,nc).color()))continue;nb[nb_cnt++]={nr,nc};}
            for(int i=0;i<nb_cnt;++i)for(int j=i+1;j<nb_cnt;++j){
                auto eval=[&](const std::pair<int,int>& p)->int{int pr=p.first,pc=p.second,pot=0;
                    for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc])continue;
                        if(compatible_color(fixed,b.at(ppr,ppc).color()))pot++;}
                    if(b.level<=2){int same=0;
                        for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                            if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc])continue;
                            if(b.at(ppr,ppc).color()==fixed)same++;}
                        if(b.level==1)return pot*10-same;else return pot*10+same;}
                    int base=0;if(b.at(pr,pc).is_wildcard())base=40;
                    else if(b.at(pr,pc).is_bomb())base=(b.level==4?35:25);
                    else if(b.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(b.level<=2){if(eval(nb[j])<eval(nb[i]))std::swap(nb[i],nb[j]);}
                else{if(eval(nb[j])>eval(nb[i]))std::swap(nb[i],nb[j]);}}
            for(int i=0;i<nb_cnt;++i){auto [nr,nc]=nb[i];path.push_back({nr,nc});vis[nr*b.N+nc]=true;dfs(nr,nc,fixed);vis[nr*b.N+nc]=false;path.pop_back();if(nodes>max_nodes)return;}};
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){path.clear();path.push_back({r,c});vis.assign(b.N*b.N,false);vis[r*b.N+c]=true;dfs(r,c,0);if(nodes>max_nodes)break;}
        return best;
    }
    static std::pair<int,std::vector<std::pair<int,int>>> exact_best_with_path(const Board& b,int max_nodes){
        int best_score=0,nodes=0;std::vector<std::pair<int,int>> best_path,path;std::vector<bool> vis(b.N*b.N,false);
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target){
            if(++nodes>max_nodes)return;
            int co=b.at(r,c).color();int fixed=target;if(fixed==0&&co!=0)fixed=co;
            if((int)path.size()>=2){int sc=path_score(b,path);
                if(sc>best_score||(sc==best_score&&path<best_path)){best_score=sc;best_path=path;}
                int ub=upper_bound_reachable(b,path,fixed);if(path_score(ub)<=best_score)return;}
            std::pair<int,int> nb[4];int nb_cnt=0;
            for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!b.in_bounds(nr,nc)||vis[nr*b.N+nc])continue;
                if(!compatible_color(fixed,b.at(nr,nc).color()))continue;nb[nb_cnt++]={nr,nc};}
            for(int i=0;i<nb_cnt;++i)for(int j=i+1;j<nb_cnt;++j){
                auto eval=[&](const std::pair<int,int>& p)->int{int pr=p.first,pc=p.second,pot=0;
                    for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                        if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc])continue;
                        if(compatible_color(fixed,b.at(ppr,ppc).color()))pot++;}
                    if(b.level<=2){int same=0;
                        for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                            if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc])continue;
                            if(b.at(ppr,ppc).color()==fixed)same++;}
                        if(b.level==1)return pot*10-same;else return pot*10+same;}
                    int base=0;if(b.at(pr,pc).is_wildcard())base=40;
                    else if(b.at(pr,pc).is_bomb())base=(b.level==4?35:25);
                    else if(b.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(b.level<=2){if(eval(nb[j])<eval(nb[i]))std::swap(nb[i],nb[j]);}
                else{if(eval(nb[j])>eval(nb[i]))std::swap(nb[i],nb[j]);}}
            for(int i=0;i<nb_cnt;++i){auto [nr,nc]=nb[i];path.push_back({nr,nc});vis[nr*b.N+nc]=true;dfs(nr,nc,fixed);vis[nr*b.N+nc]=false;path.pop_back();if(nodes>max_nodes)return;}};
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){path.clear();path.push_back({r,c});vis.assign(b.N*b.N,false);vis[r*b.N+c]=true;dfs(r,c,0);if(nodes>max_nodes)break;}
        return {best_score,best_path};
    }
    int survival_steps(const Board& b,int max_check)const{
        Board sim=b;int i;
        for(i=0;i<max_check;++i){if(sim.is_deadlocked())break;
            auto [sc,bp]=exact_best_with_path(sim,tdead()?5000:20000);
            if(bp.size()<2){i++;break;}sim=sim.preview(bp);}
        return i;
    }
    double beam_evaluate(const Board& start_b,int beam_w,int beam_d)const{
        struct BNode{Board bd;double acc;};std::vector<BNode> beams;beams.push_back({start_b,0.0});
        for(int d=0;d<beam_d;++d){struct Cand{Board bd;double acc,val;};std::vector<Cand> cands;
            for(auto& bm:beams){auto [sc,bp]=exact_best_with_path(bm.bd,d==beam_d-1?10000:5000);
                if(bp.size()<2)continue;Board nb=bm.bd.preview(bp);double hv=board_heuristic(nb);
                double total=bm.acc+(double)sc+hv*(d==beam_d-1?0.5:0.15);cands.push_back({std::move(nb),bm.acc+(double)sc,total});
                for(int tries=0;tries<4&&(int)cands.size()<beam_w*4;++tries){auto [sc2,bp2]=exact_best_with_path(bm.bd,2000+tries*1000);
                    if(bp2.size()<2||(bp2[0]==bp[0]&&bp2.size()==bp.size()))continue;Board nb2=bm.bd.preview(bp2);
                    double hv2=board_heuristic(nb2);double total2=bm.acc+(double)sc2+hv2*(d==beam_d-1?0.5:0.15);cands.push_back({std::move(nb2),bm.acc+(double)sc2,total2});}}
            if(cands.empty())break;std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.val>b.val;});beams.clear();
            for(int i=0;i<std::min(beam_w,(int)cands.size());++i)beams.push_back({std::move(cands[i].bd),cands[i].acc});}
        return beams.empty()?0.0:beams[0].acc;
    }
    static std::vector<std::pair<int,int>> fallback(const Board& b){
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){
            if(b.at(r,c).is_wildcard())for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(b.in_bounds(nr,nc))return{{r,c},{nr,nc}};}
            int a=b.at(r,c).color();
            if(c+1<b.N&&compatible_color(a,b.at(r,c+1).color()))return{{r,c},{r,c+1}};
            if(r+1<b.N&&compatible_color(a,b.at(r+1,c).color()))return{{r,c},{r+1,c}};}
        return{{0,0},{0,1}};
    }
public:
    std::vector<std::pair<int,int>> solve(const Board& board){
        _tinit=false;tstart();++step_count_;board_N_=board.N;
        bool pure5=(board.level<=2);
        int lmt=(board_N_<=5?2000000:(pure5?3000000:(board.level==5?2000000:(board.level==3?1800000:1500000))));
        int cmax=cache_limit();

        switch(board.level){
            case 1:cur_cache_limit_=80000;cur_keep1_=200;cur_keep2_=100;cur_keep3_=20;mxc_weight_=25.0;dq_weight_=0.8;bh_weight_=0.02;step3_dq_weight_=0.55;step3_bh_weight_=0.025;surv_mul_=0.5;cur_beam_w_=8;cur_beam_d_=4;break;
            case 2:cur_cache_limit_=80000;cur_keep1_=200;cur_keep2_=100;cur_keep3_=20;mxc_weight_=25.0;dq_weight_=0.8;bh_weight_=0.02;step3_dq_weight_=0.55;step3_bh_weight_=0.025;surv_mul_=1.0;cur_beam_w_=8;cur_beam_d_=4;break;
            case 3:cur_cache_limit_=60000;cur_keep1_=120;cur_keep2_=60;cur_keep3_=12;mxc_weight_=18.0;dq_weight_=0.9;bh_weight_=0.02;step3_dq_weight_=0.6;step3_bh_weight_=0.025;surv_mul_=1.0;cur_beam_w_=0;cur_beam_d_=0;break;
            case 4:cur_cache_limit_=80000;cur_keep1_=120;cur_keep2_=60;cur_keep3_=12;mxc_weight_=15.0;dq_weight_=0.8;bh_weight_=0.02;step3_dq_weight_=0.55;step3_bh_weight_=0.025;surv_mul_=1.0;cur_beam_w_=5;cur_beam_d_=3;break;
            case 5:cur_cache_limit_=50000;cur_keep1_=60;cur_keep2_=30;cur_keep3_=6;mxc_weight_=15.0;dq_weight_=0.9;bh_weight_=0.02;step3_dq_weight_=0.6;step3_bh_weight_=0.025;surv_mul_=1.0;cur_beam_w_=0;cur_beam_d_=0;break;
            default:cur_cache_limit_=60000;cur_keep1_=200;cur_keep2_=100;cur_keep3_=20;mxc_weight_=15.0;dq_weight_=0.8;bh_weight_=0.02;step3_dq_weight_=0.55;step3_bh_weight_=0.025;surv_mul_=1.0;cur_beam_w_=8;cur_beam_d_=4;break;
        }

        struct Raw{std::vector<std::pair<int,int>> path;int now;int nxt;};
        std::vector<Raw> raws;raws.reserve(8192);
        int nodes=0;
        std::vector<std::pair<int,int>> path;
        std::vector<bool> vis_main(board.N*board.N,false);

        // ==== 第一阶段：DFS 收集路径，暂不计算 nxt ====
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target){
            if(++nodes>lmt)return;
            int cur=board.at(r,c).color();int fixed=target;if(fixed==0&&cur!=0)fixed=cur;
            if((int)path.size()>=2){
                int ns=path_score(board,path);
                raws.push_back({path,ns,-1}); // nxt 待定
            }
            std::pair<int,int> nb_arr[4];int nb_cnt=0;
            for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!board.in_bounds(nr,nc)||vis_main[nr*board.N+nc])continue;
                if(!compatible_color(fixed,board.at(nr,nc).color()))continue;nb_arr[nb_cnt++]={nr,nc};}
            for(int i=0;i<nb_cnt;++i)for(int j=i+1;j<nb_cnt;++j){
                auto eval=[&](const std::pair<int,int>& p)->int{int pr=p.first,pc=p.second,pot=0;
                    for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                        if(!board.in_bounds(ppr,ppc)||vis_main[ppr*board.N+ppc])continue;
                        if(compatible_color(fixed,board.at(ppr,ppc).color()))pot++;}
                    if(pure5){int same=0;
                        for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                            if(!board.in_bounds(ppr,ppc)||vis_main[ppr*board.N+ppc])continue;
                            if(board.at(ppr,ppc).color()==fixed)same++;}
                        if(board.level==1)return pot*10-same;else return pot*10+same;}
                    int base=0;if(board.at(pr,pc).is_wildcard())base=40;
                    else if(board.at(pr,pc).is_bomb())base=(board.level==4?35:25);
                    else if(board.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(pure5){if(eval(nb_arr[j])<eval(nb_arr[i]))std::swap(nb_arr[i],nb_arr[j]);}
                else{if(eval(nb_arr[j])>eval(nb_arr[i]))std::swap(nb_arr[i],nb_arr[j]);}}
            for(int i=0;i<nb_cnt;++i){auto [nr,nc]=nb_arr[i];path.push_back({nr,nc});vis_main[nr*board.N+nc]=true;dfs(nr,nc,fixed);vis_main[nr*board.N+nc]=false;path.pop_back();if(nodes>lmt)return;}};

        std::vector<std::pair<int,int>> starts;
        if(pure5){
            std::vector<std::vector<int>> cc_size(board.N,std::vector<int>(board.N,0));
            std::vector<std::vector<bool>> vis(board.N,std::vector<bool>(board.N,false));
            for(int r=0;r<board.N;++r)for(int c=0;c<board.N;++c){if(vis[r][c])continue;
                int co=board.at(r,c).color();std::vector<std::pair<int,int>> cells;std::deque<std::pair<int,int>> q;
                q.push_back({r,c});vis[r][c]=true;
                while(!q.empty()){auto [cr,cc]=q.front();q.pop_front();cells.push_back({cr,cc});
                    for(int d=0;d<4;++d){int nr=cr+DR[d],nc=cc+DC[d];
                        if(!board.in_bounds(nr,nc)||vis[nr][nc])continue;
                        if(board.at(nr,nc).color()!=co)continue;vis[nr][nc]=true;q.push_back({nr,nc});}}
                for(auto& p:cells)cc_size[p.first][p.second]=(int)cells.size();}
            for(int r=0;r<board.N;++r)for(int c=0;c<board.N;++c)starts.push_back({r,c});
            if(board.level==1)std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){return cc_size[a.first][a.second]>cc_size[b.first][b.second];});
            else std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){return cc_size[a.first][a.second]<cc_size[b.first][b.second];});
        }else{for(int r=0;r<board.N;++r)for(int c=0;c<board.N;++c)starts.push_back({r,c});}

        for(auto [r,c]:starts){path.clear();path.push_back({r,c});vis_main.assign(board.N*board.N,false);vis_main[r*board.N+c]=true;dfs(r,c,0);if(nodes>lmt)break;}
        if(raws.empty()){for(int r=0;r<board.N;++r)for(int c=0;c<board.N;++c){path.clear();path.push_back({r,c});vis_main.assign(board.N*board.N,false);vis_main[r*board.N+c]=true;dfs(r,c,0);if(nodes>lmt)break;}}
        if(raws.empty())return fallback(board);

        // ==== 第二阶段：多线程并行计算 nxt ====
        int nthreads=std::max(1u,std::min(8u,std::thread::hardware_concurrency()));
        std::atomic<int> idx(0);
        std::vector<std::thread> threads;
        for(int t=0;t<nthreads;++t){
            threads.emplace_back([&](){
                std::unordered_map<std::uint64_t,int> local_cache;
                while(true){
                    int i=idx.fetch_add(1);
                    if(i>=(int)raws.size())break;
                    Board nb=board.preview(raws[i].path);
                    auto h=board_hash(nb);
                    auto it=local_cache.find(h);
                    if(it!=local_cache.end()){raws[i].nxt=it->second;continue;}
                    int nxt=exact_best_one_step_score(nb,cmax);
                    local_cache[h]=nxt;
                    raws[i].nxt=nxt;
                }
            });
        }
        for(auto& t:threads)t.join();

        // ==== 第三阶段：排序与评估（原有逻辑） ====
        std::vector<int> order(raws.size());for(int i=0;i<(int)raws.size();++i)order[i]=i;
        std::sort(order.begin(),order.end(),[&](int a,int b){
            int va=raws[a].now+raws[a].nxt,vb=raws[b].now+raws[b].nxt;
            if(pure5){int pa=(int)raws[a].path.size(),pb=(int)raws[b].path.size();
                if(board.level==1){if(pa<6)va-=55;if(pb<6)vb-=55;}
                else{if(pa<6)va-=25;if(pb<6)vb-=25;}}
            return va!=vb?va>vb:raws[a].now>raws[b].now;});
        int k1=std::min(keep1(),(int)raws.size());order.resize(k1);

        struct Cand{std::vector<std::pair<int,int>> path;int now,nxt;Board next_b;double v2;};
        std::vector<Cand> cands;cands.reserve(k1);
        for(int idx:order){Board nb=board.preview(raws[idx].path);
            double dq=drop_quality(nb),bh=board_heuristic(nb);
            double v2=(double)raws[idx].now+(double)raws[idx].nxt+dq*dq_weight_+bh*bh_weight_;
            cands.push_back({raws[idx].path,raws[idx].now,raws[idx].nxt,std::move(nb),v2});}
        std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.v2>b.v2;});
        if(tdead())return cands[0].path;

        static std::unordered_map<std::uint64_t,int> cache3;
        int k2=std::min(keep2(),(int)cands.size());
        struct Scored{double val;int idx;};
        std::vector<Scored> scored;scored.reserve(k1);
        for(int i=0;i<k1;++i)scored.push_back({cands[i].v2,i});

        // ==== 并行 Step 2：多线程三步前瞻 ====
        if(!tdead()){
            int nt=std::max(1,(int)std::min(8u,std::thread::hardware_concurrency()));
            std::mutex mx;std::atomic<int> ai(0);std::vector<std::thread> th;
            for(int t=0;t<nt;++t)th.emplace_back([&]{while(true){
                int i=ai.fetch_add(1);if(i>=k2)break;
                auto [nsc,np]=exact_best_with_path(cands[i].next_b,cmax);if(np.size()<2)continue;
                Board tb=cands[i].next_b.preview(np);auto h=board_hash(tb);
                int tsc=0;{std::lock_guard<std::mutex> lk(mx);
                    auto it=cache3.find(h);if(it!=cache3.end())tsc=it->second;
                    else{tsc=exact_best_one_step_score(tb,cmax/3);cache3[h]=tsc;}}
                scored[i].val=(double)cands[i].now+(double)nsc+(double)tsc+drop_quality(tb)*step3_dq_weight_+board_heuristic(tb)*step3_bh_weight_;
            }});
            for(auto& t:th)t.join();
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        if(tdead())return cands[scored[0].idx].path;

        int k3=std::min(keep3(),(int)scored.size());

        // ==== 并行 Step 3：多线程生存分析 ====
        {
            int nt=std::max(1,(int)std::min(8u,std::thread::hardware_concurrency()));
            std::atomic<int> ai(0);std::vector<std::thread> th;
            for(int t=0;t<nt;++t)th.emplace_back([&]{while(true){
                int j=ai.fetch_add(1);if(j>=k3)break;
                int st=survival_steps(cands[scored[j].idx].next_b,6);
                if(st<=2)scored[j].val-=120.0*surv_mul_*(3.0-(double)st);
                else if(st<=3)scored[j].val-=50.0*surv_mul_;else if(st<=4)scored[j].val-=15.0*surv_mul_;
            }});
            for(auto& t:th)t.join();
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});

        if(!tfast()&&cur_beam_w_>0){int bs_cnt=std::min(3,(int)scored.size());
            for(int j=0;j<bs_cnt;++j){double beam_val=beam_evaluate(cands[scored[j].idx].next_b,cur_beam_w_,cur_beam_d_);
                scored[j].val+=beam_val*0.3;}
            std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});}

        int sel=0;
        if(cache3.size()>300000)cache3.clear();
        return cands[scored[sel].idx].path;
    }
};

int ImprovedSolver::cur_cache_limit_=60000,ImprovedSolver::cur_keep1_=200,ImprovedSolver::cur_keep2_=100,ImprovedSolver::cur_keep3_=20;
double ImprovedSolver::mxc_weight_=15.0,ImprovedSolver::dq_weight_=0.8,ImprovedSolver::bh_weight_=0.02;
double ImprovedSolver::step3_dq_weight_=0.55,ImprovedSolver::step3_bh_weight_=0.025,ImprovedSolver::surv_mul_=1.0;
int ImprovedSolver::cur_beam_w_=8,ImprovedSolver::cur_beam_d_=4;

class LocalJudge{
    int level_,N_,step_=0,score_=0,invalid_streak_=0;Board board_;bool done_=false;static constexpr int MAX_STEPS=50;
    static int gen_block(std::mt19937& rng,int level){
        if(level<=2)return(rng()%5)+1;if(level==3)return((rng()%100)<15)?0:(rng()%5)+1;
        if(level==4){int c=(rng()%5)+1;return((rng()%100)<10)?-c:c;}
        if((rng()%100)<15)return 0;int base=(rng()%5)+1;return((rng()%100)<10)?-base:base;}
public:
    LocalJudge(int level,int seed,int N):level_(level),N_(N){board_=Board(N);board_.level=level;
        board_.drop_queue=std::make_shared<std::vector<std::vector<int>>>();board_.drop_queue->assign(N,std::vector<int>(1000));board_.queue_ptr.assign(N,0);
        std::mt19937 rng(seed);for(int c=0;c<N;++c)for(int i=0;i<1000;++i)(*board_.drop_queue)[c][i]=gen_block(rng,level);
        std::mt19937 rng_board(seed^0x9E3779B9);for(int r=0;r<N;++r)for(int c=0;c<N;++c)board_.at(r,c).value=gen_block(rng_board,level);}
    const Board& board()const{return board_;}int score()const{return score_;}int step()const{return step_;}bool done()const{return done_;}
    bool play(const std::vector<std::pair<int,int>>& path){
        if(done_||step_>=MAX_STEPS){done_=true;return false;}
        if(path.size()<2){invalid_streak_++;step_++;if(invalid_streak_>=3)done_=true;return!done_;}
        invalid_streak_=0;int gained=path_score(path.size());
        if(level_>=4){std::vector<std::vector<bool>> ip(N_,std::vector<bool>(N_)),ex(N_,std::vector<bool>(N_));
            for(auto [r,c]:path)ip[r][c]=true;
            for(auto [r,c]:path){if(!board_.at(r,c).is_bomb())continue;
                for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){int nr=r+dr,nc=c+dc;
                    if(nr>=0&&nr<N_&&nc>=0&&nc<N_&&!ip[nr][nc]&&!ex[nr][nc]){ex[nr][nc]=true;gained+=10;}}}}
        score_+=gained;step_++;
        std::vector<std::vector<bool>> to_remove(N_,std::vector<bool>(N_));for(auto [r,c]:path)to_remove[r][c]=true;
        if(level_>=4)for(auto [r,c]:path){if(!board_.at(r,c).is_bomb())continue;
            for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){int nr=r+dr,nc=c+dc;
                if(nr>=0&&nr<N_&&nc>=0&&nc<N_&&!to_remove[nr][nc])to_remove[nr][nc]=true;}}
        for(int c=0;c<N_;++c){std::vector<Cell> rem;for(int r=0;r<N_;++r)if(!to_remove[r][c])rem.push_back(board_.at(r,c));
            int empty=N_-(int)rem.size();for(int i=0;i<empty;++i)board_.at(i,c).value=(*board_.drop_queue)[c][board_.queue_ptr[c]++];
            for(int i=0;i<(int)rem.size();++i)board_.at(empty+i,c)=rem[i];}
        if(board_.is_deadlocked())done_=true;if(step_>=MAX_STEPS)done_=true;return!done_;}
};

int main(int argc,char** argv){
    int seed=argc>1?std::stoi(argv[1]):114514;
    ImprovedSolver solver;int total=0;
    std::vector<std::tuple<int,int,int,const char*>> levels={{1,seed,10,"L1"},{2,seed,10,"L2"},{3,seed,10,"L3"},{4,seed,10,"L4"},{5,seed,12,"L5"}};
    for(auto& [lv,s,N,name]:levels){
        LocalJudge judge(lv,s,N);
        while(!judge.done()){auto path=solver.solve(judge.board());if(path.size()<2)path={{0,0},{0,1}};if(!judge.play(path))break;}
        std::cerr<<name<<":"<<judge.score()<<" ";total+=judge.score();
    }
    std::cout<<total<<"\n";
    return 0;
}
