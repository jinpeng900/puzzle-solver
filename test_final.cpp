/**
 * test_final.cpp — 测试 2_final.cpp 求解器（独立编译，自包含）
 * 编译: g++ -std=c++14 -O2 -o test_final.exe test_final.cpp
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
#include <fstream>
#include <cstdint>

// ============================
// 从 2_final.cpp 复制核心代码
// ============================
struct Cell { int value = 1; int color() const { return value>0?value:(-value); } bool is_bomb() const { return value<0; } bool is_wildcard() const { return value==0; } };

struct Board {
    int N=0,level=1; std::vector<std::vector<Cell>> grid;
    std::shared_ptr<std::vector<std::vector<int>>> drop_queue;
    std::vector<int> queue_ptr;
    Board(int n=0):N(n),grid(n,std::vector<Cell>(n)),drop_queue(std::make_shared<std::vector<std::vector<int>>>()){}
    Cell& at(int r,int c){return grid[r][c];} const Cell& at(int r,int c)const{return grid[r][c];}
    bool in_bounds(int r,int c)const{return r>=0&&r<N&&c>=0&&c<N;}
    Board preview(const std::vector<std::pair<int,int>>& path)const; // 定义在后
    bool is_deadlocked()const;
};

constexpr int DR[]={-1,1,0,0}, DC[]={0,0,-1,1};

Board Board::preview(const std::vector<std::pair<int,int>>& path)const{
    Board nb=*this; if(path.size()<2)return nb; auto& dq=*nb.drop_queue;
    std::vector<std::vector<bool>> in(N,std::vector<bool>(N));
    for(auto& p:path)in[p.first][p.second]=true; auto rm=in;
    if(level>=4)for(auto& p:path){int r=p.first,c=p.second;if(!at(r,c).is_bomb())continue;
        for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){int nr=r+dr,nc=c+dc;if(in_bounds(nr,nc)&&!in[nr][nc])rm[nr][nc]=true;}}
    for(int c=0;c<N;++c){std::vector<Cell> rem;for(int r=0;r<N;++r)if(!rm[r][c])rem.push_back(at(r,c));
        int emp=N-(int)rem.size();for(int i=0;i<emp;++i)nb.at(i,c).value=dq[c][nb.queue_ptr[c]++];
        for(int i=0;i<(int)rem.size();++i)nb.at(emp+i,c)=rem[i];}
    return nb;
}
bool Board::is_deadlocked()const{for(int r=0;r<N;++r)for(int c=0;c<N;++c){int ac=at(r,c).color();
    if(c+1<N){int c2=at(r,c+1).color();if(ac==c2||ac==0||c2==0)return false;}
    if(r+1<N){int c2=at(r+1,c).color();if(ac==c2||ac==0||c2==0)return false;}}return true;}

int path_score(int k){double t=std::sqrt((double)k)-1.0;return 10*k+18*(int)(t*t);}
int path_score(const Board& b,const std::vector<std::pair<int,int>>& path){
    int k=(int)path.size(),s=path_score(k);
    std::vector<std::vector<bool>> in(b.N,std::vector<bool>(b.N)),expl(b.N,std::vector<bool>(b.N));
    for(auto& p:path)in[p.first][p.second]=true;
    for(auto& p:path){int r=p.first,c=p.second;if(!b.at(r,c).is_bomb())continue;
        for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){int nr=r+dr,nc=c+dc;
            if(b.in_bounds(nr,nc)&&!in[nr][nc]&&!expl[nr][nc]){expl[nr][nc]=true;s+=10;}}}
    return s;
}
static bool comp(int a,int b){return a==0||b==0||a==b;}
static uint64_t bit(int r,int c,int N){return 1ULL<<(r*N+c);}
static uint64_t bhash(const Board& b){uint64_t h=1469598103934665603ULL,pr=1099511628211ULL;
    h^=(uint64_t)b.N;h*=pr;h^=(uint64_t)b.level;h*=pr;
    for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){h^=(uint64_t)(b.at(r,c).value+17);h*=pr;}return h;}

// NN 模块（精简版，保持接口兼容）
class NNEval { bool en=false; public:
    void enable(bool f){en=f;} bool ok()const{return en;}
    double eval(const Board& b);
    bool load_w(const std::string&){return false;} // stub
    static double heuristic(const Board& b);
};
static NNEval g_nn;

double NNEval::heuristic(const Board& b){if(b.is_deadlocked())return-1e5;
    int N=b.N,cp=0,sp=0,wc=0,bb=0,cnt[6]={},mxc=0;
    std::vector<std::vector<bool>> vis(N,std::vector<bool>(N));
    for(int r=0;r<N;++r)for(int c=0;c<N;++c){int co=b.at(r,c).color();if(b.at(r,c).is_wildcard())wc++;if(b.at(r,c).is_bomb())bb++;cnt[std::min(co,5)]++;
        if(c+1<N){int c2=b.at(r,c+1).color();if(comp(co,c2)){cp++;if(co==c2&&co!=0)sp++;}}
        if(r+1<N){int c2=b.at(r+1,c).color();if(comp(co,c2)){cp++;if(co==c2&&co!=0)sp++;}}
        if(!vis[r][c]){int sz=0;std::deque<std::pair<int,int>> q;vis[r][c]=true;q.push_back({r,c});
            while(!q.empty()){auto pp=q.front();q.pop_front();sz++;int cr=pp.first,ct=pp.second;int cco=b.at(cr,ct).color();
                for(int d=0;d<4;++d){int nr=cr+DR[d],nc=ct+DC[d];if(!b.in_bounds(nr,nc)||vis[nr][nc])continue;
                    if(!comp(cco,b.at(nr,nc).color()))continue;vis[nr][nc]=true;q.push_back({nr,nc});}}mxc=std::max(mxc,sz);}}
    double ent=0;int tc=N*N-wc;if(tc>0)for(int i=1;i<=5;++i)if(cnt[i]>0){double p=(double)cnt[i]/tc;ent-=p*std::log(p);}
    double bbw=(b.level==4||b.level==5)?22.0:15.0;bool pp=(b.level<=2);return cp*6.0+sp*10.0+wc*35.0+bb*bbw+mxc*(pp?25.0:15.0)-ent*12.0;}
double NNEval::eval(const Board& b){return heuristic(b);}

// ===== ImprovedSolver（完整复制 2_final.cpp 版本）=====
class ImprovedSolver {
    static constexpr int MAX_T=10000; mutable std::chrono::steady_clock::time_point _t0; mutable bool _ti=false;
    void ts()const{if(!_ti){_t0=std::chrono::steady_clock::now();_ti=true;}}
    long long te()const{return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_t0).count();}
    mutable int sc_=0; bool tf()const{return te()>MAX_T*0.90;} bool td()const{return te()>MAX_T*0.97;}
    static int ccl,ck1,ck2,ck3; static double mxw,dqw,bhw,s3dq,s3bh,svm; static int cbw,cbd;
    int cl()const{return tf()?(int)(ccl*0.65):ccl;} int k1()const{return tf()?(int)(ck1*0.7):ck1;}
    int k2()const{return tf()?(int)(ck2*0.7):ck2;} int k3()const{return tf()?(int)(ck3*0.7):ck3;}
    mutable int bN=6;
    // adaptive tracking
    mutable double lpdq=0,lpbh=0; mutable int lpnxt=0,lpnow=0,lplev=-1; mutable bool pf=false;
    mutable double lps3=0;
    void af(const Board& b); void sp(const Board& nb,int now,int nxt,int level,double s3=0);
public:
    static double bh(const Board& b){return g_nn.eval(b);}
    static double dropq(const Board& b){double q=0;auto& dq=*b.drop_queue;
        for(int c=0;c<b.N;++c){int qp=b.queue_ptr[c];for(int i=0;i<7;++i){if(qp+i>=1000)break;int v=dq[c][qp+i];
            if(v==0){q+=2.8;continue;}if(v<0){q+=0.9;continue;}int col=std::abs(v);double adj=0;
            if(b.in_bounds(0,c-1)&&comp(col,b.at(0,c-1).color()))adj+=0.5;
            if(b.in_bounds(0,c+1)&&comp(col,b.at(0,c+1).color()))adj+=0.5;
            if(b.in_bounds(1,c)&&comp(col,b.at(1,c).color()))adj+=0.3;q+=adj;
            if(i<3)for(int j=i+1;j<3&&j<7;++j){int v2=dq[c][qp+j];if(v2==0||v==v2)q+=0.2;}}}return q;}
    static int ub(const Board& b,const std::vector<std::pair<int,int>>& path,int target){
        std::vector<std::vector<bool>> blocked(b.N,std::vector<bool>(b.N));
        for(auto& p:path)blocked[p.first][p.second]=true;int tr=path.back().first,tc=path.back().second;
        std::deque<std::pair<int,int>> q;std::vector<std::vector<bool>> seen(b.N,std::vector<bool>(b.N));
        for(int d=0;d<4;++d){int nr=tr+DR[d],nc=tc+DC[d];if(!b.in_bounds(nr,nc)||blocked[nr][nc])continue;
            if(!comp(target,b.at(nr,nc).color()))continue;seen[nr][nc]=true;q.push_back({nr,nc});}
        int reach=0,black=0,white=0;for(auto& p:path){if((p.first+p.second)%2==0)black++;else white++;}
        while(!q.empty()){auto pp=q.front();q.pop_front();++reach;int r=pp.first,c=pp.second;
            if((r+c)%2==0)black++;else white++;for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];
                if(!b.in_bounds(nr,nc)||blocked[nr][nc]||seen[nr][nc])continue;
                if(!comp(target,b.at(nr,nc).color()))continue;seen[nr][nc]=true;q.push_back({nr,nc});}}
        int ub2=(int)path.size()+reach;if(b.level<=2||b.level==4){int ub3=std::min(black,white)*2;if(black!=white)ub3++;return std::min(ub2,ub3);}return ub2;}
    static int best1(const Board& b,int maxn){int best=0,nodes=0;std::vector<std::pair<int,int>> path;
        std::function<void(uint64_t,int,int,int)> dfs=[&](uint64_t mask,int r,int c,int target){if(++nodes>maxn)return;
            int co=b.at(r,c).color(),fixed=target;if(fixed==0&&co!=0)fixed=co;
            if(path.size()>=2){int sc=path_score(b,path);if(sc>best)best=sc;int u=ub(b,path,fixed);if(path_score(u)<=best)return;}
            std::pair<int,int> nb[4];int nbc=0;for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];
                if(!b.in_bounds(nr,nc)||(mask&bit(nr,nc,b.N)))continue;if(!comp(fixed,b.at(nr,nc).color()))continue;nb[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){auto ev=[&](auto& p)->int{int pr=p.first,pc=p.second,pot=0;
                for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N)))continue;if(comp(fixed,b.at(ppr,ppc).color()))pot++;}
                if(b.level<=2){int sm=0;for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N)))continue;
                    if(b.at(ppr,ppc).color()==fixed)sm++;}return(b.level==1)?pot*10-sm:pot*10+sm;}
                int base=0;if(b.at(pr,pc).is_wildcard())base=40;else if(b.at(pr,pc).is_bomb())base=(b.level==4?35:25);else if(b.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(b.level<=2){if(ev(nb[j])<ev(nb[i]))std::swap(nb[i],nb[j]);}else{if(ev(nb[j])>ev(nb[i]))std::swap(nb[i],nb[j]);}}
            for(int i=0;i<nbc;++i){path.push_back(nb[i]);dfs(mask|bit(nb[i].first,nb[i].second,b.N),nb[i].first,nb[i].second,fixed);path.pop_back();if(nodes>maxn)return;}};
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){path.clear();path.push_back({r,c});dfs(bit(r,c,b.N),r,c,0);if(nodes>maxn)break;}return best;}
    static std::pair<int,std::vector<std::pair<int,int>>> bpw(const Board& b,int maxn){
        int bs=0,nodes=0;std::vector<std::pair<int,int>> bp,path;
        std::function<void(uint64_t,int,int,int)> dfs=[&](uint64_t mask,int r,int c,int target){if(++nodes>maxn)return;
            int co=b.at(r,c).color(),fixed=target;if(fixed==0&&co!=0)fixed=co;
            if(path.size()>=2){int sc=path_score(b,path);if(sc>bs||(sc==bs&&path<bp)){bs=sc;bp=path;}int u=ub(b,path,fixed);if(path_score(u)<=bs)return;}
            std::pair<int,int> nb[4];int nbc=0;for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!b.in_bounds(nr,nc)||(mask&bit(nr,nc,b.N)))continue;
                if(!comp(fixed,b.at(nr,nc).color()))continue;nb[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){auto ev=[&](auto& p)->int{int pr=p.first,pc=p.second,pot=0;
                for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N)))continue;if(comp(fixed,b.at(ppr,ppc).color()))pot++;}
                if(b.level<=2){int sm=0;for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!b.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,b.N)))continue;
                    if(b.at(ppr,ppc).color()==fixed)sm++;}return(b.level==1)?pot*10-sm:pot*10+sm;}
                int base=0;if(b.at(pr,pc).is_wildcard())base=40;else if(b.at(pr,pc).is_bomb())base=(b.level==4?35:25);else if(b.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(b.level<=2){if(ev(nb[j])<ev(nb[i]))std::swap(nb[i],nb[j]);}else{if(ev(nb[j])>ev(nb[i]))std::swap(nb[i],nb[j]);}}
            for(int i=0;i<nbc;++i){path.push_back(nb[i]);dfs(mask|bit(nb[i].first,nb[i].second,b.N),nb[i].first,nb[i].second,fixed);path.pop_back();if(nodes>maxn)return;}};
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){path.clear();path.push_back({r,c});dfs(bit(r,c,b.N),r,c,0);if(nodes>maxn)break;}return {bs,bp};}
    int surv_steps(const Board& b,int maxc)const{Board sim=b;int i;
        for(i=0;i<maxc;++i){if(sim.is_deadlocked())break;auto sbp=bpw(sim,td()?5000:20000);
            if(sbp.second.size()<2){i++;break;}sim=sim.preview(sbp.second);}return i;}
    int surv_score(const Board& b,int maxc)const{Board sim=b;int i,asc=0;
        for(i=0;i<maxc;++i){if(sim.is_deadlocked())break;auto sbp=bpw(sim,td()?5000:20000);
            int sc=sbp.first;if(sbp.second.size()<2){i++;break;}asc+=sc;sim=sim.preview(sbp.second);}return asc;}
    double beam_e(const Board& sb,int bw,int bd)const{struct BN{Board bd;double acc;};std::vector<BN> beams;beams.push_back({sb,0.0});
        for(int d=0;d<bd;++d){struct CN{Board bd;double acc,val;};std::vector<CN> cands;
            for(auto& bm:beams){auto sbp=bpw(bm.bd,d==bd-1?10000:5000);int sc=sbp.first;auto& bp2=sbp.second;
                if(bp2.size()<2)continue;Board nb=bm.bd.preview(bp2);double hv=bh(nb);
                double total=bm.acc+(double)sc+hv*(d==bd-1?0.5:0.15);cands.push_back({std::move(nb),bm.acc+(double)sc,total});}
            if(cands.empty())break;std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.val>b.val;});
            beams.clear();for(int i=0;i<std::min(bw,(int)cands.size());++i)beams.push_back({std::move(cands[i].bd),cands[i].acc});}
        return beams.empty()?0.0:beams[0].acc;}
    static std::vector<std::pair<int,int>> fb(const Board& b){for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){
        if(b.at(r,c).is_wildcard())for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(b.in_bounds(nr,nc))return{{r,c},{nr,nc}};}
        int a=b.at(r,c).color();if(c+1<b.N&&comp(a,b.at(r,c+1).color()))return{{r,c},{r,c+1}};
        if(r+1<b.N&&comp(a,b.at(r+1,c).color()))return{{r,c},{r+1,c}};}return{{0,0},{0,1}};}
public:
    static void reset_adaptive(int lvl){switch(lvl){case 1:mxw=25.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=0.5;break;
        case 2:mxw=25.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=1.0;break;case 3:mxw=18.0;dqw=0.9;bhw=0.02;s3dq=0.6;s3bh=0.025;svm=1.0;break;
        case 4:mxw=15.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=1.0;break;case 5:mxw=15.0;dqw=0.9;bhw=0.02;s3dq=0.6;s3bh=0.025;svm=1.0;break;}}
    std::vector<std::pair<int,int>> solve(const Board& board){
        _ti=false;ts();++sc_;
        if(pf){if(board.level==lplev)af(board);else pf=false;}
        bN=board.N;bool pure5=(board.level<=2);int lmt=(bN<=5?800000:(pure5?1200000:(board.level==5?1200000:(board.level==3?700000:600000))));
        int cmax=cl();
        switch(board.level){case 1:ccl=80000;ck1=200;ck2=100;ck3=20;cbw=8;cbd=4;break;
            case 2:ccl=80000;ck1=200;ck2=100;ck3=20;cbw=8;cbd=4;break;
            case 3:ccl=60000;ck1=120;ck2=60;ck3=12;cbw=2;cbd=2;break;
            case 4:ccl=80000;ck1=120;ck2=60;ck3=12;cbw=5;cbd=3;break;
            case 5:ccl=50000;ck1=60;ck2=30;ck3=6;cbw=2;cbd=2;break;
            default:ccl=60000;ck1=200;ck2=100;ck3=20;cbw=8;cbd=4;}
        static int lwl=-1;if(board.level!=lwl){switch(board.level){case 1:mxw=25.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=0.5;break;
            case 2:mxw=25.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=1.0;break;
            case 3:mxw=18.0;dqw=0.9;bhw=0.02;s3dq=0.6;s3bh=0.025;svm=1.0;break;
            case 4:mxw=15.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=1.0;break;
            case 5:mxw=15.0;dqw=0.9;bhw=0.02;s3dq=0.6;s3bh=0.025;svm=1.0;break;
            default:mxw=15.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=1.0;}lwl=board.level;}

        struct Raw{std::vector<std::pair<int,int>> path;int now,nxt;};std::vector<Raw> raws;raws.reserve(8192);
        static std::unordered_map<uint64_t,int> cache2;int nodes=0;std::vector<std::pair<int,int>> path;

        std::function<void(uint64_t,int,int,int)> dfs=[&](uint64_t mask,int r,int c,int target){if(++nodes>lmt)return;
            int cur=board.at(r,c).color(),fixed=target;if(fixed==0&&cur!=0)fixed=cur;
            if(path.size()>=2){int ns=path_score(board,path);Board nb=board.preview(path);
                auto h=bhash(nb);int nxt=0;auto it=cache2.find(h);if(it!=cache2.end())nxt=it->second;else{nxt=best1(nb,cmax);cache2[h]=nxt;}raws.push_back({path,ns,nxt});}
            std::pair<int,int> nb2[4];int nbc=0;for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!board.in_bounds(nr,nc)||(mask&bit(nr,nc,board.N)))continue;
                if(!comp(fixed,board.at(nr,nc).color()))continue;nb2[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){auto ev=[&](auto& p)->int{int pr=p.first,pc=p.second,pot=0;
                for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!board.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,board.N)))continue;if(comp(fixed,board.at(ppr,ppc).color()))pot++;}
                if(pure5){int pot2=0;for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!board.in_bounds(ppr,ppc)||(mask&bit(ppr,ppc,board.N)))continue;
                    if(!comp(fixed,board.at(ppr,ppc).color()))continue;for(int ee=0;ee<4;++ee){int ppr2=ppr+DR[ee],ppc2=ppc+DC[ee];
                        if(!board.in_bounds(ppr2,ppc2))continue;if(ppr2==pr&&ppc2==pc)continue;if(mask&bit(ppr2,ppc2,board.N))continue;
                        if(comp(fixed,board.at(ppr2,ppc2).color()))pot2++;}}return pot*10+pot2*2;}
                int base=(board.at(pr,pc).is_wildcard()?40:(board.at(pr,pc).is_bomb()?35:(board.at(pr,pc).color()==fixed?20:0)));return base+pot;};
                if(ev(nb2[j])>ev(nb2[i]))std::swap(nb2[i],nb2[j]);}
            for(int i=0;i<nbc;++i){path.push_back(nb2[i]);dfs(mask|bit(nb2[i].first,nb2[i].second,board.N),nb2[i].first,nb2[i].second,fixed);path.pop_back();if(nodes>lmt)return;}};

        std::vector<std::pair<int,int>> starts;
        if(pure5){std::vector<std::vector<int>> cc(bN,std::vector<int>(bN));std::vector<std::vector<bool>> vs(bN,std::vector<bool>(bN));
            for(int r=0;r<bN;++r)for(int c=0;c<bN;++c){if(vs[r][c])continue;int co=board.at(r,c).color();
                std::vector<std::pair<int,int>> cells;std::deque<std::pair<int,int>> q;q.push_back({r,c});vs[r][c]=true;
                while(!q.empty()){auto pp=q.front();q.pop_front();cells.push_back(pp);int cr=pp.first,ct=pp.second;
                    for(int d=0;d<4;++d){int nr=cr+DR[d],nc=ct+DC[d];if(!board.in_bounds(nr,nc)||vs[nr][nc])continue;
                        if(board.at(nr,nc).color()!=co)continue;vs[nr][nc]=true;q.push_back({nr,nc});}}
                for(auto&pp:cells)cc[pp.first][pp.second]=(int)cells.size();}
            for(int r=0;r<bN;++r)for(int c=0;c<bN;++c)starts.push_back({r,c});
            if(board.level==1)std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){return cc[a.first][a.second]>cc[b.first][b.second];});
            else std::sort(starts.begin(),starts.end(),[&](auto&a,auto&b){return cc[a.first][a.second]<cc[b.first][b.second];});}
        else for(int r=0;r<bN;++r)for(int c=0;c<bN;++c)starts.push_back({r,c});

        for(auto& pp:starts){path.clear();path.push_back(pp);dfs(bit(pp.first,pp.second,bN),pp.first,pp.second,0);if(nodes>lmt)break;}
        if(raws.empty()){for(int r=0;r<bN;++r)for(int c=0;c<bN;++c){path.clear();path.push_back({r,c});dfs(bit(r,c,bN),r,c,0);if(nodes>lmt)break;}}
        if(raws.empty())return fb(board);

        std::vector<int> order(raws.size());for(int i=0;i<(int)raws.size();++i)order[i]=i;
        std::sort(order.begin(),order.end(),[&](int a,int b){int va=raws[a].now+raws[a].nxt,vb=raws[b].now+raws[b].nxt;
            if(pure5){int pa=(int)raws[a].path.size(),pb=(int)raws[b].path.size();if(board.level==1){if(pa<6)va-=55;if(pb<6)vb-=55;}else{if(pa<6)va-=25;if(pb<6)vb-=25;}}
            return va!=vb?va>vb:raws[a].now>raws[b].now;});
        int kk1=std::min(k1(),(int)raws.size());order.resize(kk1);
        struct Cand{std::vector<std::pair<int,int>> path;int now,nxt;Board next_b;double v2;};std::vector<Cand> cands;cands.reserve(kk1);
        for(int idx:order){Board nb=board.preview(raws[idx].path);double dq=dropq(nb),bhv=bh(nb);
            double v2=(double)raws[idx].now+(double)raws[idx].nxt+dq*dqw+bhv*bhw;cands.push_back({raws[idx].path,raws[idx].now,raws[idx].nxt,std::move(nb),v2});}
        std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.v2>b.v2;});if(td())return cands[0].path;

        static std::unordered_map<uint64_t,int> cache3;int kk2=std::min(k2(),(int)cands.size());
        struct Scored{double val;int idx;};std::vector<Scored> scored;scored.reserve(kk1);
        for(int i=0;i<kk1;++i){double val=cands[i].v2;
            if(i<kk2&&!td()){auto sbp=bpw(cands[i].next_b,cmax);int nsc=sbp.first;auto& np=sbp.second;
                if(np.size()>=2){Board tb=cands[i].next_b.preview(np);auto th=bhash(tb);int tsc=0;
                    auto it=cache3.find(th);if(it!=cache3.end())tsc=it->second;else{tsc=best1(tb,cmax/3);cache3[th]=tsc;}
                    val=(double)cands[i].now+(double)nsc+(double)tsc+dropq(tb)*s3dq+bh(tb)*s3bh;}}
            scored.push_back({val,i});}
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});if(td())return cands[scored[0].idx].path;

        int kk3=std::min(k3(),(int)scored.size());
        for(int j=0;j<kk3;++j){
            int st=surv_steps(cands[scored[j].idx].next_b,6);
            if(st<=2)scored[j].val-=120.0*svm*(3.0-(double)st);
            else if(st<=3)scored[j].val-=50.0*svm;
            else if(st<=4)scored[j].val-=15.0*svm;
            int ss=surv_score(cands[scored[j].idx].next_b,std::min(3,6));
            scored[j].val+=(double)ss*0.02;
        }
        std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});
        if(!tf()&&cbw>0){int bsc=std::min(3,(int)scored.size());for(int j=0;j<bsc;++j){double bv=beam_e(cands[scored[j].idx].next_b,cbw,cbd);scored[j].val+=bv*0.3;}
            std::sort(scored.begin(),scored.end(),[](auto&a,auto&b){return a.val>b.val;});}

        int sel=0,ci=scored[sel].idx;double s3v=0;{auto sbp=bpw(cands[ci].next_b,cmax/2);if(sbp.second.size()>=2){
            Board tb=cands[ci].next_b.preview(sbp.second);int ts=best1(tb,cmax/3);s3v=(double)sbp.first+(double)ts;}}
        sp(cands[ci].next_b,cands[ci].now,cands[ci].nxt,board.level,s3v);
        if(cache2.size()>400000)cache2.clear();if(cache3.size()>300000)cache3.clear();return cands[ci].path;
    }
};

// 定义 adaptive feedback 方法（必须在类外定义，因为用到类的静态成员）
void ImprovedSolver::af(const Board& b){if(!pf)return;double ad=dropq(b),ab=bh(b);int an=best1(b,std::min(20000,cl()));
    double dd=ad-lpdq,pred=lpnxt+lpdq*dqw+lpbh*bhw,act=an+ad*dqw+ab*bhw;
    double nm=std::max(1.0,(std::abs(pred)+std::abs(act))/2.0),re=(act-pred)/nm;
    dqw+=0.002*re*dd;dqw=std::max(0.1,std::min(2.5,dqw));
    double bd=ab-lpbh;bhw+=0.0005*re*bd;bhw=std::max(-0.05,std::min(0.15,bhw));
    if(lplev>=1&&lplev<=5){if(an<lpnxt*0.5){mxw-=0.3;mxw=std::max(8.0,mxw);}else if(an>lpnxt*1.8){mxw+=0.2;mxw=std::min(35.0,mxw);}}
    if(lps3<0&&an>0){s3dq-=0.01;s3dq=std::max(0.2,s3dq);s3bh-=0.002;s3bh=std::max(0.0,s3bh);}
    int st2=surv_steps(b,std::min(4,6));if(st2<=2)svm+=0.05*(3.0-(double)st2);else if(st2<=3)svm-=0.02;else if(st2>=5)svm-=0.015;
    svm=std::max(0.1,std::min(3.0,svm));pf=false;}
void ImprovedSolver::sp(const Board& nb,int now,int nxt,int level,double s3){lpdq=dropq(nb);lpbh=bh(nb);lpnxt=nxt;lpnow=now;lplev=level;lps3=s3;pf=true;}

int ImprovedSolver::ccl=60000;int ImprovedSolver::ck1=200;int ImprovedSolver::ck2=100;int ImprovedSolver::ck3=20;
double ImprovedSolver::mxw=15.0;double ImprovedSolver::dqw=0.8;double ImprovedSolver::bhw=0.02;
double ImprovedSolver::s3dq=0.55;double ImprovedSolver::s3bh=0.025;double ImprovedSolver::svm=1.0;
int ImprovedSolver::cbw=8;int ImprovedSolver::cbd=4;

// ============================
// LocalJudge + 测试
// ============================
static int gen_block(std::mt19937& rng,int lv){if(lv<=2)return(rng()%5)+1;if(lv==3)return((rng()%100)<15)?0:(rng()%5)+1;
    if(lv==4){int c=(rng()%5)+1;return((rng()%100)<10)?-c:c;}if((rng()%100)<15)return 0;int base=(rng()%5)+1;return((rng()%100)<10)?-base:base;}

Board init_board(int lv,int sd,int N){Board b(N);b.level=lv;b.drop_queue->assign(N,std::vector<int>(1000));b.queue_ptr.assign(N,0);
    std::mt19937 rng(sd);for(int c=0;c<N;++c)for(int i=0;i<1000;++i)(*b.drop_queue)[c][i]=gen_block(rng,lv);
    std::mt19937 rng_b(sd^0x9E3779B9);for(int r=0;r<N;++r)for(int c=0;c<N;++c)b.at(r,c).value=gen_block(rng_b,lv);return b;}

int main(int argc,char** argv){int ns=(argc>1)?std::atoi(argv[1]):20,start=(argc>2)?std::atoi(argv[2]):1;
    printf("=== 2_final Test (%d seeds, start=%d) ===\n\n",ns,start);
    struct Lc{int l,N;};Lc levels[]={{1,10},{2,10},{3,10},{4,10},{5,12}};int gt=0,lt[5]={};
    for(int si=0;si<ns;++si){int sd=start+si,st2=0;
        for(auto& lc:levels){ImprovedSolver::reset_adaptive(lc.l);Board b=init_board(lc.l,sd,lc.N);ImprovedSolver solver;int step=0,sc=0;
            while(step<50&&!b.is_deadlocked()){auto path=solver.solve(b);if(path.size()<2)break;
                int g=path_score(b,path);sc+=g;step++;b=b.preview(path);}
            st2+=sc;lt[lc.l-1]+=sc;}gt+=st2;
        if((si+1)%5==0||si==0)printf("  seed %d: %d (avg %.1f)\n",sd,st2,(double)gt/(si+1));}
    printf("\n=== RESULTS ===\n");for(int i=0;i<5;++i)printf("  L%d avg: %.1f\n",i+1,(double)lt[i]/ns);
    printf("  TOTAL avg: %.1f  sum: %d\n",(double)gt/ns,gt);return 0;}
