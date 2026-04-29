#if __cplusplus < 201402L
#error "C++14 required"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// ===== Cell & Board =====
struct Cell { int value = 1; int color() const { return value>0?value:(-value); } bool is_bomb() const { return value<0; } bool is_wildcard() const { return value==0; } };

struct Board {
    int N=0,level=1; std::vector<std::vector<Cell>> grid;
    std::shared_ptr<std::vector<std::vector<int>>> drop_queue;
    std::vector<int> queue_ptr;
    Board(int n=0):N(n),grid(n,std::vector<Cell>(n)),drop_queue(std::make_shared<std::vector<std::vector<int>>>()){}
    Cell& at(int r,int c){return grid[r][c];}
    const Cell& at(int r,int c)const{return grid[r][c];}
    bool in_bounds(int r,int c)const{return r>=0&&r<N&&c>=0&&c<N;}
    Board preview(const std::vector<std::pair<int,int>>& path)const{
        Board nb=*this; if(path.size()<2)return nb;
        auto& dq=*nb.drop_queue;
        std::vector<std::vector<bool>> in(N,std::vector<bool>(N));
        for(auto& p:path)in[p.first][p.second]=true;
        auto rm=in;
        if(level>=4)for(auto& p:path){int r=p.first,c=p.second; if(!at(r,c).is_bomb())continue;
            for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){int nr=r+dr,nc=c+dc; if(in_bounds(nr,nc)&&!in[nr][nc])rm[nr][nc]=true;}}
        for(int c=0;c<N;++c){std::vector<Cell> rem;for(int r=0;r<N;++r)if(!rm[r][c])rem.push_back(at(r,c));
            int emp=N-(int)rem.size(); for(int i=0;i<emp;++i)nb.at(i,c).value=dq[c][nb.queue_ptr[c]++];
            for(int i=0;i<(int)rem.size();++i)nb.at(emp+i,c)=rem[i];}
        return nb;
    }
    bool is_deadlocked()const{for(int r=0;r<N;++r)for(int c=0;c<N;++c){int ac=at(r,c).color();
        if(c+1<N){int c2=at(r,c+1).color();if(ac==c2||ac==0||c2==0)return false;}
        if(r+1<N){int c2=at(r+1,c).color();if(ac==c2||ac==0||c2==0)return false;}}return true;}
};

constexpr int DR[]={-1,1,0,0}, DC[]={0,0,-1,1};
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
static uint64_t bit(int r,int c,int N){int idx=r*N+c;if(idx>=64)return (1ULL<<(idx-64))<<1;return 1ULL<<idx;} // fixed for up to 128 cells
static uint64_t bhash(const Board& b){uint64_t h=1469598103934665603ULL,pr=1099511628211ULL;
    h^=(uint64_t)b.N;h*=pr;h^=(uint64_t)b.level;h*=pr;
    for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){h^=(uint64_t)(b.at(r,c).value+17);h*=pr;}return h;}

// ===== NN Module (52-dim features + MLP) =====
constexpr int NF=52;
struct BFeat { double f[NF]; BFeat(){for(int i=0;i<NF;++i)f[i]=0;} double& operator[](int i){return f[i];} };

static BFeat extract(const Board& b){
    BFeat bf; int N=b.N,tot=N*N,cc[6]={},bc=0,wc=0,as=0,ac=0; double ps[5]={};
    std::vector<std::vector<bool>> vis(N,std::vector<bool>(N)); std::vector<int> cs; int mc=0,cb[5]={};
    for(int r=0;r<N;++r)for(int c=0;c<N;++c){int v=b.at(r,c).value,co=std::abs(v);
        if(v==0)wc++; if(v<0){bc++;cc[std::abs(v)]++;} else if(v>0){cc[v]++;ps[v-1]+=r*N+c;}
        if(vis[r][c])continue; int sz=0; std::deque<std::pair<int,int>> q;
        q.push_back({r,c});vis[r][c]=true;
        while(!q.empty()){auto pp=q.front();q.pop_front();sz++;int cr=pp.first,ct=pp.second;
            int cco=std::abs(b.at(cr,ct).value);
            for(int d=0;d<4;++d){int nr=cr+DR[d],nc=ct+DC[d];
                if(nr<0||nr>=N||nc<0||nc>=N||vis[nr][nc])continue;
                if(!comp(cco,std::abs(b.at(nr,nc).value)))continue;vis[nr][nc]=true;q.push_back({nr,nc});}}
        cs.push_back(sz);mc=std::max(mc,sz);
        if(sz==1)cb[0]++;else if(sz<=3)cb[1]++;else if(sz<=6)cb[2]++;else if(sz<=10)cb[3]++;else cb[4]++;
    }
    for(int r=0;r<N;++r)for(int c=0;c<N;++c){int co=std::abs(b.at(r,c).value);
        if(c+1<N){int c2=std::abs(b.at(r,c+1).value);if(comp(co,c2)){ac++;if(co==c2&&co!=0)as++;}}
        if(r+1<N){int c2=std::abs(b.at(r+1,c).value);if(comp(co,c2)){ac++;if(co==c2&&co!=0)as++;}}}
    double ent=0; int tc=tot-wc; for(int i=1;i<=5;++i)if(cc[i]>0&&tc>0){double p=(double)cc[i]/tc;ent-=p*std::log(p);}
    double avg=0; if(!cs.empty()){double s=0;for(int sz:cs)s+=sz;avg=s/cs.size();}
    double vr=0; if(!cs.empty()){for(int sz:cs)vr+=(sz-avg)*(sz-avg);vr/=cs.size();}
    double dqc=0,dqw=0,dqb=0; if(b.drop_queue){auto& dq=*b.drop_queue;
        for(int c=0;c<N;++c){int qp=b.queue_ptr[c];for(int i=0;i<10;++i){if(qp+i>=1000)break;int v=dq[c][qp+i];
            if(v==0){dqw+=1.0/N/10;continue;}if(v<0){dqb+=1.0/N/10;continue;}
            int col=std::abs(v);for(int d=0;d<4;++d){int nr=DR[d],nc=c+DC[d];
                if(b.in_bounds(nr,nc)&&comp(col,std::abs(b.at(nr,nc).value)))dqc+=0.1/N/10;}}}}
    int blk=0,wht=0; for(int r=0;r<N;++r)for(int c=0;c<N;++c)if(std::abs(b.at(r,c).value)>0){if((r+c)%2==0)blk++;else wht++;}
    double rd=0,cd=0; for(int r=0;r<N;++r){std::set<int> s;for(int c=0;c<N;++c){int co=std::abs(b.at(r,c).value);if(co>0)s.insert(co);}rd+=(double)s.size()/N;}
    for(int c=0;c<N;++c){std::set<int> s;for(int r=0;r<N;++r){int co=std::abs(b.at(r,c).value);if(co>0)s.insert(co);}cd+=(double)s.size()/N;}
    int ce[5]={},co[5]={}; for(int r=0;r<N;++r)for(int c=0;c<N;++c){int v=b.at(r,c).value;if(v<=0)continue;int ci=v-1;if((r+c)%2==0)ce[ci]++;else co[ci]++;}
    int idx=0;
    for(int i=0;i<=5;++i)bf[idx++]=tot>0?(double)cc[i]/tot:0;
    bf[idx++]=tot>0?(double)wc/tot:0; bf[idx++]=tot>0?(double)bc/tot:0;
    bf[idx++]=(double)as/std::max(1,2*N*(N-1)); bf[idx++]=(double)ac/std::max(1,2*N*(N-1));
    bf[idx++]=tot>0?(double)as/tot:0; bf[idx++]=tot>0?(double)ac/tot:0;
    bf[idx++]=tot>0?(double)mc/tot:0; bf[idx++]=tot>0?avg/tot:0;
    bf[idx++]=tot>0?std::sqrt(std::max(0.0,vr))/tot:0; bf[idx++]=tot>0?(double)cs.size()/tot:0;
    int nc=std::max(1,(int)cs.size()); for(int i=0;i<5;++i)bf[idx++]=(double)cb[i]/nc;
    bf[idx++]=ent; for(int i=1;i<=5;++i)bf[idx++]=tot>0?(double)cc[i]/tot:0;
    for(int i=0;i<5;++i)bf[idx++]=cc[i+1]>1?1.0:0.0;
    bf[idx++]=rd; bf[idx++]=cd; bf[idx++]=b.is_deadlocked()?1.0:0.0;
    bf[idx++]=dqc; bf[idx++]=dqw; bf[idx++]=dqb;
    bf[idx++]=tot>0?(double)blk/tot:0; bf[idx++]=tot>0?(double)wht/tot:0;
    bf[idx++]=tot>0?(double)(blk+wht)/tot:0; double bp=blk+wht; bf[idx++]=bp>0?(double)blk/bp:0.5;
    double pe=0; for(int i=0;i<5;++i)if(cc[i+1]>0)pe+=(double)cc[i+1]/tot; bf[idx++]=pe;
    for(int i=0;i<5;++i){int tci=ce[i]+co[i];bf[idx++]=tci>0?(double)ce[i]/tci:0.5;}
    bf[idx++]=(double)b.level/5.0; bf[idx++]=(double)N/12.0;
    int rd2=0,cd2=0;for(int r=0;r<N;++r){int rc[6]={};for(int c=0;c<N;++c)rc[std::abs(b.at(r,c).value)]++;
        int dm=0;for(int i=1;i<=5;++i)if(rc[i]>rc[dm])dm=i;rd2+=rc[dm];}
    for(int c=0;c<N;++c){int cc2[6]={};for(int r=0;r<N;++r)cc2[std::abs(b.at(r,c).value)]++;
        int dm=0;for(int i=1;i<=5;++i)if(cc2[i]>cc2[dm])dm=i;cd2+=cc2[dm];}
    bf[idx++]=tot>0?(double)rd2/tot/N:0; bf[idx++]=tot>0?(double)cd2/tot/N:0;
    return bf;
}

struct Layer {
    std::vector<std::vector<double>> W; std::vector<double> b;
    mutable std::vector<double> z,a; int in_dim,out_dim; bool relu;
    Layer(int in,int out,bool r=true):in_dim(in),out_dim(out),relu(r){W.assign(out,std::vector<double>(in,0));b.assign(out,0);z.assign(out,0);a.assign(out,0);
        double sc=std::sqrt(2.0/std::max(1.0,(double)(in+out)));
        static std::mt19937 rng(114514); std::normal_distribution<double> nd(0,sc);
        for(int i=0;i<out;++i){for(int j=0;j<in;++j)W[i][j]=nd(rng);b[i]=0;}
    }
    void forward(const std::vector<double>& in){for(int i=0;i<out_dim;++i){double s=b[i];const auto& wr=W[i];
        for(int j=0;j<in_dim;++j)s+=wr[j]*in[j];if(s!=s)s=0;if(s>50)s=50;if(s<-50)s=-50;z[i]=s;a[i]=relu?std::max(0.0,s):std::tanh(s);}}
};

class NNModel {
    std::vector<Layer> layers_; double sf=1000.0,lr=0.001; int tc=0;
    struct Adam{std::vector<std::vector<double>> mW,vW;std::vector<double> mb,vb;double b1t=1,b2t=1;};
    std::vector<Adam> adam_;
public:
    NNModel(){std::vector<int> h={64,32};for(int sz:h){layers_.push_back(Layer(layers_.empty()?NF:layers_.back().out_dim,sz,true));}
        layers_.push_back(Layer(layers_.back().out_dim,1,false));for(auto& L:layers_){Adam a;a.mW.assign(L.out_dim,std::vector<double>(L.in_dim,0));a.vW=a.mW;a.mb.assign(L.out_dim,0);a.vb=a.mb;adam_.push_back(a);}}
    double predict(const BFeat& f){std::vector<double> in(f.f,f.f+NF);for(auto& L:layers_){L.forward(in);in=L.a;}return in[0]*sf;}
    void train(const BFeat& f,double tgt){const double b1=0.9,b2=0.999,eps=1e-8;int L=(int)layers_.size();
        double nt=tgt/std::max(1.0,sf);if(nt>1)nt=1;if(nt<-1)nt=-1;
        std::vector<std::vector<double>> act(L+1); act[0].assign(f.f,f.f+NF);
        for(int l=0;l<L;++l){layers_[l].forward(act[l]);act[l+1]=layers_[l].a;}
        double pd=act[L][0],er=pd-nt; std::vector<double> delta={2.0*er};
        if(!layers_.back().relu){double y=layers_.back().a[0];delta[0]*=(1-y*y);}
        for(int l=L-1;l>=0;--l){int od=layers_[l].out_dim,id=layers_[l].in_dim;std::vector<double> pd2(id,0);
            adam_[l].b1t*=b1;adam_[l].b2t*=b2;double b1t=adam_[l].b1t,b2t=adam_[l].b2t;
            for(int i=0;i<od;++i){double da=layers_[l].relu?(layers_[l].z[i]>0?delta[i]:0):delta[i];
                for(int j=0;j<id;++j)pd2[j]+=da*layers_[l].W[i][j];
                adam_[l].mb[i]=b1*adam_[l].mb[i]+(1-b1)*da;adam_[l].vb[i]=b2*adam_[l].vb[i]+(1-b2)*da*da;
                double mh=adam_[l].mb[i]/(1-b1t),vh=adam_[l].vb[i]/(1-b2t);layers_[l].b[i]-=lr*mh/(std::sqrt(vh)+eps);
                for(int j=0;j<id;++j){double gw=da*act[l][j];adam_[l].mW[i][j]=b1*adam_[l].mW[i][j]+(1-b1)*gw;
                    adam_[l].vW[i][j]=b2*adam_[l].vW[i][j]+(1-b2)*gw*gw;double mwh=adam_[l].mW[i][j]/(1-b1t),vwh=adam_[l].vW[i][j]/(1-b2t);layers_[l].W[i][j]-=lr*mwh/(std::sqrt(vwh)+eps);}}
            delta=pd2;}tc++;if(tc%10000==0){lr*=0.95;if(lr<1e-5)lr=1e-5;}}
    bool save(const std::string& p)const{std::ofstream o(p,std::ios::binary);if(!o)return false;o.write((char*)&sf,sizeof(sf));int nl=(int)layers_.size();o.write((char*)&nl,sizeof(nl));
        for(auto& L:layers_){o.write((char*)&L.in_dim,sizeof(L.in_dim));o.write((char*)&L.out_dim,sizeof(L.out_dim));o.write((char*)&L.relu,sizeof(L.relu));
            for(auto& r:L.W)o.write((char*)r.data(),sizeof(double)*L.in_dim);o.write((char*)L.b.data(),sizeof(double)*L.out_dim);}return true;}
    bool load(const std::string& p){std::ifstream i(p,std::ios::binary);if(!i)return false;i.read((char*)&sf,sizeof(sf));int nl;i.read((char*)&nl,sizeof(nl));layers_.clear();adam_.clear();
        for(int l=0;l<nl;++l){int id,od;bool rl;i.read((char*)&id,sizeof(id));i.read((char*)&od,sizeof(od));i.read((char*)&rl,sizeof(rl));Layer ly(id,od,rl);
            for(auto& r:ly.W)i.read((char*)r.data(),sizeof(double)*id);i.read((char*)ly.b.data(),sizeof(double)*od);layers_.push_back(std::move(ly));}
        for(auto& L:layers_){Adam a;a.mW.assign(L.out_dim,std::vector<double>(L.in_dim,0));a.vW=a.mW;a.mb.assign(L.out_dim,0);a.vb=a.mb;adam_.push_back(a);}return true;}
    void set_lr(double v){lr=v;} void set_scale(double v){sf=v;} int num_layers()const{return (int)layers_.size();}
};

// ===== NNEvaluator =====
class NNEval { NNModel model_; double nw=0.3; bool en=false; std::string wp="nn_weights.bin"; public:
    void enable(bool f){en=f;} bool ok()const{return en;}
    double eval(const Board& b){double h=heuristic(b);if(!en)return h;return nw*model_.predict(extract(b))+(1.0-nw)*h;}
    NNModel& model(){return model_;} void set_w(double w){nw=w;}
    bool load_w(const std::string& p=""){return model_.load(p.empty()?wp:p);}
    static double heuristic(const Board& b){if(b.is_deadlocked())return-1e5;int N=b.N,cp=0,sp=0,wc=0,bb=0,cnt[6]={},mxc=0;
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
};
static NNEval g_nn;

// ===== ImprovedSolver =====
class ImprovedSolver {
    static constexpr int MAX_T=10000; mutable std::chrono::steady_clock::time_point _t0; mutable bool _ti=false;
    void ts()const{if(!_ti){_t0=std::chrono::steady_clock::now();_ti=true;}}
    long long te()const{return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-_t0).count();}
    mutable int sc_=0; bool tf()const{return te()>MAX_T*0.90;} bool td()const{return te()>MAX_T*0.97;}
    static int ccl,ck1,ck2,ck3; static double mxw,dqw,bhw,s3dq,s3bh,svm; static int cbw,cbd;
    int cl()const{return tf()?(int)(ccl*0.65):ccl;} int k1()const{return tf()?(int)(ck1*0.7):ck1;}
    int k2()const{return tf()?(int)(ck2*0.7):ck2;} int k3()const{return tf()?(int)(ck3*0.7):ck3;}
    mutable int bN=6; mutable std::mt19937 rng{std::random_device{}()};
    // adaptive tracking
    mutable double lpdq=0,lpbh=0; mutable int lpnxt=0,lpnow=0,lplev=-1; mutable bool pf=false;
    mutable double lpmxc=0,lps3=0;
    void af(const Board& b){if(!pf)return;double ad=dropq(b),ab=bh(b);int an=best1(b,std::min(20000,cl()));
        double dd=ad-lpdq,pred=lpnxt+lpdq*dqw+lpbh*bhw,act=an+ad*dqw+ab*bhw;
        double nm=std::max(1.0,(std::abs(pred)+std::abs(act))/2.0),re=(act-pred)/nm;
        dqw+=0.002*re*dd;dqw=std::max(0.1,std::min(2.5,dqw));
        double bd=ab-lpbh;bhw+=0.0005*re*bd;bhw=std::max(-0.05,std::min(0.15,bhw));
        if(lplev>=1&&lplev<=5){if(an<lpnxt*0.5){mxw-=0.3;mxw=std::max(8.0,mxw);}else if(an>lpnxt*1.8){mxw+=0.2;mxw=std::min(35.0,mxw);}}
        if(lps3<0&&an>0){s3dq-=0.01;s3dq=std::max(0.2,s3dq);s3bh-=0.002;s3bh=std::max(0.0,s3bh);}
        int st=surv_steps(b,std::min(4,6));if(st<=2)svm+=0.05*(3.0-(double)st);else if(st<=3)svm-=0.02;else if(st>=5)svm-=0.015;
        svm=std::max(0.1,std::min(3.0,svm));pf=false;}
    void sp(const Board& nb,int now,int nxt,int level,double s3=0){lpdq=dropq(nb);lpbh=bh(nb);lpnxt=nxt;lpnow=now;lplev=level;lps3=s3;pf=true;}
public:
    void give_fb(const Board& b){if(!pf)return;af(b);}
private:
    static double bh(const Board& b){return g_nn.eval(b);}
    static double dropq(const Board& b){double q=0;auto& dq=*b.drop_queue;
        for(int c=0;c<b.N;++c){int qp=b.queue_ptr[c];for(int i=0;i<7;++i){if(qp+i>=1000)break;int v=dq[c][qp+i];
            if(v==0){q+=2.8;continue;}if(v<0){q+=0.9;continue;}int col=std::abs(v);double adj=0;
            if(b.in_bounds(0,c-1)&&comp(col,b.at(0,c-1).color()))adj+=0.5;
            if(b.in_bounds(0,c+1)&&comp(col,b.at(0,c+1).color()))adj+=0.5;
            if(b.in_bounds(1,c)&&comp(col,b.at(1,c).color()))adj+=0.3;q+=adj;
            if(i<3)for(int j=i+1;j<3&&j<7;++j){int v2=dq[c][qp+j];if(v2==0||v==v2)q+=0.2;}}}return q;}
    static int quickb(const Board& b){if(b.is_deadlocked())return 0;int best=0;
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){int co=b.at(r,c).color();
            if(b.at(r,c).is_wildcard()){for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(b.in_bounds(nr,nc))best=std::max(best,path_score(2));}continue;}
            if(c+1<b.N&&comp(co,b.at(r,c+1).color()))best=std::max(best,path_score(2));
            if(r+1<b.N&&comp(co,b.at(r+1,c).color()))best=std::max(best,path_score(2));}return best;}
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
        std::vector<bool> vis(b.N*b.N,false);
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target){if(++nodes>maxn)return;
            int co=b.at(r,c).color(),fixed=target;if(fixed==0&&co!=0)fixed=co;
            if(path.size()>=2){int sc=path_score(b,path);if(sc>best)best=sc;int u=ub(b,path,fixed);if(path_score(u)<=best)return;}
            std::pair<int,int> nb[4];int nbc=0;for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];
                if(!b.in_bounds(nr,nc)||vis[nr*b.N+nc])continue;if(!comp(fixed,b.at(nr,nc).color()))continue;nb[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){auto ev=[&](auto& p)->int{int pr=p.first,pc=p.second,pot=0;
                for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc])continue;if(comp(fixed,b.at(ppr,ppc).color()))pot++;}
                if(b.level<=2){int sm=0;for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc])continue;if(b.at(ppr,ppc).color()==fixed)sm++;}return(b.level==1)?pot*10-sm:pot*10+sm;}
                int base=0;if(b.at(pr,pc).is_wildcard())base=40;else if(b.at(pr,pc).is_bomb())base=(b.level==4?35:25);else if(b.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(b.level<=2){if(ev(nb[j])<ev(nb[i]))std::swap(nb[i],nb[j]);}else{if(ev(nb[j])>ev(nb[i]))std::swap(nb[i],nb[j]);}}
            for(int i=0;i<nbc;++i){path.push_back(nb[i]);vis[nb[i].first*b.N+nb[i].second]=true;dfs(nb[i].first,nb[i].second,fixed);vis[nb[i].first*b.N+nb[i].second]=false;path.pop_back();if(nodes>maxn)return;}};
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){path.clear();path.push_back({r,c});vis.assign(b.N*b.N,false);vis[r*b.N+c]=true;dfs(r,c,0);if(nodes>maxn)break;}return best;}
    static std::pair<int,std::vector<std::pair<int,int>>> bpw(const Board& b,int maxn){
        int bs=0,nodes=0;std::vector<std::pair<int,int>> bp,path;
        std::vector<bool> vis(b.N*b.N,false);
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target){if(++nodes>maxn)return;
            int co=b.at(r,c).color(),fixed=target;if(fixed==0&&co!=0)fixed=co;
            if(path.size()>=2){int sc=path_score(b,path);if(sc>bs||(sc==bs&&path<bp)){bs=sc;bp=path;}int u=ub(b,path,fixed);if(path_score(u)<=bs)return;}
            std::pair<int,int> nb[4];int nbc=0;for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(!b.in_bounds(nr,nc)||vis[nr*b.N+nc])continue;
                if(!comp(fixed,b.at(nr,nc).color()))continue;nb[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){auto ev=[&](auto& p)->int{int pr=p.first,pc=p.second,pot=0;
                for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc])continue;if(comp(fixed,b.at(ppr,ppc).color()))pot++;}
                if(b.level<=2){int sm=0;for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];if(!b.in_bounds(ppr,ppc)||vis[ppr*b.N+ppc])continue;if(b.at(ppr,ppc).color()==fixed)sm++;}return(b.level==1)?pot*10-sm:pot*10+sm;}
                int base=0;if(b.at(pr,pc).is_wildcard())base=40;else if(b.at(pr,pc).is_bomb())base=(b.level==4?35:25);else if(b.at(pr,pc).color()==fixed)base=20;return base+pot;};
                if(b.level<=2){if(ev(nb[j])<ev(nb[i]))std::swap(nb[i],nb[j]);}else{if(ev(nb[j])>ev(nb[i]))std::swap(nb[i],nb[j]);}}
            for(int i=0;i<nbc;++i){path.push_back(nb[i]);vis[nb[i].first*b.N+nb[i].second]=true;dfs(nb[i].first,nb[i].second,fixed);vis[nb[i].first*b.N+nb[i].second]=false;path.pop_back();if(nodes>maxn)return;}};
        for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){path.clear();path.push_back({r,c});vis.assign(b.N*b.N,false);vis[r*b.N+c]=true;dfs(r,c,0);if(nodes>maxn)break;}return {bs,bp};}
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
                double total=bm.acc+(double)sc+hv*(d==bd-1?0.5:0.15);cands.push_back({std::move(nb),bm.acc+(double)sc,total});
                for(int tries=0;tries<4&&(int)cands.size()<bw*4;++tries){auto sbp2=bpw(bm.bd,2000+tries*1000);
                    if(sbp2.second.size()<2||(sbp2.second[0]==bp2[0]&&sbp2.second.size()==bp2.size()))continue;
                    Board nb2=bm.bd.preview(sbp2.second);double hv2=bh(nb2);cands.push_back({std::move(nb2),bm.acc+(double)sbp2.first,hv2*(d==bd-1?0.5:0.15)});}}
            if(cands.empty())break;std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.val>b.val;});
            beams.clear();for(int i=0;i<std::min(bw,(int)cands.size());++i)beams.push_back({std::move(cands[i].bd),cands[i].acc});}
        return beams.empty()?0.0:beams[0].acc;}
    static std::vector<std::pair<int,int>> fb(const Board& b){for(int r=0;r<b.N;++r)for(int c=0;c<b.N;++c){
        if(b.at(r,c).is_wildcard())for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];if(b.in_bounds(nr,nc))return{{r,c},{nr,nc}};}
        int a=b.at(r,c).color();if(c+1<b.N&&comp(a,b.at(r,c+1).color()))return{{r,c},{r,c+1}};
        if(r+1<b.N&&comp(a,b.at(r+1,c).color()))return{{r,c},{r+1,c}};}return{{0,0},{0,1}};}
public:
    std::vector<std::pair<int,int>> solve(const Board& board){
        _ti=false;ts();++sc_;
        if(pf){if(board.level==lplev)af(board);else pf=false;}
        bN=board.N;bool pure5=(board.level<=2);        int lmt=(bN<=5?1500000:(pure5?3000000:(board.level==5?2500000:(board.level==3?1500000:1200000))));
        int cmax=cl();
        switch(board.level){            case 1:ccl=100000;ck1=250;ck2=120;ck3=25;cbw=10;cbd=6;break;
            case 2:ccl=100000;ck1=250;ck2=120;ck3=25;cbw=10;cbd=6;break;
            case 3:ccl=80000;ck1=150;ck2=80;ck3=15;cbw=4;cbd=4;break;
            case 4:ccl=100000;ck1=150;ck2=80;ck3=15;cbw=6;cbd=5;break;
            case 5:ccl=80000;ck1=100;ck2=50;ck3=10;cbw=4;cbd=4;break;
            default:ccl=60000;ck1=200;ck2=100;ck3=20;cbw=8;cbd=4;}
        static int lwl=-1;if(board.level!=lwl){switch(board.level){case 1:mxw=25.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=0.5;break;
            case 2:mxw=25.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=1.0;break;
            case 3:mxw=18.0;dqw=0.9;bhw=0.02;s3dq=0.6;s3bh=0.025;svm=1.0;break;
            case 4:mxw=15.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=1.0;break;
            case 5:mxw=15.0;dqw=0.9;bhw=0.02;s3dq=0.6;s3bh=0.025;svm=1.0;break;
            default:mxw=15.0;dqw=0.8;bhw=0.02;s3dq=0.55;s3bh=0.025;svm=1.0;}lwl=board.level;}
        struct Raw{std::vector<std::pair<int,int>> path;int now,nxt;};std::vector<Raw> raws;raws.reserve(8192);
        static std::unordered_map<uint64_t,int> cache2;int nodes=0;std::vector<std::pair<int,int>> path;
        std::vector<bool> vis_main(bN*bN,false);
        std::function<void(int,int,int)> dfs=[&](int r,int c,int target){if(++nodes>lmt)return;
            int cur=board.at(r,c).color(),fixed=target;if(fixed==0&&cur!=0)fixed=cur;
            if(path.size()>=2){int ns=path_score(board,path);Board nb=board.preview(path);
                auto h=bhash(nb);int nxt=0;auto it=cache2.find(h);if(it!=cache2.end())nxt=it->second;
                else{nxt=best1(nb,cmax);cache2[h]=nxt;}raws.push_back({path,ns,nxt});}
            std::pair<int,int> nb2[4];int nbc=0;for(int d=0;d<4;++d){int nr=r+DR[d],nc=c+DC[d];
                if(!board.in_bounds(nr,nc)||vis_main[nr*bN+nc])continue;
                if(!comp(fixed,board.at(nr,nc).color()))continue;nb2[nbc++]={nr,nc};}
            for(int i=0;i<nbc;++i)for(int j=i+1;j<nbc;++j){auto ev=[&](auto& p)->int{int pr=p.first,pc=p.second,pot=0;
                for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                    if(!board.in_bounds(ppr,ppc)||vis_main[ppr*bN+ppc])continue;if(comp(fixed,board.at(ppr,ppc).color()))pot++;}
                if(pure5){int pot2=0;for(int dd=0;dd<4;++dd){int ppr=pr+DR[dd],ppc=pc+DC[dd];
                    if(!board.in_bounds(ppr,ppc)||vis_main[ppr*bN+ppc])continue;
                    if(!comp(fixed,board.at(ppr,ppc).color()))continue;
                    for(int ee=0;ee<4;++ee){int ppr2=ppr+DR[ee],ppc2=ppc+DC[ee];
                        if(!board.in_bounds(ppr2,ppc2))continue;if(ppr2==pr&&ppc2==pc)continue;
                        if(vis_main[ppr2*bN+ppc2])continue;
                        if(comp(fixed,board.at(ppr2,ppc2).color()))pot2++;}}return pot*10+pot2*2;}
                int base=(board.at(pr,pc).is_wildcard()?40:(board.at(pr,pc).is_bomb()?35:(board.at(pr,pc).color()==fixed?20:0)));
                return base+pot;};if(ev(nb2[j])>ev(nb2[i]))std::swap(nb2[i],nb2[j]);}
            for(int i=0;i<nbc;++i){path.push_back(nb2[i]);vis_main[nb2[i].first*bN+nb2[i].second]=true;dfs(nb2[i].first,nb2[i].second,fixed);vis_main[nb2[i].first*bN+nb2[i].second]=false;path.pop_back();if(nodes>lmt)return;}};
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
        for(auto& pp:starts){path.clear();path.push_back(pp);vis_main.assign(bN*bN,false);vis_main[pp.first*bN+pp.second]=true;dfs(pp.first,pp.second,0);if(nodes>lmt)break;}
        if(raws.empty()){for(int r=0;r<bN;++r)for(int c=0;c<bN;++c){path.clear();path.push_back({r,c});vis_main.assign(bN*bN,false);vis_main[r*bN+c]=true;dfs(r,c,0);if(nodes>lmt)break;}}
        if(raws.empty())return fb(board);
        std::vector<int> order(raws.size());for(int i=0;i<(int)raws.size();++i)order[i]=i;
        std::sort(order.begin(),order.end(),[&](int a,int b){int va=raws[a].now+raws[a].nxt,vb=raws[b].now+raws[b].nxt;
            if(pure5){int pa=(int)raws[a].path.size(),pb=(int)raws[b].path.size();
                if(board.level==1){if(pa<6)va-=55;if(pb<6)vb-=55;}else{if(pa<6)va-=25;if(pb<6)vb-=25;}}
            return va!=vb?va>vb:raws[a].now>raws[b].now;});int kk1=std::min(k1(),(int)raws.size());order.resize(kk1);
        struct Cand{std::vector<std::pair<int,int>> path;int now,nxt;Board next_b;double v2;};std::vector<Cand> cands;cands.reserve(kk1);
        for(int idx:order){Board nb=board.preview(raws[idx].path);double dq=dropq(nb),bhv=bh(nb);
            double v2=(double)raws[idx].now+(double)raws[idx].nxt+dq*dqw+bhv*bhw;cands.push_back({raws[idx].path,raws[idx].now,raws[idx].nxt,std::move(nb),v2});}
        std::sort(cands.begin(),cands.end(),[](auto&a,auto&b){return a.v2>b.v2;});if(td())return cands[0].path;
        static std::unordered_map<uint64_t,int> cache3;int kk2=std::min(k2(),(int)cands.size());
        struct Scored{double val;int idx;};std::vector<Scored> scored;scored.reserve(kk1);
        for(int i=0;i<kk1;++i){double val=cands[i].v2;if(i<kk2&&!td()){
            auto sbp=bpw(cands[i].next_b,cmax);int nsc=sbp.first;auto& np=sbp.second;if(np.size()>=2){
                Board tb=cands[i].next_b.preview(np);auto th=bhash(tb);int tsc=0;
                auto it=cache3.find(th);if(it!=cache3.end())tsc=it->second;else{tsc=best1(tb,cmax/3);cache3[th]=tsc;}
                val=(double)cands[i].now+(double)nsc+(double)tsc+dropq(tb)*s3dq+bh(tb)*s3bh;}}scored.push_back({val,i});}
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
int ImprovedSolver::ccl=60000;int ImprovedSolver::ck1=200;int ImprovedSolver::ck2=100;int ImprovedSolver::ck3=20;
double ImprovedSolver::mxw=15.0;double ImprovedSolver::dqw=0.8;double ImprovedSolver::bhw=0.02;
double ImprovedSolver::s3dq=0.55;double ImprovedSolver::s3bh=0.025;double ImprovedSolver::svm=1.0;
int ImprovedSolver::cbw=8;int ImprovedSolver::cbd=4;

// ===== GameController =====
class GameController {public:struct DO{int col=0,value=0;};private:Board _b;int _lv=0,_st=0,_sc=0;bool _dn=false,_hf=false,_lv2=true;
    int _lr=0;std::string _pl;std::vector<std::pair<int,int>> _lp;std::vector<DO> _do;
    static int tpl(const std::string& l,int& lv,int& sd){int a,b,c,d;if(std::sscanf(l.c_str(),"LEVEL %d SEED %d SIZE %d STEPS %d",&a,&b,&c,&d)==4){lv=a;sd=b;return c;}return 0;}
    static bool tps(const std::string& l,int& st,int& sc,bool& v){char buf[16]={};if(std::sscanf(l.c_str(),"STEP %d SCORE %d %15s",&st,&sc,buf)>=3){v=(std::string(buf)=="VALID");return true;}return false;}
    static int gb(std::mt19937& rng,int lv){if(lv<=2)return(rng()%5)+1;if(lv==3)return((rng()%100)<15)?0:(rng()%5)+1;if(lv==4){int c=(rng()%5)+1;return((rng()%100)<10)?-c:c;}
        if((rng()%100)<15)return 0;int base=(rng()%5)+1;return((rng()%100)<10)?-base:base;}
    static void iq(Board& b,int sd,int N,int lv){b.level=lv;std::mt19937 rng(sd);b.drop_queue->clear();b.drop_queue->assign(N,std::vector<int>(1000));b.queue_ptr.assign(N,0);
        for(int c=0;c<N;++c)for(int i=0;i<1000;++i)(*b.drop_queue)[c][i]=gb(rng,lv);}
    static std::vector<int> rpc(const Board& b,const std::vector<std::pair<int,int>>& path){std::vector<int> rm(b.N,0);
        if(path.size()<2)return rm;std::vector<std::vector<bool>> in(b.N,std::vector<bool>(b.N)),tr(b.N,std::vector<bool>(b.N));
        for(auto& p:path)in[p.first][p.second]=tr[p.first][p.second]=true;if(b.level>=4)for(auto& p:path){int r=p.first,c=p.second;
            if(!b.at(r,c).is_bomb())continue;for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){int nr=r+dr,nc=c+dc;
                if(b.in_bounds(nr,nc)&&!in[nr][nc])tr[nr][nc]=true;}}
        for(int c=0;c<b.N;++c){int cnt=0;for(int r=0;r<b.N;++r)if(tr[r][c])++cnt;rm[c]=cnt;}return rm;}
    bool rl(std::string& l){if(!_pl.empty()){l=std::move(_pl);_pl.clear();return true;}return(bool)std::getline(std::cin,l);}
    Board rb(int N){Board b(N);for(int r=0;r<N;++r){std::string l;rl(l);std::istringstream ls(l);for(int c=0;c<N;++c)ls>>b.at(r,c).value;}return b;}
    void dt(){std::string l;while(std::cin.rdbuf()->in_avail()>0){if(!rl(l))break;if(l.empty()||l.find("LEVEL_END")!=std::string::npos)continue;
        if(l.find("FINAL_SCORE")!=std::string::npos){_dn=true;continue;}_pl=std::move(l);break;}}
public:const Board& board()const{return _b;}int level()const{return _lv;}int step()const{return _st;}int score()const{return _sc;}bool done()const{return _dn;}
    bool update(){std::string fl;while(true){if(!rl(fl)){_dn=true;return false;}if(!fl.empty())break;}
        if(fl.find("LEVEL_END")!=std::string::npos||fl.find("FINAL_SCORE")!=std::string::npos){_dn=true;return false;}
        int sd,NN=tpl(fl,_lv,sd);if(NN>0){Board nb=rb(NN);iq(nb,sd,NN,_lv);_b=std::move(nb);_st=0;_sc=0;_hf=false;_do.clear();dt();return true;}
        int st,sc;bool v;if(tps(fl,st,sc,v)){int ps=_sc;_st=st;_sc=sc;_lv2=v;_lr=v?(_sc-ps):-30;_hf=true;_do.clear();
            std::vector<int> rc;if(v&&!_lp.empty())rc=rpc(_b,_lp);Board prd=(v&&!_lp.empty())?_b.preview(_lp):_b;Board nb=rb(_b.N);
            if(!rc.empty())for(int c=0;c<_b.N;++c){int ds=rc[c];for(int r=0;r<ds&&r<_b.N;++r)_do.push_back({c,nb.at(r,c).value});}
            nb.level=_lv;nb.drop_queue=std::move(prd.drop_queue);nb.queue_ptr=std::move(prd.queue_ptr);_b=std::move(nb);_lp.clear();dt();
            if(!_pl.empty()){int nl,ns;int nN=tpl(_pl,nl,ns);if(nN>0){_lv=nl;_pl.clear();Board nb2=rb(nN);iq(nb2,ns,nN,nl);_b=std::move(nb2);_st=0;_sc=0;_hf=false;_do.clear();dt();}}return true;}
        _dn=true;return false;}
    void respond(const std::vector<std::pair<int,int>>& path){_lp=path;std::cout<<path.size();for(auto& p:path)std::cout<<' '<<p.first<<' '<<p.second;std::cout<<'\n';std::cout.flush();}
};

// ===== Main =====
#ifndef TEST_MODE
int main(){
    std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
    g_nn.enable(false);if(g_nn.load_w("nn_weights.bin"))g_nn.enable(true);
    GameController ctl;ImprovedSolver solver;
    while(ctl.update()){auto path=solver.solve(ctl.board());ctl.respond(path);}
    return 0;
}
#endif
