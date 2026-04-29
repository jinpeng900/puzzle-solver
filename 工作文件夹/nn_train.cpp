// nn_train.cpp - Neural Network Self-Play Trainer
#include "solver_core.h"
#include <fstream>
#include <cstdlib>

constexpr int NUM_F = 52;

// Board Features
struct BF { double f[NUM_F]; BF() { for(int i=0;i<NUM_F;++i) f[i]=0.0; } };

static BF extract_bf(const Board& b) {
    BF bf; int N=b.N, total=N*N;
    int cc[6]={}, bomb=0, wc=0, adj_s=0, adj_c=0;
    std::vector<std::vector<bool>> vis(N,std::vector<bool>(N,false));
    std::vector<int> cs; int mc=0; int cb[5]={};

    for (int r=0;r<N;++r) for (int c=0;c<N;++c) {
        int v=b.at(r,c).value, co=std::abs(v);
        if(v==0)wc++; if(v<0){bomb++;cc[co]++;} else if(v>0)cc[v]++;
        if(vis[r][c])continue;
        int sz=0; std::deque<std::pair<int,int>> q; q.push_back({r,c}); vis[r][c]=true;
        while(!q.empty()){auto p=q.front();q.pop_front();sz++;
            int cco=std::abs(b.at(p.first,p.second).value);
            for(int d=0;d<4;++d){int nr=p.first+DR[d],nc=p.second+DC[d];
                if(nr<0||nr>=N||nc<0||nc>=N||vis[nr][nc])continue;
                if(!compatible_color(cco,std::abs(b.at(nr,nc).value)))continue;
                vis[nr][nc]=true; q.push_back({nr,nc});}}
        cs.push_back(sz); mc=std::max(mc,sz);
        if(sz==1)cb[0]++;else if(sz<=3)cb[1]++;else if(sz<=6)cb[2]++;else if(sz<=10)cb[3]++;else cb[4]++;
    }

    for(int r=0;r<N;++r) for(int c=0;c<N;++c) {int ac=std::abs(b.at(r,c).value);
        if(c+1<N){int bc2=std::abs(b.at(r,c+1).value);if(compatible_color(ac,bc2)){adj_c++;if(ac==bc2&&ac!=0)adj_s++;}}
        if(r+1<N){int bc3=std::abs(b.at(r+1,c).value);if(compatible_color(ac,bc3)){adj_c++;if(ac==bc3&&ac!=0)adj_s++;}}}

    double ent=0; int tc=total-wc;
    for(int i=1;i<=5;++i)if(cc[i]>0&&tc>0){double p=(double)cc[i]/tc;ent-=p*std::log(p);}
    double aco=0; if(!cs.empty()){double s=0;for(int sz:cs)s+=sz;aco=s/cs.size();}
    double vc=0; if(!cs.empty()){for(int sz:cs)vc+=(sz-aco)*(sz-aco);vc/=cs.size();}

    double dqc=0,dqw=0,dqb=0;
    if(b.drop_queue){auto& dq=*b.drop_queue;
        for(int c=0;c<N;++c){int qp=b.queue_ptr[c];
            for(int i=0;i<10;++i){if(qp+i>=1000)break;int v=dq[c][qp+i];
                if(v==0){dqw+=1.0/N/10;continue;}if(v<0){dqb+=1.0/N/10;continue;}
                int col=std::abs(v);
                for(int d=0;d<4;++d){int nr=DR[d],nc=c+DC[d];
                    if(b.in_bounds(nr,nc)&&compatible_color(col,std::abs(b.at(nr,nc).value)))dqc+=0.1/N/10;}}}}

    int bl=0,wh=0;
    for(int r=0;r<N;++r)for(int c=0;c<N;++c)if(std::abs(b.at(r,c).value)>0){if((r+c)%2==0)bl++;else wh++;}

    double rd=0,cd=0;
    for(int r=0;r<N;++r){std::set<int> s;for(int c=0;c<N;++c){int co=std::abs(b.at(r,c).value);if(co>0)s.insert(co);}rd+=(double)s.size()/N;}
    for(int c=0;c<N;++c){std::set<int> s;for(int r=0;r<N;++r){int co=std::abs(b.at(r,c).value);if(co>0)s.insert(co);}cd+=(double)s.size()/N;}

    int ce[5]={},co2[5]={};
    for(int r=0;r<N;++r)for(int c=0;c<N;++c){int v=b.at(r,c).value;if(v<=0)continue;int ci=v-1;if((r+c)%2==0)ce[ci]++;else co2[ci]++;}

    int idx=0;
    for(int i=0;i<=5;++i)bf.f[idx++]=total>0?(double)cc[i]/total:0;
    bf.f[idx++]=total>0?(double)wc/total:0; bf.f[idx++]=total>0?(double)bomb/total:0;
    bf.f[idx++]=(double)adj_s/std::max(1,2*N*(N-1)); bf.f[idx++]=(double)adj_c/std::max(1,2*N*(N-1));
    bf.f[idx++]=total>0?(double)adj_s/total:0; bf.f[idx++]=total>0?(double)adj_c/total:0;
    bf.f[idx++]=total>0?(double)mc/total:0; bf.f[idx++]=total>0?aco/total:0;
    bf.f[idx++]=total>0?std::sqrt(std::max(0.0,vc))/total:0; bf.f[idx++]=total>0?(double)cs.size()/total:0;
    int nc=std::max(1,(int)cs.size()); for(int i=0;i<5;++i)bf.f[idx++]=(double)cb[i]/nc;
    bf.f[idx++]=ent; for(int i=1;i<=5;++i)bf.f[idx++]=total>0?(double)cc[i]/total:0;
    for(int i=0;i<5;++i)bf.f[idx++]=cc[i+1]>1?1.0:0.0;
    bf.f[idx++]=rd; bf.f[idx++]=cd; bf.f[idx++]=b.is_deadlocked()?1.0:0.0;
    bf.f[idx++]=dqc; bf.f[idx++]=dqw; bf.f[idx++]=dqb;
    bf.f[idx++]=total>0?(double)bl/total:0; bf.f[idx++]=total>0?(double)wh/total:0;
    bf.f[idx++]=total>0?(double)(bl+wh)/total:0;
    double bp=bl+wh; bf.f[idx++]=bp>0?(double)bl/bp:0.5;
    double pe=0; for(int ci=0;ci<5;++ci)if(cc[ci+1]>0)pe+=(double)cc[ci+1]/total; bf.f[idx++]=pe;
    for(int ci=0;ci<5;++ci){int t=ce[ci]+co2[ci];bf.f[idx++]=t>0?(double)ce[ci]/t:0.5;}
    bf.f[idx++]=(double)b.level/5.0; bf.f[idx++]=(double)N/12.0;
    int rd2=0,cd2=0;
    for(int r=0;r<N;++r){int rc[6]={};for(int c=0;c<N;++c)rc[std::abs(b.at(r,c).value)]++;int dom=0;for(int i=1;i<=5;++i)if(rc[i]>rc[dom])dom=i;rd2+=rc[dom];}
    for(int c=0;c<N;++c){int cc2[6]={};for(int r=0;r<N;++r)cc2[std::abs(b.at(r,c).value)]++;int dom=0;for(int i=1;i<=5;++i)if(cc2[i]>cc2[dom])dom=i;cd2+=cc2[dom];}
    bf.f[idx++]=total>0?(double)rd2/total/N:0; bf.f[idx++]=total>0?(double)cd2/total/N:0;
    return bf;
}

// Simple MLP + Adam
class SimpleNN {
    int L;
    std::vector<std::vector<std::vector<double>>> W;
    std::vector<std::vector<double>> b, z, a;
    std::vector<bool> relu;
    std::vector<std::vector<std::vector<double>>> mW, vW;
    std::vector<std::vector<double>> mb, vb;
    double scale=1000.0;

public:
    SimpleNN() {
        int sizes[]={NUM_F,64,32,1}; L=3;
        W.resize(L);b.resize(L);z.resize(L);a.resize(L);relu.resize(L);
        mW.resize(L);vW.resize(L);mb.resize(L);vb.resize(L);
        std::mt19937 rng(114514);
        for(int l=0;l<L;++l){
            int in=sizes[l],out=sizes[l+1];relu[l]=(l<L-1);
            W[l].assign(out,std::vector<double>(in,0));b[l].assign(out,0);
            z[l].assign(out,0);a[l].assign(out,0);
            mW[l].assign(out,std::vector<double>(in,0));vW[l].assign(out,std::vector<double>(in,0));
            mb[l].assign(out,0);vb[l].assign(out,0);
            double sc=std::sqrt(2.0/(in+out));
            std::normal_distribution<double> dist(0,sc);
            for(int i=0;i<out;++i){for(int j=0;j<in;++j)W[l][i][j]=dist(rng);b[l][i]=0;}
        }
    }

    double predict(const BF& bf){
        std::vector<double> inp(bf.f,bf.f+NUM_F);
        for(int l=0;l<L;++l){
            for(int i=0;i<(int)b[l].size();++i){double s=b[l][i];
                for(int j=0;j<(int)W[l][i].size();++j)s+=W[l][i][j]*inp[j];
                z[l][i]=s;a[l][i]=relu[l]?std::max(0.0,s):std::tanh(s);}
            inp=a[l];
        }
        return inp[0]*scale;
    }

    void train_one(const BF& bf, double target){
        double nt=target/scale;if(nt>1.0)nt=1.0;if(nt<-1.0)nt=-1.0;
        std::vector<std::vector<double>> act;act.push_back(std::vector<double>(bf.f,bf.f+NUM_F));
        for(int l=0;l<L;++l){act.push_back(std::vector<double>());
            int in=(int)W[l][0].size(),out=(int)b[l].size();
            for(int i=0;i<out;++i){double s=b[l][i];
                for(int j=0;j<in;++j)s+=W[l][i][j]*act[l][j];
                z[l][i]=s;a[l][i]=relu[l]?std::max(0.0,s):std::tanh(s);}
            act[l+1]=a[l];}
        double err=act.back()[0]-nt;
        std::vector<double> delta={2.0*err};
        double b1t=1.0,b2t=1.0;const double lr=0.001,b1=0.9,b2=0.999,eps=1e-8;
        for(int l=L-1;l>=0;--l){
            b1t*=b1;b2t*=b2;
            std::vector<double> pd(W[l][0].size(),0);
            for(int i=0;i<(int)b[l].size();++i){
                double da=relu[l]?(z[l][i]>0?delta[i]:0):delta[i];
                mb[l][i]=b1*mb[l][i]+(1-b1)*da;vb[l][i]=b2*vb[l][i]+(1-b2)*da*da;
                double mh=mb[l][i]/(1-b1t),vh=vb[l][i]/(1-b2t);
                b[l][i]-=lr*mh/(std::sqrt(vh)+eps);
                for(int j=0;j<(int)W[l][i].size();++j){
                    double g=da*act[l][j];
                    mW[l][i][j]=b1*mW[l][i][j]+(1-b1)*g;vW[l][i][j]=b2*vW[l][i][j]+(1-b2)*g*g;
                    double mwh=mW[l][i][j]/(1-b1t),vwh=vW[l][i][j]/(1-b2t);
                    W[l][i][j]-=lr*mwh/(std::sqrt(vwh)+eps);
                    pd[j]+=da*W[l][i][j];}
            }
            delta=pd;
        }
    }

    bool save(const char* path){std::ofstream o(path,std::ios::binary);
        if(!o)return false;o.write((char*)&scale,sizeof(scale));
        o.write((char*)&L,sizeof(L));
        for(int l=0;l<L;++l){int in=(int)W[l][0].size(),out=(int)b[l].size();
            o.write((char*)&in,sizeof(in));o.write((char*)&out,sizeof(out));
            o.write((char*)&relu[l],sizeof(relu[l]));
            for(auto& r:W[l])o.write((char*)r.data(),sizeof(double)*in);
            o.write((char*)b[l].data(),sizeof(double)*out);}
        return true;}

    bool load(const char* path){std::ifstream i(path,std::ios::binary);
        if(!i)return false;i.read((char*)&scale,sizeof(scale));
        i.read((char*)&L,sizeof(L));W.resize(L);b.resize(L);z.resize(L);a.resize(L);relu.resize(L);
        mW.resize(L);vW.resize(L);mb.resize(L);vb.resize(L);
        for(int l=0;l<L;++l){int in,out;bool rl;i.read((char*)&in,sizeof(in));i.read((char*)&out,sizeof(out));i.read((char*)&rl,sizeof(rl));
            relu[l]=rl;W[l].assign(out,std::vector<double>(in,0));b[l].assign(out,0);z[l].assign(out,0);a[l].assign(out,0);
            mW[l].assign(out,std::vector<double>(in,0));vW[l].assign(out,std::vector<double>(in,0));mb[l].assign(out,0);vb[l].assign(out,0);
            for(auto& r:W[l])i.read((char*)r.data(),sizeof(double)*in);
            i.read((char*)b[l].data(),sizeof(double)*out);}
        return true;}
};

// Replay Buffer
struct Exp { BF f; double v; };
static std::vector<Exp> buffer;
static std::mt19937 rng(42);

static int genb(std::mt19937& r, int lv){
    if(lv<=2)return(r()%5)+1;if(lv==3)return((r()%100)<15)?0:(r()%5)+1;
    if(lv==4){int c=(r()%5)+1;return((r()%100)<10)?-c:c;}
    if((r()%100)<15)return 0;int b=(r()%5)+1;return((r()%100)<10)?-b:b;}

static Board make_board(int lv, int sd, int N){
    Board b(N);b.level=lv;b.drop_queue=std::make_shared<std::vector<std::vector<int>>>();
    b.drop_queue->assign(N,std::vector<int>(1000));b.queue_ptr.assign(N,0);
    std::mt19937 r(sd);for(int c=0;c<N;++c)for(int i=0;i<1000;++i)(*b.drop_queue)[c][i]=genb(r,lv);
    std::mt19937 r2(sd^0x9E3779B9);for(int r=0;r<N;++r)for(int c=0;c<N;++c)b.at(r,c).value=genb(r2,lv);
    return b;}

static int collect_game(ImprovedSolver& sv, int lv, int sd, int N){
    Board b=make_board(lv,sd,N);int st=0,sc=0;
    std::vector<BF> ss;std::vector<double> rs;
    while(st<50&&!b.is_deadlocked()){ss.push_back(extract_bf(b));int prev=sc;
        auto p=sv.solve(b);if(p.size()<2)break;
        int g=path_score(b,p);sc+=g;rs.push_back((double)g);
        b=b.preview(p);st++;}
    if(!ss.empty()){const double d=0.95;double f=0;int si=(int)rs.size();
        for(int i=(int)ss.size()-1;i>=0;--i){si--;if(si>=0)f=rs[si]+d*f;buffer.push_back({ss[i],f});}}
    return sc;}

int main(int argc, char** argv){
    int ng=(argc>1)?std::atoi(argv[1]):30;
    int ne=(argc>2)?std::atoi(argv[2]):50;
    const char* lp=(argc>3)?argv[3]:nullptr;

    printf("=== NN Self-Play Trainer ===\nGames: %d, Epochs: %d\n\n",ng,ne);
    init_best_params();
    ImprovedSolver sv;
    SimpleNN nn;
    if(lp&&nn.load(lp))printf("[Load] Weights from %s\n",lp);

    int lvs[]={1,2,3,4,5},Ns[]={10,10,10,10,12};
    printf("[Collect] %d games...\n",ng);
    for(int g=0;g<ng;++g){int sd=(int)rng()%1000000+1;
        for(int i=0;i<5;++i)collect_game(sv,lvs[i],sd,Ns[i]);
        if((g+1)%5==0)printf("  %d/%d, samples: %zu\n",g+1,ng,buffer.size());}
    printf("[Collect] Total: %zu samples\n",buffer.size());

    if(buffer.size()<100){printf("Not enough samples!\n");return 1;}
    printf("[Train] %d epochs on %zu samples...\n",ne,buffer.size());
    std::uniform_int_distribution<size_t> ud(0,buffer.size()-1);
    for(int ep=0;ep<ne;++ep){
        for(size_t i=0;i<buffer.size();++i){auto& e=buffer[ud(rng)];
            nn.predict(e.f);nn.train_one(e.f,e.v);}
        if((ep+1)%10==0||ep==0)printf("  Epoch %d/%d\n",ep+1,ne);}
    printf("[Train] Done\n");

    nn.save("nn_weights.bin");
    printf("[Done] Weights -> nn_weights.bin\n");
    return 0;
}
