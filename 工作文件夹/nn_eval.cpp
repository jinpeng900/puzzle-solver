// nn_eval.cpp - Neural Network Evaluator (ASCII-only, no Chinese chars)
#include "solver_core.h"
#include <fstream>
#include <set>

NNEvaluator g_nn_eval;

// 52-dim board feature extraction (must match nn_train.cpp exactly)
NNFeatures nn_extract_features(const Board& b) {
    NNFeatures bf; int N=b.N, total=N*N;
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
    double blp=bl+wh; bf.f[idx++]=blp>0?(double)bl/blp:0.5;
    double pe=0; for(int ci=0;ci<5;++ci)if(cc[ci+1]>0)pe+=(double)cc[ci+1]/total; bf.f[idx++]=pe;
    for(int ci=0;ci<5;++ci){int t=ce[ci]+co2[ci];bf.f[idx++]=t>0?(double)ce[ci]/t:0.5;}
    bf.f[idx++]=(double)b.level/5.0; bf.f[idx++]=(double)N/12.0;
    int rd2=0,cd2=0;
    for(int r=0;r<N;++r){int rc[6]={};for(int c=0;c<N;++c)rc[std::abs(b.at(r,c).value)]++;int dom=0;for(int i=1;i<=5;++i)if(rc[i]>rc[dom])dom=i;rd2+=rc[dom];}
    for(int c=0;c<N;++c){int cc2[6]={};for(int r=0;r<N;++r)cc2[std::abs(b.at(r,c).value)]++;int dom=0;for(int i=1;i<=5;++i)if(cc2[i]>cc2[dom])dom=i;cd2+=cc2[dom];}
    bf.f[idx++]=total>0?(double)rd2/total/N:0; bf.f[idx++]=total>0?(double)cd2/total/N:0;
    return bf;
}

NNEvaluator::NNEvaluator() : L(0), scale(1000.0) {}

double NNEvaluator::predict(const NNFeatures& bf) {
    if (L <= 0) return 0.0;
    std::vector<double> inp(bf.f, bf.f + NN_NUM_F);
    for (int l = 0; l < L; ++l) {
        int in = (int)W[l][0].size(), out = (int)b[l].size();
        std::vector<double> next(out, 0);
        for (int i = 0; i < out; ++i) {
            double s = b[l][i];
            for (int j = 0; j < in; ++j) s += W[l][i][j] * inp[j];
            next[i] = relu[l] ? std::max(0.0, s) : std::tanh(s);
        }
        inp = next;
    }
    return inp[0] * scale;
}

bool NNEvaluator::load(const char* path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.read((char*)&scale, sizeof(scale));
    ifs.read((char*)&L, sizeof(L));
    W.resize(L); b.resize(L); relu.resize(L);
    for (int l = 0; l < L; ++l) {
        int in, out; bool rl;
        ifs.read((char*)&in, sizeof(in));
        ifs.read((char*)&out, sizeof(out));
        ifs.read((char*)&rl, sizeof(rl));
        relu[l] = rl;
        W[l].assign(out, std::vector<double>(in, 0));
        b[l].assign(out, 0);
        for (auto& row : W[l]) ifs.read((char*)row.data(), sizeof(double) * in);
        ifs.read((char*)b[l].data(), sizeof(double) * out);
    }
    return true;
}
