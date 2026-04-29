// test_2final.cpp — 直接测试 2_final.cpp 求解器
// 编译: g++ -std=c++14 -O2 -o test_2f.exe test_2final.cpp
#define TEST_MODE
#include "2_final.cpp"
// 去掉 main() — 通过 undef + 重定义解决
// 直接在下面定义新的 main

static int gb(std::mt19937& rng,int lv){if(lv<=2)return(rng()%5)+1;if(lv==3)return((rng()%100)<15)?0:(rng()%5)+1;
    if(lv==4){int c=(rng()%5)+1;return((rng()%100)<10)?-c:c;}if((rng()%100)<15)return 0;int base=(rng()%5)+1;return((rng()%100)<10)?-base:base;}
Board init_board(int lv,int sd,int N){Board b(N);b.level=lv;b.drop_queue->assign(N,std::vector<int>(1000));b.queue_ptr.assign(N,0);
    std::mt19937 rng(sd);for(int c=0;c<N;++c)for(int i=0;i<1000;++i)(*b.drop_queue)[c][i]=gb(rng,lv);
    std::mt19937 rng_b(sd^0x9E3779B9);for(int r=0;r<N;++r)for(int c=0;c<N;++c)b.at(r,c).value=gb(rng_b,lv);return b;}

int main(int argc,char** argv){
    int ns=(argc>1)?std::atoi(argv[1]):10,start=(argc>2)?std::atoi(argv[2]):1;
    printf("=== 2_final test(%d seeds,s=%d) ===\n",ns,start);
    struct Lc{int l,N;};Lc lv[]={{1,10},{2,10},{3,10},{4,10},{5,12}};int gt=0,lt[5]={};
    for(int si=0;si<ns;++si){int sd=start+si,st2=0;
        for(auto& lc:lv){Board b=init_board(lc.l,sd,lc.N);ImprovedSolver sv;int step=0,sc=0;
            while(step<50&&!b.is_deadlocked()){auto path=sv.solve(b);if(path.size()<2)break;
                int g=path_score(b,path);sc+=g;step++;b=b.preview(path);}st2+=sc;lt[lc.l-1]+=sc;}gt+=st2;
        printf("  s%d:%d (avg%.1f)\n",sd,st2,(double)gt/(si+1));}
    printf("\n===R===\n");for(int i=0;i<5;++i)printf(" L%d:%.1f\n",i+1,(double)lt[i]/ns);
    printf(" TOT:%.1f\n",(double)gt/ns);return 0;}
