#include <iostream>
#include <random>
#include <iomanip>

static int gen_block_rng(std::mt19937& rng, int level) {
    if (level<=2) return (rng()%5)+1;
    if (level==3) return ((rng()%100)<15)?0:(rng()%5)+1;
    if (level==4) { int c=(rng()%5)+1; return ((rng()%100)<10)?-c:c; }
    if ((rng()%100)<15) return 0;
    int base=(rng()%5)+1; return ((rng()%100)<10)?-base:base;
}

static int gen_block_uni(std::mt19937& rng, int level) {
    std::uniform_int_distribution<int> d100(0,99);
    std::uniform_int_distribution<int> d5(0,4);
    if (level<=2) return d5(rng)+1;
    if (level==3) { if (d100(rng)<15) return 0; return d5(rng)+1; }
    if (level==4) { int c=d5(rng)+1; return (d100(rng)<10)?-c:c; }
    if (d100(rng)<15) return 0;
    int base=d5(rng)+1; return (d100(rng)<10)?-base:base;
}

int main(int argc, char** argv) {
    int level = argc>1 ? std::stoi(argv[1]) : 1;
    int seed  = argc>2 ? std::stoi(argv[2]) : 114514;
    int N = (level == 5) ? 12 : 10;

    std::cout << "Level " << level << "  seed=" << seed << "  N=" << N << "\n\n";

    std::cout << "=== rng()%5 方案 ===\n";
    std::mt19937 rng1(seed ^ 0x9E3779B9);
    for (int r = 0; r < N; ++r) {
        std::cout << "row " << r << ":";
        for (int c = 0; c < N; ++c) {
            int v = gen_block_rng(rng1, level);
            if (v >= 0) std::cout << ' ';
            std::cout << std::setw(2) << v << ' ';
        }
        std::cout << "\n";
    }

    std::cout << "\n=== uniform_int_distribution 方案 ===\n";
    std::mt19937 rng2(seed ^ 0x9E3779B9);
    for (int r = 0; r < N; ++r) {
        std::cout << "row " << r << ":";
        for (int c = 0; c < N; ++c) {
            int v = gen_block_uni(rng2, level);
            if (v >= 0) std::cout << ' ';
            std::cout << std::setw(2) << v << ' ';
        }
        std::cout << "\n";
    }

    // 期望对比
    int expect[10] = {1,3,0,3,2,0,3,0,3,0};
    {
        std::mt19937 rng3(seed ^ 0x9E3779B9);
        std::cout << "\n期望 row0: ";
        for (int c=0;c<10;++c) std::cout << std::setw(2) << expect[c] << " ";
        std::cout << "\n";

        int m0=0, m1=0;
        for (int c=0;c<N;++c) {
            int v0 = gen_block_rng(rng3, level);
            if (v0 == expect[c]) m0++;
        }
        std::mt19937 rng4(seed ^ 0x9E3779B9);
        for (int c=0;c<N;++c) {
            int v1 = gen_block_uni(rng4, level);
            if (v1 == expect[c]) m1++;
        }
        std::cout << "rng()%5 匹配: " << m0 << "/10\n";
        std::cout << "uni     匹配: " << m1 << "/10\n";
    }
    return 0;
}
