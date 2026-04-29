#include <iostream>
#include <vector>
#include <random>

int gen_block(std::mt19937& rng, int level) {
    if (level <= 2) return (rng() % 5) + 1;
    else if (level == 3) return ((rng() % 100) < 15) ? 0 : (rng() % 5) + 1;
    else if (level == 4) {
        int color = (rng() % 5) + 1;
        return ((rng() % 100) < 10) ? -color : color;
    } else {
        if ((rng() % 100) < 15) return 0;
        int base = (rng() % 5) + 1;
        return ((rng() % 100) < 10) ? -base : base;
    }
}

int main() {
    int seed = 114514;
    for (int level = 1; level <= 5; ++level) {
        int N = (level <= 4) ? 10 : 12;
        // 初始棋盘用 rng_board = seed ^ 0x9E3779B9（与 LocalJudge/GameController 一致）
        std::mt19937 rng_board(seed ^ 0x9E3779B9);
        std::cout << "=== LEVEL " << level << " SEED " << seed << " SIZE " << N << " ===\n";

        // 打印初始棋盘
        std::cout << "[初始棋盘]\n";
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                int v = gen_block(rng_board, level);
                std::cout << v << " ";
            }
            std::cout << "\n";
        }

        // drop_queue 用 rng(seed)
        std::mt19937 rng(seed);
        std::cout << "\n[掉落队列前 10 个] col0: ";
        for (int i = 0; i < 10; ++i)
            std::cout << gen_block(rng, level) << " ";
        std::cout << "\n";
    }
    return 0;
}
