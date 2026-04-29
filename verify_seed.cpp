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
        std::mt19937 rng(seed);
        std::cout << "=== LEVEL " << level << " SEED " << seed << " SIZE " << N << " ===\n";
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                int v = gen_block(rng, level);
                std::cout << v << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
    return 0;
}
