/**
 * train_nn.cpp
 * 从训练数据训练神经网络
 * 编译: g++ -std=c++17 -O2 -o train_nn scripts/train_nn.cpp
 * 用法: ./train_nn <data_file> <weights_out> [epochs] [lr]
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include "../nn_solver/board_features.h"
#include "../nn_solver/nn_model.h"
#include "../nn_solver/replay_buffer.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <data.bin> <weights_out.bin> [epochs=50] [lr=0.001]\n", argv[0]);
        return 1;
    }

    const char* data_path = argv[1];
    const char* weights_path = argv[2];
    int epochs = (argc > 3) ? std::atoi(argv[3]) : 50;
    double lr = (argc > 4) ? std::atof(argv[4]) : 0.001;

    // 加载数据
    std::printf("[Train] Loading data from %s\n", data_path);
    ReplayBuffer buf;
    if (!buf.load(data_path)) {
        std::fprintf(stderr, "[Train] ERROR: Failed to load %s\n", data_path);
        return 1;
    }
    std::printf("[Train] Loaded %zu samples\n", buf.size());

    // 创建网络
    NNModel model;
    model.set_lr(lr);

    // 准备训练数据
    const auto& all_data = buf.all();
    std::vector<BoardFeatures> feats;
    std::vector<double> targets;
    feats.reserve(all_data.size());
    targets.reserve(all_data.size());
    for (auto& e : all_data) {
        feats.push_back(e.features);
        targets.push_back(e.value);
    }

    // 打乱数据
    std::vector<size_t> indices(all_data.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    // 80/20 train/val split
    size_t train_n = indices.size() * 8 / 10;
    size_t val_n = indices.size() - train_n;

    std::printf("[Train] Train=%zu Val=%zu Epochs=%d LR=%f\n", train_n, val_n, epochs, lr);

    // 训练循环
    int batch_size = 128;
    for (int ep = 0; ep < epochs; ++ep) {
        double train_loss = 0;
        for (size_t bi = 0; bi < train_n; bi += batch_size) {
            size_t end = std::min(bi + (size_t)batch_size, train_n);
            for (size_t i = bi; i < end; ++i) {
                size_t idx = indices[i];
                model.train(feats[idx], targets[idx]);
                double pred = model.predict(feats[idx]);
                double err = pred - targets[idx];
                train_loss += err * err;
            }
        }
        train_loss /= (double)train_n;

        // 验证损失
        double val_loss = 0;
        for (size_t i = train_n; i < indices.size(); ++i) {
            size_t idx = indices[i];
            double pred = model.predict(feats[idx]);
            double err = pred - targets[idx];
            val_loss += err * err;
        }
        val_loss /= (double)val_n;

        if (ep % 5 == 0 || ep == epochs - 1)
            std::printf("  Epoch %3d: train_loss=%.2f val_loss=%.2f\n", ep, std::sqrt(train_loss), std::sqrt(val_loss));
    }

    // 保存权重
    if (model.save(weights_path))
        std::printf("[Train] Weights saved to %s\n", weights_path);
    else
        std::fprintf(stderr, "[Train] ERROR: Failed to save weights\n");

    return 0;
}
