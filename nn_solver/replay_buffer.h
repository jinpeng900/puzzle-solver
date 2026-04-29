#ifndef REPLAY_BUFFER_H
#define REPLAY_BUFFER_H

#include "board_features.h"
#include <vector>
#include <random>
#include <fstream>
#include <cstdint>

// ============================================================
// 经验回放缓冲区
// 存储 (features, target_score, step, level) 元组
// 支持二进制持久化，跨训练会话复用
// ============================================================
struct Experience {
    BoardFeatures features;
    double value;   // 目标值：从该状态开始能获得的未来总得分
    int step;
    int level;
};

class ReplayBuffer {
    std::vector<Experience> buffer_;
    size_t capacity_;
    size_t pos_ = 0;
    mutable std::mt19937 rng_{std::random_device{}()};

public:
    ReplayBuffer(size_t cap = 200000) : capacity_(cap) { buffer_.reserve(cap); }

    void push(const Experience& e) {
        if (buffer_.size() < capacity_)
            buffer_.push_back(e);
        else
            buffer_[pos_ % capacity_] = e;
        pos_++;
    }

    void sample(std::vector<Experience>& batch, size_t n) const {
        batch.clear();
        size_t sz = buffer_.size();
        if (sz == 0) return;
        std::uniform_int_distribution<size_t> dist(0, sz - 1);
        for (size_t i = 0; i < n && i < sz; ++i)
            batch.push_back(buffer_[dist(rng_)]);
    }

    // 随机打乱后取前 n 条
    void sample_shuffled(std::vector<Experience>& batch, size_t n) const {
        batch.clear();
        size_t sz = buffer_.size();
        if (sz == 0 || n == 0) return;
        n = std::min(n, sz);
        // 随机采样 n 个不同索引
        std::vector<size_t> indices(sz);
        for (size_t i = 0; i < sz; ++i) indices[i] = i;
        for (size_t i = 0; i < n; ++i) {
            size_t j = i + (rng_() % (sz - i));
            std::swap(indices[i], indices[j]);
        }
        for (size_t i = 0; i < n; ++i)
            batch.push_back(buffer_[indices[i]]);
    }

    const std::vector<Experience>& all() const { return buffer_; }
    size_t size() const { return buffer_.size(); }

    bool save(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) return false;
        size_t n = buffer_.size();
        ofs.write((char*)&n, sizeof(n));
        for (auto& e : buffer_) {
            ofs.write((char*)e.features.f, sizeof(double) * NUM_FEATURES);
            ofs.write((char*)&e.value, sizeof(e.value));
            ofs.write((char*)&e.step, sizeof(e.step));
            ofs.write((char*)&e.level, sizeof(e.level));
        }
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        size_t n;
        ifs.read((char*)&n, sizeof(n));
        buffer_.clear();
        buffer_.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            Experience e;
            ifs.read((char*)e.features.f, sizeof(double) * NUM_FEATURES);
            ifs.read((char*)&e.value, sizeof(e.value));
            ifs.read((char*)&e.step, sizeof(e.step));
            ifs.read((char*)&e.level, sizeof(e.level));
            buffer_.push_back(e);
        }
        pos_ = n;
        return true;
    }
};

#endif
