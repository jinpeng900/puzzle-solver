#ifndef NN_MODEL_H
#define NN_MODEL_H

#include "board_features.h"
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <cstdint>

// ============================================================
// 轻量级 MLP 神经网络（3层：52→64→32→1）
// 内置 Adam 优化器，纯 C++ 实现，零外部依赖
// ============================================================
class NNModel {
public:
    struct Layer {
        std::vector<std::vector<double>> W; // [out][in]
        std::vector<double> b;              // [out]
        mutable std::vector<double> z;      // pre-activation (mutable for const predict)
        mutable std::vector<double> a;      // post-activation
        int in_dim, out_dim;
        bool use_relu;

        Layer(int in, int out, bool relu = true) : in_dim(in), out_dim(out), use_relu(relu) {
            W.assign(out, std::vector<double>(in, 0));
            b.assign(out, 0);
            z.assign(out, 0);
            a.assign(out, 0);
            init_xavier();
        }

        void init_xavier() {
            static std::mt19937 rng(114514);
            double scale = std::sqrt(2.0 / std::max(1.0, (double)(in_dim + out_dim)));
            std::normal_distribution<double> dist(0, scale);
            for (int i = 0; i < out_dim; ++i) {
                for (int j = 0; j < in_dim; ++j) W[i][j] = dist(rng);
                b[i] = 0;
            }
        }

        void forward(const std::vector<double>& input) {
            for (int i = 0; i < out_dim; ++i) {
                double sum = b[i];
                const auto& wr = W[i];
                for (int j = 0; j < in_dim; ++j) sum += wr[j] * input[j];
                if (sum != sum) sum = 0;        // NaN guard
                if (sum > 50.0) sum = 50.0;     // clip explosion
                if (sum < -50.0) sum = -50.0;
                z[i] = sum;
                a[i] = use_relu ? std::max(0.0, sum) : std::tanh(sum);
            }
        }

        // 返回输出引用
        const std::vector<double>& output() const { return a; }
    };

private:
    std::vector<Layer> layers_;
    double scale_factor_ = 1000.0;  // [−1,1] → [−1000,1000]
    double learning_rate_ = 0.001;
    int train_count_ = 0;

    // Adam state
    struct Adam {
        std::vector<std::vector<double>> mW, vW;
        std::vector<double> mb, vb;
        double beta1_t = 1.0, beta2_t = 1.0;
    };
    std::vector<Adam> adam_;

public:
    NNModel() {
        build({64, 32});
    }

    NNModel(const std::vector<int>& hidden) {
        build(hidden);
    }

    void build(const std::vector<int>& hidden) {
        layers_.clear();
        adam_.clear();
        int prev = NUM_FEATURES;
        for (int sz : hidden) {
            layers_.emplace_back(prev, sz, true);
            prev = sz;
        }
        layers_.emplace_back(prev, 1, false);  // tanh output

        for (auto& L : layers_) {
            Adam a;
            a.mW.assign(L.out_dim, std::vector<double>(L.in_dim, 0));
            a.vW.assign(L.out_dim, std::vector<double>(L.in_dim, 0));
            a.mb.assign(L.out_dim, 0);
            a.vb.assign(L.out_dim, 0);
            adam_.push_back(a);
        }
    }

    // 推理：board_features → 评估值
    double predict(const BoardFeatures& feat) {
        std::vector<double> input(feat.f, feat.f + NUM_FEATURES);
        for (auto& L : layers_) {
            L.forward(input);
            input = L.output();
        }
        return input[0] * scale_factor_;
    }

    // 获取原始网络输出（未缩放）
    double predict_raw(const BoardFeatures& feat) {
        std::vector<double> input(feat.f, feat.f + NUM_FEATURES);
        for (auto& L : layers_) {
            L.forward(input);
            input = L.output();
        }
        return input[0];
    }

    // 单样本 SGD 训练（带 Adam）
    void train(const BoardFeatures& feat, double target) {
        const double beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
        int L = (int)layers_.size();

        // 归一化目标
        double norm_target = target / std::max(1.0, scale_factor_);
        if (norm_target > 1.0) norm_target = 1.0;
        if (norm_target < -1.0) norm_target = -1.0;

        // 前向传播，收集激活值
        std::vector<std::vector<double>> activations(L + 1);
        activations[0].assign(feat.f, feat.f + NUM_FEATURES);
        for (int l = 0; l < L; ++l) {
            layers_[l].forward(activations[l]);
            activations[l + 1] = layers_[l].output();
        }

        double predicted = activations[L][0];
        double error = predicted - norm_target;

        // 输出层 delta
        std::vector<double> delta = { 2.0 * error };
        if (!layers_.back().use_relu) {
            double y = layers_.back().a[0];
            delta[0] *= (1.0 - y * y);  // tanh derivative
        }

        // 反向传播 + Adam 更新
        for (int l = L - 1; l >= 0; --l) {
            int od = layers_[l].out_dim;
            int id = layers_[l].in_dim;
            std::vector<double> prev_delta(id, 0);

            adam_[l].beta1_t *= beta1;
            adam_[l].beta2_t *= beta2;
            double b1t = adam_[l].beta1_t;
            double b2t = adam_[l].beta2_t;

            for (int i = 0; i < od; ++i) {
                double d_act;
                if (layers_[l].use_relu)
                    d_act = (layers_[l].z[i] > 0) ? delta[i] : 0;
                else
                    d_act = delta[i];

                // 累积 prev_delta
                for (int j = 0; j < id; ++j)
                    prev_delta[j] += d_act * layers_[l].W[i][j];

                // Adam 更新 bias
                adam_[l].mb[i] = beta1 * adam_[l].mb[i] + (1 - beta1) * d_act;
                adam_[l].vb[i] = beta2 * adam_[l].vb[i] + (1 - beta2) * d_act * d_act;
                double mh = adam_[l].mb[i] / (1 - b1t);
                double vh = adam_[l].vb[i] / (1 - b2t);
                layers_[l].b[i] -= learning_rate_ * mh / (std::sqrt(vh) + eps);

                // Adam 更新 weights
                for (int j = 0; j < id; ++j) {
                    double gw = d_act * activations[l][j];
                    adam_[l].mW[i][j] = beta1 * adam_[l].mW[i][j] + (1 - beta1) * gw;
                    adam_[l].vW[i][j] = beta2 * adam_[l].vW[i][j] + (1 - beta2) * gw * gw;
                    double mwh = adam_[l].mW[i][j] / (1 - b1t);
                    double vwh = adam_[l].vW[i][j] / (1 - b2t);
                    layers_[l].W[i][j] -= learning_rate_ * mwh / (std::sqrt(vwh) + eps);
                }
            }
            delta = prev_delta;
        }

        train_count_++;
        if (train_count_ % 10000 == 0) {
            learning_rate_ *= 0.95;
            if (learning_rate_ < 0.00001) learning_rate_ = 0.00001;
        }
    }

    // 批量训练（epochs 轮）
    void train_batch(const std::vector<BoardFeatures>& feats,
                     const std::vector<double>& targets,
                     int epochs) {
        for (int e = 0; e < epochs; ++e) {
            double total_loss = 0;
            for (size_t i = 0; i < feats.size(); ++i) {
                train(feats[i], targets[i]);
                double pred = predict(feats[i]);
                double err = pred - targets[i];
                total_loss += err * err;
            }
            // 每个 epoch 打印一次 loss（可选）
        }
    }

    // 保存/加载权重
    bool save(const std::string& path) const {
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs) return false;
        ofs.write((char*)&scale_factor_, sizeof(scale_factor_));
        int nl = (int)layers_.size();
        ofs.write((char*)&nl, sizeof(nl));
        for (auto& L : layers_) {
            ofs.write((char*)&L.in_dim, sizeof(L.in_dim));
            ofs.write((char*)&L.out_dim, sizeof(L.out_dim));
            ofs.write((char*)&L.use_relu, sizeof(L.use_relu));
            for (auto& row : L.W)
                ofs.write((char*)row.data(), sizeof(double) * L.in_dim);
            ofs.write((char*)L.b.data(), sizeof(double) * L.out_dim);
        }
        return true;
    }

    bool load(const std::string& path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) return false;
        ifs.read((char*)&scale_factor_, sizeof(scale_factor_));
        int nl;
        ifs.read((char*)&nl, sizeof(nl));
        layers_.clear();
        adam_.clear();
        for (int l = 0; l < nl; ++l) {
            int in_dim, out_dim;
            bool relu;
            ifs.read((char*)&in_dim, sizeof(in_dim));
            ifs.read((char*)&out_dim, sizeof(out_dim));
            ifs.read((char*)&relu, sizeof(relu));
            Layer layer(in_dim, out_dim, relu);
            for (auto& row : layer.W) ifs.read((char*)row.data(), sizeof(double)*in_dim);
            ifs.read((char*)layer.b.data(), sizeof(double)*out_dim);
            layers_.push_back(std::move(layer));
        }
        for (auto& L : layers_) {
            Adam a;
            a.mW.assign(L.out_dim, std::vector<double>(L.in_dim, 0));
            a.vW.assign(L.out_dim, std::vector<double>(L.in_dim, 0));
            a.mb.assign(L.out_dim, 0);
            a.vb.assign(L.out_dim, 0);
            adam_.push_back(a);
        }
        return true;
    }

    void set_lr(double lr) { learning_rate_ = lr; }
    void set_scale(double s) { scale_factor_ = s; }
    int num_layers() const { return (int)layers_.size(); }
};

#endif
