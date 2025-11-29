#include "Optimizer.h"

#include <cmath>

void optimizerResetState(
    std::vector<float>& mW1, std::vector<float>& mb1,
    std::vector<float>& mW2, std::vector<float>& mb2,
    std::vector<float>& mW3, std::vector<float>& mb3,
    std::vector<float>& vW1, std::vector<float>& vb1,
    std::vector<float>& vW2, std::vector<float>& vb2,
    std::vector<float>& vW3, std::vector<float>& vb3,
    int&                adamStep)
{
    std::fill(mW1.begin(), mW1.end(), 0.0f);
    std::fill(mb1.begin(), mb1.end(), 0.0f);
    std::fill(mW2.begin(), mW2.end(), 0.0f);
    std::fill(mb2.begin(), mb2.end(), 0.0f);
    std::fill(mW3.begin(), mW3.end(), 0.0f);
    std::fill(mb3.begin(), mb3.end(), 0.0f);

    std::fill(vW1.begin(), vW1.end(), 0.0f);
    std::fill(vb1.begin(), vb1.end(), 0.0f);
    std::fill(vW2.begin(), vW2.end(), 0.0f);
    std::fill(vb2.begin(), vb2.end(), 0.0f);
    std::fill(vW3.begin(), vW3.end(), 0.0f);
    std::fill(vb3.begin(), vb3.end(), 0.0f);

    adamStep = 0;
}

void optimizerApplyUpdate(
    const OptimizerConfig& cfg,
    std::vector<float>&    W1,  std::vector<float>&    b1,
    std::vector<float>&    W2,  std::vector<float>&    b2,
    std::vector<float>&    W3,  std::vector<float>&    b3,
    const std::vector<float>& dW1, const std::vector<float>& db1,
    const std::vector<float>& dW2, const std::vector<float>& db2,
    const std::vector<float>& dW3, const std::vector<float>& db3,
    std::vector<float>&       mW1, std::vector<float>&       mb1,
    std::vector<float>&       mW2, std::vector<float>&       mb2,
    std::vector<float>&       mW3, std::vector<float>&       mb3,
    std::vector<float>&       vW1, std::vector<float>&       vb1,
    std::vector<float>&       vW2, std::vector<float>&       vb2,
    std::vector<float>&       vW3, std::vector<float>&       vb3,
    int&                      adamStep)
{
    if (cfg.type == OptimizerType::SGD) {
        // Plain SGD: param -= lr * grad
        const float lr = cfg.learningRate;
        for (std::size_t i = 0; i < W1.size(); ++i) W1[i] -= lr * dW1[i];
        for (std::size_t i = 0; i < b1.size(); ++i) b1[i] -= lr * db1[i];
        for (std::size_t i = 0; i < W2.size(); ++i) W2[i] -= lr * dW2[i];
        for (std::size_t i = 0; i < b2.size(); ++i) b2[i] -= lr * db2[i];
        for (std::size_t i = 0; i < W3.size(); ++i) W3[i] -= lr * dW3[i];
        for (std::size_t i = 0; i < b3.size(); ++i) b3[i] -= lr * db3[i];
        return;
    }

    if (cfg.type == OptimizerType::SGDMomentum) {
        // SGD with momentum: v = mu * v - lr * grad; param += v
        const float lr = cfg.learningRate;
        const float mu = cfg.momentum;

        for (std::size_t i = 0; i < W1.size(); ++i) {
            mW1[i] = mu * mW1[i] - lr * dW1[i];
            W1[i] += mW1[i];
        }
        for (std::size_t i = 0; i < b1.size(); ++i) {
            mb1[i] = mu * mb1[i] - lr * db1[i];
            b1[i] += mb1[i];
        }
        for (std::size_t i = 0; i < W2.size(); ++i) {
            mW2[i] = mu * mW2[i] - lr * dW2[i];
            W2[i] += mW2[i];
        }
        for (std::size_t i = 0; i < b2.size(); ++i) {
            mb2[i] = mu * mb2[i] - lr * db2[i];
            b2[i] += mb2[i];
        }
        for (std::size_t i = 0; i < W3.size(); ++i) {
            mW3[i] = mu * mW3[i] - lr * dW3[i];
            W3[i] += mW3[i];
        }
        for (std::size_t i = 0; i < b3.size(); ++i) {
            mb3[i] = mu * mb3[i] - lr * db3[i];
            b3[i] += mb3[i];
        }
        return;
    }

    if (cfg.type == OptimizerType::Adam) {
        // Adam: maintain first (m) and second (v) moments with bias correction
        const float lr     = cfg.learningRate;
        const float beta1  = cfg.beta1;
        const float beta2  = cfg.beta2;
        const float eps    = cfg.eps;

        ++adamStep;
        const float t         = static_cast<float>(adamStep);
        const float biasCorr1 = 1.0f - std::pow(beta1, t);
        const float biasCorr2 = 1.0f - std::pow(beta2, t);

        for (std::size_t i = 0; i < W1.size(); ++i) {
            float g = dW1[i];
            mW1[i] = beta1 * mW1[i] + (1.0f - beta1) * g;
            vW1[i] = beta2 * vW1[i] + (1.0f - beta2) * g * g;
            float mHat = mW1[i] / biasCorr1;
            float vHat = vW1[i] / biasCorr2;
            W1[i] -= lr * mHat / (std::sqrt(vHat) + eps);
        }
        for (std::size_t i = 0; i < b1.size(); ++i) {
            float g = db1[i];
            mb1[i] = beta1 * mb1[i] + (1.0f - beta1) * g;
            vb1[i] = beta2 * vb1[i] + (1.0f - beta2) * g * g;
            float mHat = mb1[i] / biasCorr1;
            float vHat = vb1[i] / biasCorr2;
            b1[i] -= lr * mHat / (std::sqrt(vHat) + eps);
        }
        for (std::size_t i = 0; i < W2.size(); ++i) {
            float g = dW2[i];
            mW2[i] = beta1 * mW2[i] + (1.0f - beta1) * g;
            vW2[i] = beta2 * vW2[i] + (1.0f - beta2) * g * g;
            float mHat = mW2[i] / biasCorr1;
            float vHat = vW2[i] / biasCorr2;
            W2[i] -= lr * mHat / (std::sqrt(vHat) + eps);
        }
        for (std::size_t i = 0; i < b2.size(); ++i) {
            float g = db2[i];
            mb2[i] = beta1 * mb2[i] + (1.0f - beta1) * g;
            vb2[i] = beta2 * vb2[i] + (1.0f - beta2) * g * g;
            float mHat = mb2[i] / biasCorr1;
            float vHat = vb2[i] / biasCorr2;
            b2[i] -= lr * mHat / (std::sqrt(vHat) + eps);
        }
        for (std::size_t i = 0; i < W3.size(); ++i) {
            float g = dW3[i];
            mW3[i] = beta1 * mW3[i] + (1.0f - beta1) * g;
            vW3[i] = beta2 * vW3[i] + (1.0f - beta2) * g * g;
            float mHat = mW3[i] / biasCorr1;
            float vHat = vW3[i] / biasCorr2;
            W3[i] -= lr * mHat / (std::sqrt(vHat) + eps);
        }
        for (std::size_t i = 0; i < b3.size(); ++i) {
            float g = db3[i];
            mb3[i] = beta1 * mb3[i] + (1.0f - beta1) * g;
            vb3[i] = beta2 * vb3[i] + (1.0f - beta2) * g * g;
            float mHat = mb3[i] / biasCorr1;
            float vHat = vb3[i] / biasCorr2;
            b3[i] -= lr * mHat / (std::sqrt(vHat) + eps);
        }

        return;
    }
}
