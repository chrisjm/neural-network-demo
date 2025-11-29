#include "ToyNet.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace {

inline int idx(int row, int col, int cols) {
    return row * cols + col;
}

inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

}

ToyNet::ToyNet() {
    hidden1 = 4;
    hidden2 = 4;
    setHiddenSizes(hidden1, hidden2);
    resetParameters(1);
}

void ToyNet::resetParameters(unsigned int seed) {
    std::srand(seed);

    auto randUniform = []() {
        return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    };

    const float scale1 = 1.0f / std::sqrt(static_cast<float>(InputDim));
    const float scale2 = 1.0f / std::sqrt(static_cast<float>(hidden1));
    const float scale3 = 1.0f / std::sqrt(static_cast<float>(hidden2));

    for (auto& w : W1) w = scale1 * randUniform();
    for (auto& w : W2) w = scale2 * randUniform();
    for (auto& w : W3) w = scale3 * randUniform();

    std::fill(b1.begin(), b1.end(), 0.0f);
    std::fill(b2.begin(), b2.end(), 0.0f);
    std::fill(b3.begin(), b3.end(), 0.0f);
}

void ToyNet::setHiddenSizes(int h1, int h2) {
    if (h1 < MinHidden) h1 = MinHidden;
    if (h1 > MaxHidden) h1 = MaxHidden;
    if (h2 < MinHidden) h2 = MinHidden;
    if (h2 > MaxHidden) h2 = MaxHidden;

    hidden1 = h1;
    hidden2 = h2;

    W1.resize(hidden1 * InputDim);
    b1.resize(hidden1);
    W2.resize(hidden2 * hidden1);
    b2.resize(hidden2);
    W3.resize(OutputDim * hidden2);
    b3.resize(OutputDim);

    a0.resize(MaxBatch * InputDim);
    z1.resize(MaxBatch * hidden1);
    a1.resize(MaxBatch * hidden1);
    z2.resize(MaxBatch * hidden2);
    a2.resize(MaxBatch * hidden2);
    logits.resize(MaxBatch * OutputDim);
    probs.resize(MaxBatch * OutputDim);

    dW1.resize(W1.size());
    db1.resize(b1.size());
    dW2.resize(W2.size());
    db2.resize(b2.size());
    dW3.resize(W3.size());
    db3.resize(b3.size());
}

float ToyNet::trainBatch(const std::vector<DataPoint>& batch, float& outAccuracy) {
    const int N = static_cast<int>(batch.size());
    if (N <= 0) {
        outAccuracy = 0.0f;
        return 0.0f;
    }
    const int batchSize = std::min(N, MaxBatch);

    // Copy inputs
    for (int n = 0; n < batchSize; ++n) {
        a0[idx(n, 0, InputDim)] = batch[n].x;
        a0[idx(n, 1, InputDim)] = batch[n].y;
    }

    for (int n = 0; n < batchSize; ++n) {
        for (int j = 0; j < hidden1; ++j) {
            float sum = b1[j];
            for (int i = 0; i < InputDim; ++i) {
                sum += W1[idx(j, i, InputDim)] * a0[idx(n, i, InputDim)];
            }
            const int zIndex = idx(n, j, hidden1);
            z1[zIndex] = sum;
            a1[zIndex] = relu(sum);
        }
    }

    for (int n = 0; n < batchSize; ++n) {
        for (int j = 0; j < hidden2; ++j) {
            float sum = b2[j];
            for (int i = 0; i < hidden1; ++i) {
                sum += W2[idx(j, i, hidden1)] * a1[idx(n, i, hidden1)];
            }
            const int zIndex = idx(n, j, hidden2);
            z2[zIndex] = sum;
            a2[zIndex] = relu(sum);
        }
    }

    // Forward pass: output layer (logits + softmax)
    float lossSum   = 0.0f;
    int   correct   = 0;

    for (int n = 0; n < batchSize; ++n) {
        // logits
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < OutputDim; ++k) {
            float sum = b3[k];
            for (int j = 0; j < hidden2; ++j) {
                sum += W3[idx(k, j, hidden2)] * a2[idx(n, j, hidden2)];
            }
            const int lIndex = idx(n, k, OutputDim);
            logits[lIndex] = sum;
            if (sum > maxLogit) maxLogit = sum;
        }

        // softmax
        float expSum = 0.0f;
        for (int k = 0; k < OutputDim; ++k) {
            const int lIndex = idx(n, k, OutputDim);
            float e = std::exp(logits[lIndex] - maxLogit);
            probs[lIndex] = e;
            expSum += e;
        }

        int   label      = batch[n].label;
        int   predicted  = 0;
        float bestProb   = -1.0f;
        float correctProb = 0.0f;
        for (int k = 0; k < OutputDim; ++k) {
            const int pIndex = idx(n, k, OutputDim);
            probs[pIndex] /= expSum;
            if (probs[pIndex] > bestProb) {
                bestProb  = probs[pIndex];
                predicted = k;
            }
            if (k == label) {
                correctProb = probs[pIndex];
            }
        }

        if (predicted == label) {
            ++correct;
        }

        const float eps = 1e-6f;
        lossSum += -std::log(std::max(correctProb, eps));
    }

    const float invN = 1.0f / static_cast<float>(batchSize);
    float loss = lossSum * invN;
    outAccuracy = static_cast<float>(correct) * invN;

    // Zero gradients
    std::fill(dW1.begin(), dW1.end(), 0.0f);
    std::fill(db1.begin(), db1.end(), 0.0f);
    std::fill(dW2.begin(), dW2.end(), 0.0f);
    std::fill(db2.begin(), db2.end(), 0.0f);
    std::fill(dW3.begin(), dW3.end(), 0.0f);
    std::fill(db3.begin(), db3.end(), 0.0f);

    // Backward pass
    for (int n = 0; n < batchSize; ++n) {
        int label = batch[n].label;

        float delta3[OutputDim] = {0.0f, 0.0f};
        for (int k = 0; k < OutputDim; ++k) {
            const int pIndex = idx(n, k, OutputDim);
            float yk = (k == label) ? 1.0f : 0.0f;
            delta3[k] = probs[pIndex] - yk;
        }

        float delta2Raw[MaxHidden];
        for (int j = 0; j < hidden2; ++j) {
            delta2Raw[j] = 0.0f;
        }

        // Gradients for W3, b3 and delta2Raw
        for (int k = 0; k < OutputDim; ++k) {
            for (int j = 0; j < hidden2; ++j) {
                dW3[idx(k, j, hidden2)] += delta3[k] * a2[idx(n, j, hidden2)];
                delta2Raw[j] += delta3[k] * W3[idx(k, j, hidden2)];
            }
            db3[k] += delta3[k];
        }

        float delta2[MaxHidden];
        for (int j = 0; j < hidden2; ++j) {
            float z = z2[idx(n, j, hidden2)];
            delta2[j] = (z > 0.0f) ? delta2Raw[j] : 0.0f;
        }

        float delta1Raw[MaxHidden];
        for (int i = 0; i < hidden1; ++i) {
            delta1Raw[i] = 0.0f;
        }

        // Gradients for W2, b2 and delta1Raw
        for (int j = 0; j < hidden2; ++j) {
            for (int i = 0; i < hidden1; ++i) {
                dW2[idx(j, i, hidden1)] += delta2[j] * a1[idx(n, i, hidden1)];
                delta1Raw[i] += delta2[j] * W2[idx(j, i, hidden1)];
            }
            db2[j] += delta2[j];
        }

        float delta1[MaxHidden];
        for (int i = 0; i < hidden1; ++i) {
            float z = z1[idx(n, i, hidden1)];
            delta1[i] = (z > 0.0f) ? delta1Raw[i] : 0.0f;
        }

        // Gradients for W1, b1
        for (int i = 0; i < hidden1; ++i) {
            for (int d = 0; d < InputDim; ++d) {
                dW1[idx(i, d, InputDim)] += delta1[i] * a0[idx(n, d, InputDim)];
            }
            db1[i] += delta1[i];
        }
    }

    // Average gradients over batch
    for (auto& g : dW1) g *= invN;
    for (auto& g : db1) g *= invN;
    for (auto& g : dW2) g *= invN;
    for (auto& g : db2) g *= invN;
    for (auto& g : dW3) g *= invN;
    for (auto& g : db3) g *= invN;

    // Gradient descent update
    for (std::size_t i = 0; i < W1.size(); ++i) W1[i] -= learningRate * dW1[i];
    for (std::size_t i = 0; i < b1.size(); ++i) b1[i] -= learningRate * db1[i];
    for (std::size_t i = 0; i < W2.size(); ++i) W2[i] -= learningRate * dW2[i];
    for (std::size_t i = 0; i < b2.size(); ++i) b2[i] -= learningRate * db2[i];
    for (std::size_t i = 0; i < W3.size(); ++i) W3[i] -= learningRate * dW3[i];
    for (std::size_t i = 0; i < b3.size(); ++i) b3[i] -= learningRate * db3[i];

    return loss;
}

void ToyNet::forwardSingle(float x, float y, float& p0, float& p1) const {
    float a_in[InputDim] = {x, y};
    float a_h1[MaxHidden];
    float a_h2[MaxHidden];
    float logitsLocal[OutputDim];

    for (int j = 0; j < hidden1; ++j) {
        float sum = b1[j];
        for (int i = 0; i < InputDim; ++i) {
            sum += W1[idx(j, i, InputDim)] * a_in[i];
        }
        a_h1[j] = relu(sum);
    }

    for (int j = 0; j < hidden2; ++j) {
        float sum = b2[j];
        for (int i = 0; i < hidden1; ++i) {
            sum += W2[idx(j, i, hidden1)] * a_h1[i];
        }
        a_h2[j] = relu(sum);
    }

    float maxLogit = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < OutputDim; ++k) {
        float sum = b3[k];
        for (int j = 0; j < hidden2; ++j) {
            sum += W3[idx(k, j, hidden2)] * a_h2[j];
        }
        logitsLocal[k] = sum;
        if (sum > maxLogit) maxLogit = sum;
    }

    float expSum = 0.0f;
    float probsLocal[OutputDim];
    for (int k = 0; k < OutputDim; ++k) {
        float e = std::exp(logitsLocal[k] - maxLogit);
        probsLocal[k] = e;
        expSum += e;
    }

    if (expSum <= 0.0f) {
        p0 = 0.5f;
        p1 = 0.5f;
        return;
    }

    probsLocal[0] /= expSum;
    probsLocal[1] /= expSum;

    p0 = probsLocal[0];
    p1 = probsLocal[1];
}
