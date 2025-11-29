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

ToyNet::ToyNet()
    : m_learningRate(0.1f) {
    m_W1.resize(Hidden1 * InputDim);
    m_b1.resize(Hidden1);
    m_W2.resize(Hidden2 * Hidden1);
    m_b2.resize(Hidden2);
    m_W3.resize(OutputDim * Hidden2);
    m_b3.resize(OutputDim);

    m_a0.resize(MaxBatch * InputDim);
    m_z1.resize(MaxBatch * Hidden1);
    m_a1.resize(MaxBatch * Hidden1);
    m_z2.resize(MaxBatch * Hidden2);
    m_a2.resize(MaxBatch * Hidden2);
    m_logits.resize(MaxBatch * OutputDim);
    m_probs.resize(MaxBatch * OutputDim);

    m_dW1.resize(m_W1.size());
    m_db1.resize(m_b1.size());
    m_dW2.resize(m_W2.size());
    m_db2.resize(m_b2.size());
    m_dW3.resize(m_W3.size());
    m_db3.resize(m_b3.size());

    resetParameters(1);
}

void ToyNet::resetParameters(unsigned int seed) {
    std::srand(seed);

    auto randUniform = []() {
        return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
    };

    const float scale1 = 1.0f / std::sqrt(static_cast<float>(InputDim));
    const float scale2 = 1.0f / std::sqrt(static_cast<float>(Hidden1));
    const float scale3 = 1.0f / std::sqrt(static_cast<float>(Hidden2));

    for (auto& w : m_W1) w = scale1 * randUniform();
    for (auto& w : m_W2) w = scale2 * randUniform();
    for (auto& w : m_W3) w = scale3 * randUniform();

    std::fill(m_b1.begin(), m_b1.end(), 0.0f);
    std::fill(m_b2.begin(), m_b2.end(), 0.0f);
    std::fill(m_b3.begin(), m_b3.end(), 0.0f);
}

float ToyNet::trainBatch(const std::vector<DataPoint>& batch, float& outAccuracy) {
    const int N = static_cast<int>(batch.size());
    if (N <= 0) {
        outAccuracy = 0.0f;
        return 0.0f;
    }
    const int batchSize = std::min(N, MaxBatch);

    // Copy inputs into a0 (the input activations for the batch)
    for (int n = 0; n < batchSize; ++n) {
        m_a0[idx(n, 0, InputDim)] = batch[n].x;
        m_a0[idx(n, 1, InputDim)] = batch[n].y;
    }

    // Forward pass: layer 1 (ReLU(Input * W1 + b1))
    for (int n = 0; n < batchSize; ++n) {
        for (int j = 0; j < Hidden1; ++j) {
            float sum = m_b1[j];
            for (int i = 0; i < InputDim; ++i) {
                sum += m_W1[idx(j, i, InputDim)] * m_a0[idx(n, i, InputDim)];
            }
            const int zIndex = idx(n, j, Hidden1);
            m_z1[zIndex] = sum;
            m_a1[zIndex] = relu(sum);
        }
    }

    // Forward pass: layer 2 (ReLU(a1 * W2 + b2))
    for (int n = 0; n < batchSize; ++n) {
        for (int j = 0; j < Hidden2; ++j) {
            float sum = m_b2[j];
            for (int i = 0; i < Hidden1; ++i) {
                sum += m_W2[idx(j, i, Hidden1)] * m_a1[idx(n, i, Hidden1)];
            }
            const int zIndex = idx(n, j, Hidden2);
            m_z2[zIndex] = sum;
            m_a2[zIndex] = relu(sum);
        }
    }

    // Forward pass: output layer (logits + softmax)
    float lossSum   = 0.0f;
    int   correct   = 0;

    for (int n = 0; n < batchSize; ++n) {
        // logits: z3 = a2 * W3 + b3
        float maxLogit = -std::numeric_limits<float>::infinity();
        for (int k = 0; k < OutputDim; ++k) {
            float sum = m_b3[k];
            for (int j = 0; j < Hidden2; ++j) {
                sum += m_W3[idx(k, j, Hidden2)] * m_a2[idx(n, j, Hidden2)];
            }
            const int lIndex = idx(n, k, OutputDim);
            m_logits[lIndex] = sum;
            if (sum > maxLogit) maxLogit = sum;
        }

        // softmax: p_k = exp(z3_k) / sum_j exp(z3_j)
        float expSum = 0.0f;
        for (int k = 0; k < OutputDim; ++k) {
            const int lIndex = idx(n, k, OutputDim);
            float e = std::exp(m_logits[lIndex] - maxLogit);
            m_probs[lIndex] = e;
            expSum += e;
        }

        int   label      = batch[n].label;
        int   predicted  = 0;
        float bestProb   = -1.0f;
        float correctProb = 0.0f;
        for (int k = 0; k < OutputDim; ++k) {
            const int pIndex = idx(n, k, OutputDim);
            m_probs[pIndex] /= expSum;
            if (m_probs[pIndex] > bestProb) {
                bestProb  = m_probs[pIndex];
                predicted = k;
            }
            if (k == label) {
                correctProb = m_probs[pIndex];
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
    std::fill(m_dW1.begin(), m_dW1.end(), 0.0f);
    std::fill(m_db1.begin(), m_db1.end(), 0.0f);
    std::fill(m_dW2.begin(), m_dW2.end(), 0.0f);
    std::fill(m_db2.begin(), m_db2.end(), 0.0f);
    std::fill(m_dW3.begin(), m_dW3.end(), 0.0f);
    std::fill(m_db3.begin(), m_db3.end(), 0.0f);

    // Backward pass
    // We use cross-entropy loss with softmax, so dL/dz3 = (p - y).
    // For ReLU, dL/dz = dL/da * 1(z > 0).
    for (int n = 0; n < batchSize; ++n) {
        int label = batch[n].label;

        // delta3_k = dL/dz3_k = p_k - y_k
        float delta3[OutputDim] = {0.0f, 0.0f};
        for (int k = 0; k < OutputDim; ++k) {
            const int pIndex = idx(n, k, OutputDim);
            float yk = (k == label) ? 1.0f : 0.0f;
            delta3[k] = m_probs[pIndex] - yk;
        }

        float delta2Raw[Hidden2];
        for (int j = 0; j < Hidden2; ++j) {
            delta2Raw[j] = 0.0f;
        }

        // Gradients for W3, b3 and delta2Raw.
        // dL/dW3_{k,j} += delta3_k * a2_j, and
        // delta2Raw_j = sum_k delta3_k * W3_{k,j}.
        for (int k = 0; k < OutputDim; ++k) {
            for (int j = 0; j < Hidden2; ++j) {
                m_dW3[idx(k, j, Hidden2)] += delta3[k] * m_a2[idx(n, j, Hidden2)];
                delta2Raw[j] += delta3[k] * m_W3[idx(k, j, Hidden2)];
            }
            m_db3[k] += delta3[k];
        }

        // Apply ReLU derivative at layer 2: delta2_j = delta2Raw_j * 1(z2_j > 0).
        float delta2[Hidden2];
        for (int j = 0; j < Hidden2; ++j) {
            float z = m_z2[idx(n, j, Hidden2)];
            delta2[j] = (z > 0.0f) ? delta2Raw[j] : 0.0f;
        }

        float delta1Raw[Hidden1];
        for (int i = 0; i < Hidden1; ++i) {
            delta1Raw[i] = 0.0f;
        }

        // Gradients for W2, b2 and delta1Raw.
        // dL/dW2_{j,i} += delta2_j * a1_i, and
        // delta1Raw_i = sum_j delta2_j * W2_{j,i}.
        for (int j = 0; j < Hidden2; ++j) {
            for (int i = 0; i < Hidden1; ++i) {
                m_dW2[idx(j, i, Hidden1)] += delta2[j] * m_a1[idx(n, i, Hidden1)];
                delta1Raw[i] += delta2[j] * m_W2[idx(j, i, Hidden1)];
            }
            m_db2[j] += delta2[j];
        }

        // Apply ReLU derivative at layer 1: delta1_i = delta1Raw_i * 1(z1_i > 0).
        float delta1[Hidden1];
        for (int i = 0; i < Hidden1; ++i) {
            float z = m_z1[idx(n, i, Hidden1)];
            delta1[i] = (z > 0.0f) ? delta1Raw[i] : 0.0f;
        }

        // Gradients for W1, b1.
        // dL/dW1_{i,d} += delta1_i * a0_d.
        for (int i = 0; i < Hidden1; ++i) {
            for (int d = 0; d < InputDim; ++d) {
                m_dW1[idx(i, d, InputDim)] += delta1[i] * m_a0[idx(n, d, InputDim)];
            }
            m_db1[i] += delta1[i];
        }
    }

    // Average gradients over batch
    for (auto& g : m_dW1) g *= invN;
    for (auto& g : m_db1) g *= invN;
    for (auto& g : m_dW2) g *= invN;
    for (auto& g : m_db2) g *= invN;
    for (auto& g : m_dW3) g *= invN;
    for (auto& g : m_db3) g *= invN;

    // Gradient descent update
    for (std::size_t i = 0; i < m_W1.size(); ++i) m_W1[i] -= m_learningRate * m_dW1[i];
    for (std::size_t i = 0; i < m_b1.size(); ++i) m_b1[i] -= m_learningRate * m_db1[i];
    for (std::size_t i = 0; i < m_W2.size(); ++i) m_W2[i] -= m_learningRate * m_dW2[i];
    for (std::size_t i = 0; i < m_b2.size(); ++i) m_b2[i] -= m_learningRate * m_db2[i];
    for (std::size_t i = 0; i < m_W3.size(); ++i) m_W3[i] -= m_learningRate * m_dW3[i];
    for (std::size_t i = 0; i < m_b3.size(); ++i) m_b3[i] -= m_learningRate * m_db3[i];

    return loss;
}

void ToyNet::forwardSingle(float x, float y, float& p0, float& p1) const {
    float a1[Hidden1];
    float a2[Hidden2];
    forwardSingleWithActivations(x, y, p0, p1, a1, a2);
}

void ToyNet::forwardSingleWithActivations(float x, float y,
                                          float& p0, float& p1,
                                          float* outA1,
                                          float* outA2) const {
    float a_in[InputDim] = {x, y};
    float a_h1[Hidden1];
    float a_h2[Hidden2];
    float logitsLocal[OutputDim];

    for (int j = 0; j < Hidden1; ++j) {
        float sum = m_b1[j];
        for (int i = 0; i < InputDim; ++i) {
            sum += m_W1[idx(j, i, InputDim)] * a_in[i];
        }
        a_h1[j] = relu(sum);
    }

    for (int j = 0; j < Hidden2; ++j) {
        float sum = m_b2[j];
        for (int i = 0; i < Hidden1; ++i) {
            sum += m_W2[idx(j, i, Hidden1)] * a_h1[i];
        }
        a_h2[j] = relu(sum);
    }

    if (outA1) {
        for (int j = 0; j < Hidden1; ++j) {
            outA1[j] = a_h1[j];
        }
    }
    if (outA2) {
        for (int j = 0; j < Hidden2; ++j) {
            outA2[j] = a_h2[j];
        }
    }

    float maxLogit = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < OutputDim; ++k) {
        float sum = m_b3[k];
        for (int j = 0; j < Hidden2; ++j) {
            sum += m_W3[idx(k, j, Hidden2)] * a_h2[j];
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

void ToyNet::setLearningRate(float lr) {
    m_learningRate = lr;
}

float ToyNet::getLearningRate() const {
    return m_learningRate;
}

const std::vector<float>& ToyNet::getW1() const { return m_W1; }
const std::vector<float>& ToyNet::getB1() const { return m_b1; }
const std::vector<float>& ToyNet::getW2() const { return m_W2; }
const std::vector<float>& ToyNet::getB2() const { return m_b2; }
const std::vector<float>& ToyNet::getW3() const { return m_W3; }
const std::vector<float>& ToyNet::getB3() const { return m_b3; }
