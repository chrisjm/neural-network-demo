#pragma once

#include <vector>

#include "DataPoint.h"

struct ToyNet {
public:
    static constexpr int InputDim  = 2;
    static constexpr int Hidden1   = 4;
    static constexpr int Hidden2   = 8;
    static constexpr int OutputDim = 2;
    static constexpr int MaxBatch  = 256;

    ToyNet();

    void resetParameters(unsigned int seed = 1);

    float trainBatch(const std::vector<DataPoint>& batch, float& outAccuracy);

    void forwardSingleWithActivations(float x, float y,
                                      float& p0, float& p1,
                                      float* outA1,
                                      float* outA2) const;

    void forwardSingle(float x, float y, float& p0, float& p1) const;

    void setLearningRate(float lr);
    float getLearningRate() const;

    const std::vector<float>& getW1() const;
    const std::vector<float>& getB1() const;
    const std::vector<float>& getW2() const;
    const std::vector<float>& getB2() const;
    const std::vector<float>& getW3() const;
    const std::vector<float>& getB3() const;

private:
    float m_learningRate;

    std::vector<float> m_W1;
    std::vector<float> m_b1;
    std::vector<float> m_W2;
    std::vector<float> m_b2;
    std::vector<float> m_W3;
    std::vector<float> m_b3;

    std::vector<float> m_a0;
    std::vector<float> m_z1;
    std::vector<float> m_a1;
    std::vector<float> m_z2;
    std::vector<float> m_a2;
    std::vector<float> m_logits;
    std::vector<float> m_probs;

    std::vector<float> m_dW1;
    std::vector<float> m_db1;
    std::vector<float> m_dW2;
    std::vector<float> m_db2;
    std::vector<float> m_dW3;
    std::vector<float> m_db3;
};
