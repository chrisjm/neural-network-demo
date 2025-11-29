#pragma once

#include <vector>

#include "DataPoint.h"

struct ToyNet {

    static constexpr int InputDim  = 2;
    static constexpr int Hidden1   = 4;
    static constexpr int Hidden2   = 8;
    static constexpr int OutputDim = 2;
    static constexpr int MaxBatch  = 256;

    float learningRate = 0.1f;

    std::vector<float> W1;
    std::vector<float> b1;
    std::vector<float> W2;
    std::vector<float> b2;
    std::vector<float> W3;
    std::vector<float> b3;

    std::vector<float> a0;
    std::vector<float> z1;
    std::vector<float> a1;
    std::vector<float> z2;
    std::vector<float> a2;
    std::vector<float> logits;
    std::vector<float> probs;

    std::vector<float> dW1;
    std::vector<float> db1;
    std::vector<float> dW2;
    std::vector<float> db2;
    std::vector<float> dW3;
    std::vector<float> db3;

    ToyNet();

    void resetParameters(unsigned int seed = 1);

    float trainBatch(const std::vector<DataPoint>& batch, float& outAccuracy);

    void forwardSingle(float x, float y, float& p0, float& p1) const;
};
