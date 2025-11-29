#pragma once

#include <vector>

// Optimizer types used by ToyNet and Trainer.
enum class OptimizerType {
    SGD = 0,
    SGDMomentum = 1,
    Adam = 2
};

// Configuration for a single optimizer step.
struct OptimizerConfig {
    OptimizerType type;
    float         learningRate;
    float         momentum;  // Used for SGD with momentum
    float         beta1;     // Adam first-moment decay
    float         beta2;     // Adam second-moment decay
    float         eps;       // Adam numerical stability term
};

// Reset optimizer state buffers for a new set of randomly initialized parameters.
// This is called when ToyNet parameters are (re)initialized.
void optimizerResetState(
    std::vector<float>& mW1, std::vector<float>& mb1,
    std::vector<float>& mW2, std::vector<float>& mb2,
    std::vector<float>& mW3, std::vector<float>& mb3,
    std::vector<float>& vW1, std::vector<float>& vb1,
    std::vector<float>& vW2, std::vector<float>& vb2,
    std::vector<float>& vW3, std::vector<float>& vb3,
    int&                adamStep);

// Apply an optimizer update step in-place to the network parameters.
// Gradients are assumed to already be averaged over the batch.
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
    int&                      adamStep);
