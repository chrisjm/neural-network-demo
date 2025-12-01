#ifdef __EMSCRIPTEN__

#include "WasmScene.h"

// C API functions for controlling the wasm scene from JavaScript.

extern "C" {

void nn_set_point_size(float size) {
    if (size < 1.0f) size = 1.0f;
    g_wasmState.ui.pointSize = size;
}

void nn_set_dataset(int datasetIndex, int numPoints, float spread) {
    if (numPoints < 10) numPoints = 10;
    if (numPoints > g_wasmState.maxPoints) numPoints = g_wasmState.maxPoints;

    g_wasmState.ui.datasetIndex = datasetIndex;
    g_wasmState.ui.numPoints    = numPoints;
    g_wasmState.ui.spread       = spread;

    DatasetType currentDataset = static_cast<DatasetType>(g_wasmState.ui.datasetIndex);
    generateDataset(currentDataset,
                    g_wasmState.ui.numPoints,
                    g_wasmState.ui.spread,
                    g_wasmState.dataset);
    g_wasmState.pointCloud.upload(g_wasmState.dataset);

    g_wasmState.ui.hasSelectedPoint   = false;
    g_wasmState.ui.selectedPointIndex = -1;
    g_wasmState.ui.selectedLabel      = -1;

    g_wasmState.trainer.resetForNewDataset();
    g_wasmState.fieldVis.setDirty();
}

void nn_set_auto_train(int enabled) {
    g_wasmState.trainer.autoTrain = (enabled != 0);
}

void nn_step_train() {
    g_wasmState.trainer.trainOneEpoch(g_wasmState.dataset);
    g_wasmState.fieldVis.setDirty();
}

void nn_set_learning_rate(float value) {
    if (value <= 0.0f) value = 1e-6f;
    g_wasmState.trainer.learningRate = value;
}

void nn_set_batch_size(int value) {
    if (value < 1) value = 1;
    if (value > ToyNet::MaxBatch) value = ToyNet::MaxBatch;
    g_wasmState.trainer.batchSize = value;
}

void nn_set_auto_max_epochs(int value) {
    if (value < 0) value = 0;
    g_wasmState.trainer.autoMaxEpochs = value;
}

void nn_set_auto_target_loss(float value) {
    if (value < 0.0f) value = 0.0f;
    g_wasmState.trainer.autoTargetLoss = value;
}

void nn_set_use_target_loss_stop(int enabled) {
    g_wasmState.trainer.useTargetLossStop = (enabled != 0);
}

void nn_set_optimizer(int optimizerType) {
    if (optimizerType < 0) optimizerType = 0;
    if (optimizerType > 2) optimizerType = 2;
    g_wasmState.trainer.optimizerType = static_cast<OptimizerType>(optimizerType);
}

void nn_set_momentum(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 0.99f) value = 0.99f;
    g_wasmState.trainer.momentum = value;
}

void nn_set_adam_beta1(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 0.9999f) value = 0.9999f;
    g_wasmState.trainer.adamBeta1 = value;
}

void nn_set_adam_beta2(float value) {
    if (value < 0.0f) value = 0.0f;
    if (value > 0.9999f) value = 0.9999f;
    g_wasmState.trainer.adamBeta2 = value;
}

void nn_set_adam_eps(float value) {
    if (value < 1e-10f) value = 1e-10f;
    g_wasmState.trainer.adamEps = value;
}

void nn_set_init_mode(int initMode) {
    if (initMode < 0) initMode = 0;
    if (initMode > 2) initMode = 2;
    g_wasmState.trainer.initMode = static_cast<InitMode>(initMode);
    g_wasmState.trainer.resetForNewDataset();
    g_wasmState.fieldVis.setDirty();
}

void nn_set_probe_enabled(int enabled) {
    g_wasmState.ui.probeEnabled = (enabled != 0);
}

void nn_set_probe_position(float x, float y) {
    g_wasmState.ui.probeX = x;
    g_wasmState.ui.probeY = y;
}

float nn_get_last_loss() {
    return g_wasmState.trainer.lastLoss;
}

float nn_get_last_accuracy() {
    return g_wasmState.trainer.lastAccuracy;
}

int nn_get_step_count() {
    return g_wasmState.trainer.epochCount;
}

float nn_get_learning_rate() {
    return g_wasmState.trainer.learningRate;
}

int nn_get_batch_size() {
    return g_wasmState.trainer.batchSize;
}

int nn_get_auto_train() {
    return g_wasmState.trainer.autoTrain ? 1 : 0;
}

int nn_get_dataset_index() {
    return g_wasmState.ui.datasetIndex;
}

int nn_get_num_points() {
    return g_wasmState.ui.numPoints;
}

float nn_get_spread() {
    return g_wasmState.ui.spread;
}

float nn_get_point_size() {
    return g_wasmState.ui.pointSize;
}

int nn_get_auto_max_epochs() {
    return g_wasmState.trainer.autoMaxEpochs;
}

float nn_get_auto_target_loss() {
    return g_wasmState.trainer.autoTargetLoss;
}

int nn_get_use_target_loss_stop() {
    return g_wasmState.trainer.useTargetLossStop ? 1 : 0;
}

int nn_get_optimizer() {
    return static_cast<int>(g_wasmState.trainer.optimizerType);
}

float nn_get_momentum() {
    return g_wasmState.trainer.momentum;
}

float nn_get_adam_beta1() {
    return g_wasmState.trainer.adamBeta1;
}

float nn_get_adam_beta2() {
    return g_wasmState.trainer.adamBeta2;
}

float nn_get_adam_eps() {
    return g_wasmState.trainer.adamEps;
}

int nn_get_init_mode() {
    return static_cast<int>(g_wasmState.trainer.initMode);
}

int nn_get_probe_enabled() {
    return g_wasmState.ui.probeEnabled ? 1 : 0;
}

float nn_get_probe_x() {
    return g_wasmState.ui.probeX;
}

float nn_get_probe_y() {
    return g_wasmState.ui.probeY;
}

int nn_get_selected_point_index() {
    return g_wasmState.ui.selectedPointIndex;
}

int nn_get_selected_label() {
    return g_wasmState.ui.selectedLabel;
}

int nn_get_max_points() {
    return g_wasmState.maxPoints;
}

} // extern "C"

#endif // __EMSCRIPTEN__
