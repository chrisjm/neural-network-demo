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

} // extern "C"

#endif // __EMSCRIPTEN__
