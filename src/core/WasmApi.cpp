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
    g_wasmState.trainer.stepOnce(g_wasmState.dataset);
    g_wasmState.fieldVis.setDirty();
}

} // extern "C"

#endif // __EMSCRIPTEN__
