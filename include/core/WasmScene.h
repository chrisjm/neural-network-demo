#pragma once

#ifdef __EMSCRIPTEN__

#include <vector>
#include <memory>

#include "ControlPanel.h"
#include "PlotGeometry.h"
#include "FieldVisualizer.h"
#include "Trainer.h"
#include "DataPoint.h"
#include "DatasetGenerator.h"

class ShaderProgram;

// Shared persistent scene state for the WebAssembly build.
struct WasmSceneState {
    UiState ui;
    std::vector<DataPoint> dataset;
    PointCloud pointCloud;
    GridAxes gridAxes;
    FieldVisualizer fieldVis;
    Trainer trainer;

    bool leftMousePressedLastFrame = false;
    int  maxPoints                 = 0;

    std::unique_ptr<ShaderProgram> pointShader;
    std::unique_ptr<ShaderProgram> gridShader;
    std::unique_ptr<ShaderProgram> fieldShader;

    int pointSizeLocation     = -1;
    int colorClass0Location   = -1;
    int colorClass1Location   = -1;
    int selectedIndexLocation = -1;

    int gridColorLocation = -1;

    int fieldW1Location = -1;
    int fieldB1Location = -1;
    int fieldW2Location = -1;
    int fieldB2Location = -1;
    int fieldW3Location = -1;
    int fieldB3Location = -1;
};

extern WasmSceneState g_wasmState;

#ifdef __cplusplus
extern "C" {
#endif

void nn_set_point_size(float size);
void nn_set_dataset(int datasetIndex, int numPoints, float spread);
void nn_set_auto_train(int enabled);
void nn_step_train();

// Read-back API for JS to query current state.
float nn_get_last_loss();
float nn_get_last_accuracy();
int   nn_get_step_count();

float nn_get_learning_rate();
int   nn_get_batch_size();
int   nn_get_auto_train();

int   nn_get_dataset_index();
int   nn_get_num_points();
float nn_get_spread();
float nn_get_point_size();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __EMSCRIPTEN__
