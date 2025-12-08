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
void nn_shutdown();

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

void nn_set_learning_rate(float value);
void nn_set_batch_size(int value);
void nn_set_auto_max_epochs(int value);
void nn_set_auto_target_loss(float value);
void nn_set_use_target_loss_stop(int enabled);
void nn_set_optimizer(int optimizerType);
void nn_set_momentum(float value);
void nn_set_adam_beta1(float value);
void nn_set_adam_beta2(float value);
void nn_set_adam_eps(float value);
void nn_set_init_mode(int initMode);
void nn_set_probe_enabled(int enabled);
void nn_set_probe_position(float x, float y);

int   nn_get_auto_max_epochs();
float nn_get_auto_target_loss();
int   nn_get_use_target_loss_stop();
int   nn_get_optimizer();
float nn_get_momentum();
float nn_get_adam_beta1();
float nn_get_adam_beta2();
float nn_get_adam_eps();
int   nn_get_init_mode();
int   nn_get_probe_enabled();
float nn_get_probe_x();
float nn_get_probe_y();
int   nn_get_selected_point_index();
int   nn_get_selected_label();
int   nn_get_max_points();

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __EMSCRIPTEN__
