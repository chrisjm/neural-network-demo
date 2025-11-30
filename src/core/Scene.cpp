#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#ifdef __EMSCRIPTEN__
#define GLFW_INCLUDE_ES3
#include <GLFW/glfw3.h>
#else
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#endif

#include <vector>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "Scene.h"
#include "ShaderProgram.h"
#include "GLUtils.h"

void initSceneCommon(DatasetType currentDataset,
                     UiState& ui,
                     int& maxPoints,
                     std::vector<DataPoint>& dataset,
                     PointCloud& pointCloud,
                     GridAxes& gridAxes,
                     FieldVisualizer& fieldVis,
                     bool& leftMousePressedLastFrame) {
    dataset.clear();

    maxPoints = 5000;
    pointCloud.init(maxPoints);

    ui.datasetIndex       = static_cast<int>(currentDataset);
    ui.numPoints          = 1000;
    ui.spread             = 0.25f;
    ui.pointSize          = 6.0f;
    ui.probeEnabled       = true;
    ui.probeX             = 0.0f;
    ui.probeY             = 0.0f;
    ui.hasSelectedPoint   = false;
    ui.selectedPointIndex = -1;
    ui.selectedLabel      = -1;

    generateDataset(currentDataset, ui.numPoints, ui.spread, dataset);
    pointCloud.upload(dataset);

    const float gridStep = 0.25f;
    gridAxes.init(gridStep);

    const int fieldResolution = 64;
    fieldVis.init(fieldResolution);

    leftMousePressedLastFrame = false;
}

void updateAndRenderFrame(FrameContext& ctx) {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiIO& io = ImGui::GetIO();

    bool regenerate = false;
    bool stepTrainRequested = false;

    drawControlPanel(ctx.ui,
                     ctx.trainer,
                     ctx.dataset.size(),
                     regenerate,
                     stepTrainRequested);

    handleProbeSelection(ctx.window,
                         ctx.dataset,
                         ctx.ui,
                         ctx.leftMousePressedLastFrame,
                         io.WantCaptureMouse);

    if (regenerate) {
        if (ctx.ui.numPoints < 10) ctx.ui.numPoints = 10;
        if (ctx.ui.numPoints > ctx.maxPoints) ctx.ui.numPoints = ctx.maxPoints;
        DatasetType currentDataset = static_cast<DatasetType>(ctx.ui.datasetIndex);
        generateDataset(currentDataset,
                        ctx.ui.numPoints,
                        ctx.ui.spread,
                        ctx.dataset);
        ctx.pointCloud.upload(ctx.dataset);

        ctx.ui.hasSelectedPoint   = false;
        ctx.ui.selectedPointIndex = -1;
        ctx.ui.selectedLabel      = -1;

        ctx.trainer.resetForNewDataset();
        ctx.fieldVis.setDirty();
    }

    if (stepTrainRequested) {
        ctx.trainer.stepOnce(ctx.dataset);
        ctx.fieldVis.setDirty();
    }

    if (ctx.trainer.autoTrain) {
        if (ctx.trainer.stepAuto(ctx.dataset)) {
            ctx.fieldVis.setDirty();
        }
    }

    if (ctx.fieldVis.isDirty()) {
        ctx.fieldVis.update();
    }

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    ctx.fieldShader.use();
    const auto& W1 = ctx.trainer.net.getW1();
    const auto& B1 = ctx.trainer.net.getB1();
    const auto& W2 = ctx.trainer.net.getW2();
    const auto& B2 = ctx.trainer.net.getB2();
    const auto& W3 = ctx.trainer.net.getW3();
    const auto& B3 = ctx.trainer.net.getB3();

    ctx.fieldShader.setFloatArray(ctx.fieldW1Location, W1.data(), static_cast<int>(W1.size()));
    ctx.fieldShader.setFloatArray(ctx.fieldB1Location, B1.data(), static_cast<int>(B1.size()));
    ctx.fieldShader.setFloatArray(ctx.fieldW2Location, W2.data(), static_cast<int>(W2.size()));
    ctx.fieldShader.setFloatArray(ctx.fieldB2Location, B2.data(), static_cast<int>(B2.size()));
    ctx.fieldShader.setFloatArray(ctx.fieldW3Location, W3.data(), static_cast<int>(W3.size()));
    ctx.fieldShader.setFloatArray(ctx.fieldB3Location, B3.data(), static_cast<int>(B3.size()));
    ctx.fieldVis.draw();

    ctx.gridShader.use();
    if (ctx.gridColorLocation != -1) {
        ctx.gridShader.setVec3(ctx.gridColorLocation, 0.15f, 0.15f, 0.15f);
    }
    ctx.gridAxes.drawGrid();

    if (ctx.gridColorLocation != -1) {
        ctx.gridShader.setVec3(ctx.gridColorLocation, 0.8f, 0.8f, 0.8f);
    }
    ctx.gridAxes.drawAxes();

    ctx.pointShader.use();

    if (ctx.pointSizeLocation != -1) {
        ctx.pointShader.setFloat(ctx.pointSizeLocation, ctx.ui.pointSize);
    }
    if (ctx.colorClass0Location != -1) {
        ctx.pointShader.setVec3(ctx.colorClass0Location, 0.2f, 0.6f, 1.0f);
    }
    if (ctx.colorClass1Location != -1) {
        ctx.pointShader.setVec3(ctx.colorClass1Location, 1.0f, 0.5f, 0.2f);
    }
    if (ctx.selectedIndexLocation != -1) {
        int selIndex = -1;
        if (ctx.ui.hasSelectedPoint &&
            ctx.ui.selectedPointIndex >= 0 &&
            ctx.ui.selectedPointIndex < static_cast<int>(ctx.dataset.size())) {
            selIndex = ctx.ui.selectedPointIndex;
        }
        ctx.pointShader.setInt(ctx.selectedIndexLocation, selIndex);
    }

    ctx.pointCloud.draw(ctx.dataset.size());

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(ctx.window);
    glfwPollEvents();
}
