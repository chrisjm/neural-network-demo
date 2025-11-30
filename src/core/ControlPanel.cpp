#include "ControlPanel.h"

#include "ToyNet.h"
#include "DatasetGenerator.h"
#include "NetworkVisualizer.h"

#include "imgui.h"

static void drawDatasetSection(UiState& ui,
                               std::size_t currentPointCount,
                               bool& regenerateRequested)
{
    const char* const* datasetNames = getDatasetTypeNames();

    if (ImGui::Combo("Dataset", &ui.datasetIndex, datasetNames, DatasetTypeCount)) {
        regenerateRequested = true;
    }

    regenerateRequested |= ImGui::SliderInt("Points", &ui.numPoints, 100, 5000);
    regenerateRequested |= ImGui::SliderFloat("Spread", &ui.spread, 0.01f, 0.5f);
    if (ImGui::Button("Regenerate Data")) {
        regenerateRequested = true;
    }
}

static void drawProbeSection(UiState& ui, Trainer& trainer)
{
    ImGui::Separator();
    ImGui::Checkbox("Activation Probe Enabled", &ui.probeEnabled);
    ImGui::DragFloat("Probe X", &ui.probeX, 0.01f, -1.5f, 1.5f, "%.2f");
    ImGui::DragFloat("Probe Y", &ui.probeY, 0.01f, -1.5f, 1.5f, "%.2f");

    if (ui.hasSelectedPoint && ui.selectedPointIndex >= 0) {
        float p0 = 0.0f;
        float p1 = 0.0f;
        trainer.net.forwardSingle(ui.probeX, ui.probeY, p0, p1);
        int predicted = (p1 > p0) ? 1 : 0;

        ImGui::Text("Selected sample: index %d, class %d", ui.selectedPointIndex, ui.selectedLabel);
        ImGui::Text("Coords: (x=%.3f, y=%.3f)", ui.probeX, ui.probeY);
        ImGui::Text("Prediction: class %d (p0=%.3f, p1=%.3f)", predicted, p0, p1);
    }
}

static void drawHyperparameterSection(Trainer& trainer)
{
    ImGui::Separator();
    const char* initNames[] = { "Zero", "He Uniform", "He Normal" };
    int         initIdx     = static_cast<int>(trainer.initMode);
    if (ImGui::Combo("Init Mode", &initIdx, initNames, IM_ARRAYSIZE(initNames))) {
        if (initIdx < 0) initIdx = 0;
        if (initIdx > 2) initIdx = 2;
        trainer.initMode = static_cast<InitMode>(initIdx);
        trainer.resetForNewDataset();
    }

    ImGui::Separator();
    ImGui::SliderFloat("Learning Rate", &trainer.learningRate, 0.0001f, 0.2f, "%.5f");
    ImGui::SliderInt("Batch Size", &trainer.batchSize, 1, ToyNet::MaxBatch);

    ImGui::Separator();
    const char* optimizerNames[] = { "SGD", "SGD + Momentum", "Adam" };
    static int  prevOptimizerIdx = 0;
    int         optimizerIdx     = static_cast<int>(trainer.optimizerType);
    if (ImGui::Combo("Optimizer", &optimizerIdx, optimizerNames, IM_ARRAYSIZE(optimizerNames))) {
        if (optimizerIdx < 0) optimizerIdx = 0;
        if (optimizerIdx > 2) optimizerIdx = 2;

        if (optimizerIdx != prevOptimizerIdx) {
            if (optimizerIdx == static_cast<int>(OptimizerType::Adam)) {
                if (trainer.learningRate > 0.05f) {
                    trainer.learningRate = 0.01f;
                }
            } else if (prevOptimizerIdx == static_cast<int>(OptimizerType::Adam) &&
                       optimizerIdx == static_cast<int>(OptimizerType::SGD)) {
                if (trainer.learningRate < 0.02f) {
                    trainer.learningRate = 0.1f;
                }
            }
            prevOptimizerIdx = optimizerIdx;
        }

        trainer.optimizerType = static_cast<OptimizerType>(optimizerIdx);
    }

    if (trainer.optimizerType == OptimizerType::SGDMomentum) {
        ImGui::SliderFloat("Momentum", &trainer.momentum, 0.0f, 0.95f, "%.2f");
    } else if (trainer.optimizerType == OptimizerType::Adam) {
        ImGui::SliderFloat("Adam Beta1", &trainer.adamBeta1, 0.7f, 0.99f, "%.3f");
        ImGui::SliderFloat("Adam Beta2", &trainer.adamBeta2, 0.9f, 0.999f, "%.3f");
        ImGui::SliderFloat("Adam Eps", &trainer.adamEps, 1e-8f, 1e-4f, "%.1e");
    }
}

static void drawTrainingSection(UiState& ui,
                                Trainer& trainer,
                                std::size_t currentPointCount,
                                bool& stepTrainRequested)
{
    if (ImGui::Button("Step Train")) {
        stepTrainRequested = true;
    }
    ImGui::SameLine();
    ImGui::Checkbox("Auto Train", &trainer.autoTrain);

    ImGui::Text("Step: %d", trainer.stepCount);
    ImGui::Text("Loss: %.4f", trainer.lastLoss);
    ImGui::Text("Accuracy: %.3f", trainer.lastAccuracy);

    ImGui::Separator();
    ImGui::SliderInt("Auto Max Steps", &trainer.autoMaxSteps, 1, 50000);
    ImGui::SliderFloat("Auto Target Loss", &trainer.autoTargetLoss, 0.00001f, 1.0f, "%.5f");
    ImGui::Text("Current points: %d", static_cast<int>(currentPointCount));

    ImGui::Separator();
}

static void drawNetworkDiagramWindow(const UiState& ui,
                                     Trainer& trainer,
                                     const ImVec2& controlsPos,
                                     const ImVec2& controlsSize)
{
    // Separate window for the network diagram, positioned below the controls by default.
    ImVec2 diagramSize(360.0f, 260.0f);
    ImVec2 diagramPos(controlsPos.x, controlsPos.y + controlsSize.y + 10.0f);
    ImGui::SetNextWindowPos(diagramPos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(diagramSize, ImGuiCond_FirstUseEver);

    ImGui::Begin("Network Diagram");
    static NetworkVisualizer visualizer;
    visualizer.draw(trainer.net, ui.probeEnabled, ui.probeX, ui.probeY);
    ImGui::End();
}

static void drawLossPlotWindow(const Trainer& trainer,
                               const ImVec2& controlsPos,
                               const ImGuiIO& io)
{
    // Loss plot window.
    ImVec2 lossSize(360.0f, 160.0f);
    ImVec2 lossPos(controlsPos.x, io.DisplaySize.y - lossSize.y - 20.0f);
    ImGui::SetNextWindowPos(lossPos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(lossSize, ImGuiCond_FirstUseEver);

    ImGui::Begin("Loss Plot");
    if (trainer.historyCount > 0 && !trainer.lossHistory.empty()) {
        ImGui::Text("Loss vs Step");
        ImGui::Separator();
        float maxLoss = trainer.lossHistory[0];
        for (float v : trainer.lossHistory) {
            if (v > maxLoss) maxLoss = v;
        }
        float scaleMax = maxLoss > 0.0f ? maxLoss : 1.0f;
        ImGui::PlotLines("##LossSeries",
                         trainer.lossHistory.data(),
                         trainer.historyCount,
                         0,
                         nullptr,
                         0.0f,
                         scaleMax,
                         ImVec2(-1.0f, 100.0f));
        ImGui::Text("step: 0 -> %d", trainer.stepCount);
    } else {
        ImGui::Text("No data yet");
    }
    ImGui::End();
}

static void drawTrainingOverlayWindow(Trainer& trainer,
                                      const ImGuiIO& io)
{
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav;

    ImVec2 pos(io.DisplaySize.x - 10.0f, 10.0f);
    ImGui::SetNextWindowPos(pos, ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.8f);

    ImGui::Begin("Mini Training Status", nullptr, flags);
    ImGui::Text("Step: %d", trainer.stepCount);
    ImGui::Text("Loss: %.4f", trainer.lastLoss);
    ImGui::Text("Accuracy: %.3f", trainer.lastAccuracy);
    ImGui::Checkbox("Auto Train", &trainer.autoTrain);
    ImGui::End();
}

static void drawAccuracyPlotWindow(const Trainer& trainer,
                                   const ImVec2& controlsPos,
                                   const ImGuiIO& io)
{
    // Accuracy plot window.
    ImVec2 accSize(360.0f, 160.0f);
    ImVec2 accPos(controlsPos.x - accSize.x - 10.0f, io.DisplaySize.y - accSize.y - 20.0f);
    ImGui::SetNextWindowPos(accPos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(accSize, ImGuiCond_FirstUseEver);

    ImGui::Begin("Accuracy Plot");
    if (trainer.historyCount > 0 && !trainer.accuracyHistory.empty()) {
        ImGui::Text("Accuracy vs Step");
        ImGui::Separator();
        ImGui::PlotLines("##AccuracySeries",
                         trainer.accuracyHistory.data(),
                         trainer.historyCount,
                         0,
                         nullptr,
                         0.0f,
                         1.0f,
                         ImVec2(-1.0f, 100.0f));
        ImGui::Text("step: 0 -> %d", trainer.stepCount);
    } else {
        ImGui::Text("No data yet");
    }
    ImGui::End();
}

void drawControlPanel(UiState& ui,
                      Trainer& trainer,
                      std::size_t currentPointCount,
                      bool& regenerateRequested,
                      bool& stepTrainRequested)
{
    regenerateRequested = false;
    stepTrainRequested = false;

    ImGuiIO& io = ImGui::GetIO();

    // Main controls window on the right side.
    ImVec2 controlsSize(360.0f, 260.0f);
    ImVec2 controlsPos(io.DisplaySize.x - controlsSize.x - 20.0f, 20.0f);

    ImGui::SetNextWindowPos(controlsPos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(controlsSize, ImGuiCond_FirstUseEver);
    ImGui::Begin("Data & Probe");
    drawDatasetSection(ui, currentPointCount, regenerateRequested);
    drawProbeSection(ui, trainer);
    ImGui::End();

    ImVec2 trainSize(360.0f, 260.0f);
    ImVec2 trainPos(controlsPos.x,
                    controlsPos.y + controlsSize.y + 10.0f + 260.0f + 10.0f);
    ImGui::SetNextWindowPos(trainPos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(trainSize, ImGuiCond_FirstUseEver);
    ImGui::Begin("Training & Hyperparams");
    drawHyperparameterSection(trainer);
    drawTrainingSection(ui, trainer, currentPointCount, stepTrainRequested);
    ImGui::End();

    drawNetworkDiagramWindow(ui, trainer, controlsPos, controlsSize);
    drawLossPlotWindow(trainer, controlsPos, io);
    drawAccuracyPlotWindow(trainer, controlsPos, io);
}

