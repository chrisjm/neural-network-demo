#include "ControlPanel.h"

#include "ToyNet.h"
#include "DatasetGenerator.h"
#include "NetworkVisualizer.h"

#include "imgui.h"

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
    ImGui::SetNextWindowPos(controlsPos, ImGuiCond_Once);
    ImGui::SetNextWindowSize(controlsSize, ImGuiCond_Once);

    ImGui::Begin("Neural Net Controls");

    const char* const* datasetNames = getDatasetTypeNames();

    if (ImGui::Combo("Dataset", &ui.datasetIndex, datasetNames, DatasetTypeCount)) {
        regenerateRequested = true;
    }

    regenerateRequested |= ImGui::SliderInt("Points", &ui.numPoints, 100, 5000);
    regenerateRequested |= ImGui::SliderFloat("Spread", &ui.spread, 0.01f, 0.5f);
    if (ImGui::Button("Regenerate Data")) {
        regenerateRequested = true;
    }

    ImGui::Separator();
    ImGui::SliderFloat("Point Size", &ui.pointSize, 2.0f, 12.0f);

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

    ImGui::Separator();
    ImGui::SliderFloat("Learning Rate", &trainer.learningRate, 0.0001f, 1.0f, "%.5f");
    ImGui::SliderInt("Batch Size", &trainer.batchSize, 1, ToyNet::MaxBatch);

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
    ImGui::Text("Auto stops when step >= %d or loss <= %.5f", trainer.autoMaxSteps, trainer.autoTargetLoss);

    ImGui::Text("Current points: %d", static_cast<int>(currentPointCount));

    ImGui::End();

    // Separate window for the network diagram, positioned below the controls by default.
    ImVec2 diagramSize(360.0f, 260.0f);
    ImVec2 diagramPos(controlsPos.x, controlsPos.y + controlsSize.y + 10.0f);
    ImGui::SetNextWindowPos(diagramPos, ImGuiCond_Once);
    ImGui::SetNextWindowSize(diagramSize, ImGuiCond_Once);

    ImGui::Begin("Network Diagram");
    static NetworkVisualizer visualizer;
    visualizer.draw(trainer.net, ui.probeEnabled, ui.probeX, ui.probeY);
    ImGui::End();

    // Loss plot window.
    ImVec2 lossSize(360.0f, 160.0f);
    ImVec2 lossPos(controlsPos.x, io.DisplaySize.y - lossSize.y - 20.0f);
    ImGui::SetNextWindowPos(lossPos, ImGuiCond_Once);
    ImGui::SetNextWindowSize(lossSize, ImGuiCond_Once);

    ImGui::Begin("Loss Plot");
    if (trainer.historyCount > 0) {
        ImGui::PlotLines("Loss", trainer.lossHistory, trainer.historyCount, 0, nullptr, 0.0f, trainer.lossHistory[0] + 1.0f, ImVec2(-1.0f, 100.0f));
    } else {
        ImGui::Text("No data yet");
    }
    ImGui::End();

    // Accuracy plot window.
    ImVec2 accSize(360.0f, 160.0f);
    ImVec2 accPos(controlsPos.x - accSize.x - 10.0f, io.DisplaySize.y - accSize.y - 20.0f);
    ImGui::SetNextWindowPos(accPos, ImGuiCond_Once);
    ImGui::SetNextWindowSize(accSize, ImGuiCond_Once);

    ImGui::Begin("Accuracy Plot");
    if (trainer.historyCount > 0) {
        ImGui::PlotLines("Accuracy", trainer.accuracyHistory, trainer.historyCount, 0, nullptr, 0.0f, 1.0f, ImVec2(-1.0f, 100.0f));
    } else {
        ImGui::Text("No data yet");
    }
    ImGui::End();
}
