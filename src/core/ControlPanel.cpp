#include "ControlPanel.h"

#include "ToyNet.h"
#include "DatasetGenerator.h"

#include "imgui.h"

void drawControlPanel(UiState& ui,
                      Trainer& trainer,
                      std::size_t currentPointCount,
                      bool& regenerateRequested,
                      bool& stepTrainRequested)
{
    regenerateRequested = false;
    stepTrainRequested = false;

    ImGui::Begin("Neural Net Demo");

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
}
