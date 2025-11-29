#include "ControlPanel.h"

#include "ToyNet.h"
#include "DatasetGenerator.h"

#include "imgui.h"

namespace {

void drawNetworkDiagram(const ToyNet& net)
{
    ImGui::Separator();
    ImGui::Text("Network Diagram");

    const ImVec2 canvasSize(320.0f, 220.0f);
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasEnd(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(canvasPos, canvasEnd, IM_COL32(10, 10, 10, 255));
    drawList->AddRect(canvasPos, canvasEnd, IM_COL32(80, 80, 80, 255));

    ImGui::InvisibleButton("net_canvas", canvasSize);

    const int layerCount = 4;
    int layerSizes[layerCount] = { ToyNet::InputDim, ToyNet::Hidden1, ToyNet::Hidden2, ToyNet::OutputDim };

    const float marginX = 30.0f;
    const float marginY = 20.0f;

    auto nodePos = [&](int layer, int index) -> ImVec2 {
        float x0 = canvasPos.x + marginX;
        float x1 = canvasEnd.x - marginX;
        float t  = (layerCount > 1) ? (static_cast<float>(layer) / static_cast<float>(layerCount - 1)) : 0.0f;
        float x  = x0 + t * (x1 - x0);

        int count = layerSizes[layer];
        float yTop    = canvasPos.y + marginY;
        float yBottom = canvasEnd.y - marginY;
        if (count <= 1) {
            float yMid = 0.5f * (yTop + yBottom);
            return ImVec2(x, yMid);
        }
        float step = (yBottom - yTop) / static_cast<float>(count - 1);
        float y = yTop + step * static_cast<float>(index);
        return ImVec2(x, y);
    };

    auto weightColor = [](float w) -> ImU32 {
        float aw = std::fabs(w);
        float t = aw / 2.0f;
        if (t > 1.0f) t = 1.0f;

        int r, g, b;
        if (w >= 0.0f) {
            r = static_cast<int>(80 + 175 * t);
            g = static_cast<int>(80 + 120 * t);
            b = 80;
        } else {
            r = 80;
            g = static_cast<int>(80 + 120 * t);
            b = static_cast<int>(80 + 175 * t);
        }
        return IM_COL32(r, g, b, 180);
    };

    // Connections: Input -> Hidden1 (W1)
    for (int j = 0; j < ToyNet::Hidden1; ++j) {
        ImVec2 toPos = nodePos(1, j);
        for (int i = 0; i < ToyNet::InputDim; ++i) {
            ImVec2 fromPos = nodePos(0, i);
            float w = net.W1[j * ToyNet::InputDim + i];
            drawList->AddLine(fromPos, toPos, weightColor(w), 1.0f);
        }
    }

    // Connections: Hidden1 -> Hidden2 (W2)
    for (int j = 0; j < ToyNet::Hidden2; ++j) {
        ImVec2 toPos = nodePos(2, j);
        for (int i = 0; i < ToyNet::Hidden1; ++i) {
            ImVec2 fromPos = nodePos(1, i);
            float w = net.W2[j * ToyNet::Hidden1 + i];
            drawList->AddLine(fromPos, toPos, weightColor(w), 1.0f);
        }
    }

    // Connections: Hidden2 -> Output (W3)
    for (int k = 0; k < ToyNet::OutputDim; ++k) {
        ImVec2 toPos = nodePos(3, k);
        for (int j = 0; j < ToyNet::Hidden2; ++j) {
            ImVec2 fromPos = nodePos(2, j);
            float w = net.W3[k * ToyNet::Hidden2 + j];
            drawList->AddLine(fromPos, toPos, weightColor(w), 1.0f);
        }
    }

    // Draw nodes on top of connections
    const float nodeRadius = 4.0f;
    ImU32 nodeColor = IM_COL32(220, 220, 220, 255);

    for (int layer = 0; layer < layerCount; ++layer) {
        for (int i = 0; i < layerSizes[layer]; ++i) {
            ImVec2 p = nodePos(layer, i);
            drawList->AddCircleFilled(p, nodeRadius, nodeColor, 16);
        }
    }
}

} // namespace

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
    drawNetworkDiagram(trainer.net);
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
