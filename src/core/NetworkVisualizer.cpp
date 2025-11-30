#include "NetworkVisualizer.h"

#include "ToyNet.h"

#include "imgui.h"
#include <cmath>

NetworkVisualizer::NetworkVisualizer()
    : m_canvasWidth(320.0f)
    , m_canvasHeight(220.0f)
    , m_marginX(30.0f)
    , m_marginY(20.0f)
    , m_nodeRadius(5.0f)
{
}

void NetworkVisualizer::setCanvasSize(float width, float height) {
    m_canvasWidth  = width;
    m_canvasHeight = height;
}

void NetworkVisualizer::setMargins(float marginX, float marginY) {
    m_marginX = marginX;
    m_marginY = marginY;
}

void NetworkVisualizer::setNodeRadius(float radius) {
    m_nodeRadius = radius;
}

void NetworkVisualizer::draw(const ToyNet& net,
                             bool probeEnabled,
                             float probeX,
                             float probeY)
{
    ImGui::Separator();
    ImGui::Text("Network Diagram");
    ImGui::Text("Architecture: 2 -> %d -> %d -> 2", ToyNet::Hidden1, ToyNet::Hidden2);

    const ImVec2 canvasSize(m_canvasWidth, m_canvasHeight);
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasEnd(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(canvasPos, canvasEnd, IM_COL32(10, 10, 10, 255));
    drawList->AddRect(canvasPos, canvasEnd, IM_COL32(80, 80, 80, 255));

    ImGui::InvisibleButton("net_canvas", canvasSize);

    const int layerCount = 4;
    int layerSizes[layerCount] = { ToyNet::InputDim, ToyNet::Hidden1, ToyNet::Hidden2, ToyNet::OutputDim };

    const float marginX = m_marginX;
    const float marginY = m_marginY;

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

    auto weightThickness = [](float w) -> float {
        float aw = std::fabs(w);
        float t = aw / 2.0f;
        if (t > 1.0f) t = 1.0f;
        return 0.5f + 2.0f * t;
    };

    float probeA1[ToyNet::Hidden1] = {};
    float probeA2[ToyNet::Hidden2] = {};
    float probeP0 = 0.0f;
    float probeP1 = 0.0f;
    bool  hasProbe = false;
    if (probeEnabled) {
        net.forwardSingleWithActivations(probeX, probeY, probeP0, probeP1, probeA1, probeA2);
        hasProbe = true;
    }

    ImVec2 mousePos = ImGui::GetIO().MousePos;
    bool   windowHovered = ImGui::IsWindowHovered();

    // Edge hover state (weights)
    bool  hasEdgeHover         = false;
    int   edgeFromLayer        = -1;
    int   edgeFromIndex        = -1;
    int   edgeToLayer          = -1;
    int   edgeToIndex          = -1;
    float edgeWeight           = 0.0f;
    float edgeSrcActivation    = 0.0f;
    float edgeContribution     = 0.0f;
    float bestEdgeDist2        = 0.0f;
    const float edgeHoverPixel = 6.0f;
    const float edgeHoverR2    = edgeHoverPixel * edgeHoverPixel;

    auto considerEdgeHover = [&](ImVec2 a,
                                 ImVec2 b,
                                 int fromLayer,
                                 int fromIndex,
                                 int toLayer,
                                 int toIndex,
                                 float weight,
                                 float srcActivation) {
        if (!windowHovered) return;

        ImVec2 ab(b.x - a.x, b.y - a.y);
        ImVec2 am(mousePos.x - a.x, mousePos.y - a.y);
        float ab2 = ab.x * ab.x + ab.y * ab.y;
        if (ab2 <= 1e-4f) return;
        float t = (ab.x * am.x + ab.y * am.y) / ab2;
        if (t < 0.0f) t = 0.0f;
        else if (t > 1.0f) t = 1.0f;
        ImVec2 closest(a.x + t * ab.x, a.y + t * ab.y);
        float dx = mousePos.x - closest.x;
        float dy = mousePos.y - closest.y;
        float dist2 = dx * dx + dy * dy;
        if (dist2 <= edgeHoverR2 && (!hasEdgeHover || dist2 < bestEdgeDist2)) {
            hasEdgeHover      = true;
            bestEdgeDist2     = dist2;
            edgeFromLayer     = fromLayer;
            edgeFromIndex     = fromIndex;
            edgeToLayer       = toLayer;
            edgeToIndex       = toIndex;
            edgeWeight        = weight;
            edgeSrcActivation = srcActivation;
            edgeContribution  = hasProbe ? (srcActivation * weight) : 0.0f;
        }
    };

    // Connections: Input -> Hidden1 (W1)
    const auto& W1 = net.getW1();
    const auto& B1 = net.getB1();
    const auto& W2 = net.getW2();
    const auto& B2 = net.getB2();
    const auto& W3 = net.getW3();
    const auto& B3 = net.getB3();

    for (int j = 0; j < ToyNet::Hidden1; ++j) {
        ImVec2 toPos = nodePos(1, j);
        for (int i = 0; i < ToyNet::InputDim; ++i) {
            ImVec2 fromPos = nodePos(0, i);
            float w = W1[j * ToyNet::InputDim + i];
            float thickness = weightThickness(w);
            drawList->AddLine(fromPos, toPos, weightColor(w), thickness);

            float srcAct = 0.0f;
            if (hasProbe) {
                srcAct = (i == 0) ? probeX : probeY;
            }
            considerEdgeHover(fromPos, toPos, 0, i, 1, j, w, srcAct);
        }
    }

    // Connections: Hidden1 -> Hidden2 (W2)
    for (int j = 0; j < ToyNet::Hidden2; ++j) {
        ImVec2 toPos = nodePos(2, j);
        for (int i = 0; i < ToyNet::Hidden1; ++i) {
            ImVec2 fromPos = nodePos(1, i);
            float w = W2[j * ToyNet::Hidden1 + i];
            float thickness = weightThickness(w);
            drawList->AddLine(fromPos, toPos, weightColor(w), thickness);

            float srcAct = 0.0f;
            if (hasProbe) {
                srcAct = probeA1[i];
            }
            considerEdgeHover(fromPos, toPos, 1, i, 2, j, w, srcAct);
        }
    }

    // Connections: Hidden2 -> Output (W3)
    for (int k = 0; k < ToyNet::OutputDim; ++k) {
        ImVec2 toPos = nodePos(3, k);
        for (int j = 0; j < ToyNet::Hidden2; ++j) {
            ImVec2 fromPos = nodePos(2, j);
            float w = W3[k * ToyNet::Hidden2 + j];
            float thickness = weightThickness(w);
            drawList->AddLine(fromPos, toPos, weightColor(w), thickness);

            float srcAct = 0.0f;
            if (hasProbe) {
                srcAct = probeA2[j];
            }
            considerEdgeHover(fromPos, toPos, 2, j, 3, k, w, srcAct);
        }
    }

    // Draw nodes on top of connections
    const float nodeRadius = m_nodeRadius;
    ImU32 baseNodeColor = IM_COL32(220, 220, 220, 255);

    bool   hasHover = false;
    int    hoverLayer = -1;
    int    hoverIndex = -1;
    float  hoverBias = 0.0f;
    float  hoverActivation = 0.0f;
    float  hoverP0 = 0.0f;
    float  hoverP1 = 0.0f;
    float  bestDist2 = 0.0f;

    for (int layer = 0; layer < layerCount; ++layer) {
        for (int i = 0; i < layerSizes[layer]; ++i) {
            ImVec2 p = nodePos(layer, i);

            float bias = 0.0f;
            if (layer == 1 && i < static_cast<int>(B1.size())) {
                bias = B1[i];
            } else if (layer == 2 && i < static_cast<int>(B2.size())) {
                bias = B2[i];
            } else if (layer == 3 && i < static_cast<int>(B3.size())) {
                bias = B3[i];
            }

            if (bias != 0.0f) {
                float ab = std::fabs(bias);
                float t = ab / 2.0f;
                if (t > 1.0f) t = 1.0f;
                int r, g, b;
                if (bias >= 0.0f) {
                    r = static_cast<int>(150 + 80 * t);
                    g = static_cast<int>(150 + 80 * t);
                    b = 100;
                } else {
                    r = 100;
                    g = static_cast<int>(150 + 80 * t);
                    b = static_cast<int>(150 + 80 * t);
                }
                ImU32 haloColor = IM_COL32(r, g, b, 120);
                drawList->AddCircleFilled(p, nodeRadius + 2.5f, haloColor, 16);
            }

            float activation = 0.0f;
            if (hasProbe) {
                if (layer == 0) {
                    activation = (i == 0) ? probeX : probeY;
                } else if (layer == 1 && i < ToyNet::Hidden1) {
                    activation = probeA1[i];
                } else if (layer == 2 && i < ToyNet::Hidden2) {
                    activation = probeA2[i];
                } else if (layer == 3 && i < ToyNet::OutputDim) {
                    activation = (i == 0) ? probeP0 : probeP1;
                }
            }

            ImU32 nodeColor = baseNodeColor;
            if (hasProbe) {
                float a = activation;
                float mag = std::fabs(a);
                float v = mag * 3.0f;
                if (v > 1.0f) v = 1.0f;

                const int baseR = 220;
                const int baseG = 220;
                const int baseB = 220;

                int r, g, b;
                if (a >= 0.0f) {
                    const int targetR = 255;
                    const int targetG = 180;
                    const int targetB = 50;
                    r = static_cast<int>(baseR + (targetR - baseR) * v);
                    g = static_cast<int>(baseG + (targetG - baseG) * v);
                    b = static_cast<int>(baseB + (targetB - baseB) * v);
                } else {
                    const int targetR = 80;
                    const int targetG = 140;
                    const int targetB = 255;
                    r = static_cast<int>(baseR + (targetR - baseR) * v);
                    g = static_cast<int>(baseG + (targetG - baseG) * v);
                    b = static_cast<int>(baseB + (targetB - baseB) * v);
                }
                nodeColor = IM_COL32(r, g, b, 255);
            }

            drawList->AddCircleFilled(p, nodeRadius, nodeColor, 16);

            if (windowHovered) {
                float dx = mousePos.x - p.x;
                float dy = mousePos.y - p.y;
                float dist2 = dx * dx + dy * dy;
                float radius2 = nodeRadius * nodeRadius * 1.5f;
                if (dist2 <= radius2 && (!hasHover || dist2 < bestDist2)) {
                    hasHover = true;
                    bestDist2 = dist2;
                    hoverLayer = layer;
                    hoverIndex = i;
                    hoverBias = bias;
                    hoverActivation = activation;
                    hoverP0 = probeP0;
                    hoverP1 = probeP1;
                }
            }
        }
    }

    if (hasHover) {
        const char* layerName = "";
        if (hoverLayer == 0) layerName = "Input";
        else if (hoverLayer == 1) layerName = "Hidden 1";
        else if (hoverLayer == 2) layerName = "Hidden 2";
        else if (hoverLayer == 3) layerName = "Output";

        ImGui::BeginTooltip();
        ImGui::Text("%s neuron %d", layerName, hoverIndex);
        ImGui::Text("Bias: %.4f", hoverBias);
        if (hasProbe) {
            if (hoverLayer == 0) {
                ImGui::Text("Probe input: (x=%.3f, y=%.3f)", probeX, probeY);
            } else if (hoverLayer == 3 && hoverIndex < ToyNet::OutputDim) {
                ImGui::Text("Probe probs: p0=%.3f, p1=%.3f", probeP0, probeP1);
            } else {
                ImGui::Text("Activation (probe): %.4f", hoverActivation);
            }
        }
        ImGui::EndTooltip();
    } else if (hasEdgeHover) {
        const char* fromLayerName = "";
        const char* toLayerName   = "";

        if (edgeFromLayer == 0) fromLayerName = "Input";
        else if (edgeFromLayer == 1) fromLayerName = "Hidden 1";
        else if (edgeFromLayer == 2) fromLayerName = "Hidden 2";

        if (edgeToLayer == 1) toLayerName = "Hidden 1";
        else if (edgeToLayer == 2) toLayerName = "Hidden 2";
        else if (edgeToLayer == 3) toLayerName = "Output";

        ImGui::BeginTooltip();

        if (edgeFromLayer == 0) {
            const char* comp = (edgeFromIndex == 0) ? "x" : "y";
            ImGui::Text("Weight: Input %s -> %s neuron %d", comp, toLayerName, edgeToIndex);
        } else if (edgeToLayer == 3) {
            ImGui::Text("Weight: %s neuron %d -> Output neuron %d",
                        fromLayerName, edgeFromIndex, edgeToIndex);
        } else {
            ImGui::Text("Weight: %s neuron %d -> %s neuron %d",
                        fromLayerName, edgeFromIndex, toLayerName, edgeToIndex);
        }

        ImGui::Text("Value: %.4f", edgeWeight);
        if (hasProbe) {
            ImGui::Text("Source activation (probe): %.4f", edgeSrcActivation);
            ImGui::Text("Contribution (probe): %.4f", edgeContribution);
        }

        ImGui::EndTooltip();
    }

    ImGui::Separator();
    ImGui::Text("Layers: Input (2) -> Hidden1 (4 ReLU) -> Hidden2 (8 ReLU) -> Output (2)");
    ImGui::Text("Legend:");
    ImGui::BulletText("Line color = sign of weight, thickness = |weight|");
    ImGui::BulletText("Halo = large bias magnitude");
    ImGui::BulletText("Node color (with probe) = activation for probe point");
}
