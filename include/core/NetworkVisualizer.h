#pragma once

class ToyNet;

class NetworkVisualizer {
public:
    NetworkVisualizer();

    void setCanvasSize(float width, float height);
    void setMargins(float marginX, float marginY);
    void setNodeRadius(float radius);

    void draw(const ToyNet& net,
              bool probeEnabled,
              float probeX,
              float probeY);

private:
    float m_canvasWidth;
    float m_canvasHeight;
    float m_marginX;
    float m_marginY;
    float m_nodeRadius;
};
