#pragma once

class ToyNet;

class NetworkVisualizer {
public:
    void draw(const ToyNet& net,
              bool probeEnabled,
              float probeX,
              float probeY);
};
