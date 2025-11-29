#pragma once

#include <cstddef>

#include "Trainer.h"

struct UiState {
    int   datasetIndex;
    int   numPoints;
    float spread;
    float pointSize;
    bool  probeEnabled;
    float probeX;
    float probeY;
    bool  hasSelectedPoint;
    int   selectedPointIndex;
    int   selectedLabel;
};

void drawControlPanel(UiState& ui,
                      Trainer& trainer,
                      std::size_t currentPointCount,
                      bool& regenerateRequested,
                      bool& stepTrainRequested);
