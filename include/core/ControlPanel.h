#pragma once

#include <cstddef>

#include "Trainer.h"

struct UiState {
    int   datasetIndex;
    int   numPoints;
    float spread;
    float pointSize;
    int   hidden1;
    int   hidden2;
};

void drawControlPanel(UiState& ui,
                      Trainer& trainer,
                      std::size_t currentPointCount,
                      bool& regenerateRequested,
                      bool& stepTrainRequested);
