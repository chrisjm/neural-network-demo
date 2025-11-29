#pragma once

#include <vector>

#include "DataPoint.h"

// Types of synthetic 2D datasets that can be generated for the demo.
enum class DatasetType {
    TwoBlobs = 0,
    ConcentricCircles,
    TwoMoons,
    XORQuads,
    Spirals
};

// Generate a dataset of the given type.
// - numPoints: total number of points to generate.
// - spread: used as either radial spread or noise amount depending on the dataset.
// - out: vector that will be filled with DataPoint entries.
void generateDataset(DatasetType type,
                     int numPoints,
                     float spread,
                     std::vector<DataPoint>& out);
