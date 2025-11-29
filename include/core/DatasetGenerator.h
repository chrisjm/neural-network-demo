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

// Number of dataset types defined above.
constexpr int DatasetTypeCount = 5;

// Generate a dataset of the given type.
// - numPoints: total number of points to generate.
// - spread: used as either radial spread or noise amount depending on the dataset.
// - out: vector that will be filled with DataPoint entries.
void generateDataset(DatasetType type,
                     int numPoints,
                     float spread,
                     std::vector<DataPoint>& out);

// Return a pointer to a static array of dataset type names.
// The length of the array is DatasetTypeCount.
const char* const* getDatasetTypeNames();

// Return a human-readable name for a single dataset type.
const char* datasetTypeToString(DatasetType type);
