#pragma once

#include <vector>

#include "DatasetGenerator.h"
#include "ControlPanel.h"
#include "PlotGeometry.h"
#include "FieldVisualizer.h"
#include "Trainer.h"
#include "DataPoint.h"
#include "Input.h"

struct GLFWwindow;
class ShaderProgram;

struct FrameContext {
    GLFWwindow* window;
    ShaderProgram& pointShader;
    ShaderProgram& gridShader;
    ShaderProgram& fieldShader;
    int pointSizeLocation;
    int colorClass0Location;
    int colorClass1Location;
    int selectedIndexLocation;
    int gridColorLocation;
    int fieldW1Location;
    int fieldB1Location;
    int fieldW2Location;
    int fieldB2Location;
    int fieldW3Location;
    int fieldB3Location;
    UiState& ui;
    std::vector<DataPoint>& dataset;
    PointCloud& pointCloud;
    GridAxes& gridAxes;
    FieldVisualizer& fieldVis;
    Trainer& trainer;
    bool& leftMousePressedLastFrame;
    int maxPoints;
};

void initSceneCommon(DatasetType currentDataset,
                     UiState& ui,
                     int& maxPoints,
                     std::vector<DataPoint>& dataset,
                     PointCloud& pointCloud,
                     GridAxes& gridAxes,
                     FieldVisualizer& fieldVis,
                     bool& leftMousePressedLastFrame);

void updateAndRenderFrame(FrameContext& ctx);
