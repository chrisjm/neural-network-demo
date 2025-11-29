#pragma once

#include <vector>

struct GLFWwindow;
struct UiState;
struct DataPoint;
class Trainer;
class PointCloud;
class GridAxes;
class FieldVisualizer;
class ShaderProgram;

class App {
public:
    App();

    bool init();
    int run();

private:
    void renderLoop(GLFWwindow* window,
                    ShaderProgram& pointShader,
                    ShaderProgram& gridShader,
                    ShaderProgram& fieldShader,
                    int pointSizeLocation,
                    int colorClass0Location,
                    int colorClass1Location,
                    int selectedIndexLocation,
                    int gridColorLocation,
                    int fieldW1Location,
                    int fieldB1Location,
                    int fieldW2Location,
                    int fieldB2Location,
                    int fieldW3Location,
                    int fieldB3Location,
                    UiState& ui,
                    std::vector<DataPoint>& dataset,
                    PointCloud& pointCloud,
                    GridAxes& gridAxes,
                    FieldVisualizer& fieldVis,
                    Trainer& trainer,
                    bool& leftMousePressedLastFrame,
                    int maxPoints);

    void shutdownScene(PointCloud& pointCloud,
                       GridAxes& gridAxes,
                       FieldVisualizer& fieldVis);

    void shutdownApp();

    GLFWwindow* m_window;
};
