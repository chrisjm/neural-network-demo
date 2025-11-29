#pragma once

#include <vector>

struct GLFWwindow;
struct Object2D;
struct DataPoint;
struct UiState;

struct MouseDebugState {
    bool  hasClick;
    double mouseX;
    double mouseY;
    float xNdc;
    float yNdc;
    float triLocalX;
    float triLocalY;
    float squareLocalX;
    float squareLocalY;
    bool  hitTriangle;
    bool  hitSquare;
};

// Handle keyboard input for all objects, updating the currently selected
// object index and applying movement/scale/rotation/color changes.
void handleKeyboardInput(GLFWwindow* window,
                         Object2D* objects,
                         int objectCount,
                         int& selectedObject,
                         bool& tabPressedLastFrame);

// Handle mouse input (currently selection via picking). This is designed
// to be extended later with drag-and-drop and other mouse-driven
// manipulations.
void handleMouseInput(GLFWwindow* window,
                      Object2D* objects,
                      int objectCount,
                      int& selectedObject,
                      bool& leftMousePressedLastFrame,
                      const float* triangleVertices,
                      MouseDebugState* debugState);

// Handle mouse input for selecting a probe point in the neural net demo.
// Clicks near dataset points will set UiState's probe position and selection.
void handleProbeSelection(GLFWwindow* window,
                          const std::vector<DataPoint>& dataset,
                          UiState& ui,
                          bool& leftMousePressedLastFrame,
                          bool mouseOverGui);
