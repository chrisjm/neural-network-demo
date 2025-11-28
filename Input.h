#pragma once

struct GLFWwindow;
struct Object2D;

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
                      const float* triangleVertices);
