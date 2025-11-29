// Use GLAD as the OpenGL loader and prevent GLFW from including system GL headers.
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "Input.h"
#include "Object2D.h"
#include "GeometryUtils.h"
#include "DataPoint.h"
#include "ControlPanel.h"

void handleKeyboardInput(GLFWwindow* window,
                         Object2D* objects,
                         int objectCount,
                         int& selectedObject,
                         bool& tabPressedLastFrame) {
    if (!window || !objects || objectCount <= 0) {
        return;
    }

    const float moveSpeed    = 0.01f; // how fast objects move per frame when key is held
    const float scaleStep    = 0.01f; // how much scale changes per key press
    const float rotationStep = 0.05f; // radians per key press (~3 degrees)

    // Allow cycling the active object with TAB.
    int tabState = glfwGetKey(window, GLFW_KEY_TAB);
    if (tabState == GLFW_PRESS && !tabPressedLastFrame) {
        selectedObject = (selectedObject + 1) % objectCount;
        std::cout << "[Input] TAB -> selectedObject = " << selectedObject << std::endl;
    }
    tabPressedLastFrame = (tabState == GLFW_PRESS);

    // Input acts on the currently selected object.
    Object2D& active = objects[selectedObject];

    // Arrow keys move the active object by changing the offset uniform
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        active.offsetY += moveSpeed;
        std::cout << "[Input] UP    -> offset = (" << active.offsetX << ", " << active.offsetY << ")" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        active.offsetY -= moveSpeed;
        std::cout << "[Input] DOWN  -> offset = (" << active.offsetX << ", " << active.offsetY << ")" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        active.offsetX -= moveSpeed;
        std::cout << "[Input] LEFT  -> offset = (" << active.offsetX << ", " << active.offsetY << ")" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        active.offsetX += moveSpeed;
        std::cout << "[Input] RIGHT -> offset = (" << active.offsetX << ", " << active.offsetY << ")" << std::endl;
    }

    // Scale controls (Z/X)
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
        active.scale -= scaleStep;
        if (active.scale < 0.1f) active.scale = 0.1f; // avoid inverting/vanishing
        std::cout << "[Input] Z -> scale = " << active.scale << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
        active.scale += scaleStep;
        std::cout << "[Input] X -> scale = " << active.scale << std::endl;
    }

    // Rotation controls (Q/E)
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        active.rotation -= rotationStep;
        std::cout << "[Input] Q -> rotation = " << active.rotation << " radians" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        active.rotation += rotationStep;
        std::cout << "[Input] E -> rotation = " << active.rotation << " radians" << std::endl;
    }

    // Number keys change the color uniform of the active object
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        active.color[0] = 1.0f; active.color[1] = 0.0f; active.color[2] = 0.0f; // Red
        std::cout << "[Input] 1 -> color = RED" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        active.color[0] = 0.0f; active.color[1] = 1.0f; active.color[2] = 0.0f; // Green
        std::cout << "[Input] 2 -> color = GREEN" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        active.color[0] = 0.0f; active.color[1] = 0.0f; active.color[2] = 1.0f; // Blue
        std::cout << "[Input] 3 -> color = BLUE" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
        active.color[0] = 1.0f; active.color[1] = 1.0f; active.color[2] = 1.0f; // White
        std::cout << "[Input] 4 -> color = WHITE" << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
        active.color[0] = 1.0f; active.color[1] = 0.5f; active.color[2] = 0.2f; // Back to original orange
        std::cout << "[Input] 5 -> color = ORANGE" << std::endl;
    }
}

void handleMouseInput(GLFWwindow* window,
                      Object2D* objects,
                      int objectCount,
                      int& selectedObject,
                      bool& leftMousePressedLastFrame,
                      const float* triangleVertices,
                      MouseDebugState* debugState) {
    if (!window || !objects || objectCount <= 0 || !triangleVertices) {
        return;
    }

    if (debugState) {
        debugState->hasClick = false;
    }

    int leftState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    if (leftState == GLFW_PRESS && !leftMousePressedLastFrame) {
        double mouseX, mouseY;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        int winWidth = 0, winHeight = 0;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        if (winWidth > 0 && winHeight > 0) {
            // Convert from window coordinates (origin at top-left) to
            // normalized device coordinates in [-1, 1]. We use the
            // window size here rather than the framebuffer size so the
            // math matches the coordinate system used by glfwGetCursorPos.
            float xNdc =  2.0f * static_cast<float>(mouseX) / static_cast<float>(winWidth) - 1.0f;
            float yNdc =  1.0f - 2.0f * static_cast<float>(mouseY) / static_cast<float>(winHeight);

            // Test triangle first.
            float triLocalX = 0.0f, triLocalY = 0.0f;
            worldToLocal(xNdc, yNdc,
                         objects[0].offsetX, objects[0].offsetY,
                         objects[0].scale, objects[0].rotation,
                         triLocalX, triLocalY);

            bool hitTriangle = pointInTriangle(
                triLocalX, triLocalY,
                triangleVertices[0], triangleVertices[1],
                triangleVertices[3], triangleVertices[4],
                triangleVertices[6], triangleVertices[7]
            );

            // Then test square (unit square in its local space).
            float squareLocalX = 0.0f, squareLocalY = 0.0f;
            worldToLocal(xNdc, yNdc,
                         objects[1].offsetX, objects[1].offsetY,
                         objects[1].scale, objects[1].rotation,
                         squareLocalX, squareLocalY);

            bool hitSquare = pointInUnitSquare(squareLocalX, squareLocalY);

            if (debugState) {
                debugState->hasClick      = true;
                debugState->mouseX       = mouseX;
                debugState->mouseY       = mouseY;
                debugState->xNdc         = xNdc;
                debugState->yNdc         = yNdc;
                debugState->triLocalX    = triLocalX;
                debugState->triLocalY    = triLocalY;
                debugState->squareLocalX = squareLocalX;
                debugState->squareLocalY = squareLocalY;
                debugState->hitTriangle  = hitTriangle;
                debugState->hitSquare    = hitSquare;
            }

            if (hitTriangle && !hitSquare) {
                selectedObject = 0;
                std::cout << "[Pick] Selected triangle" << std::endl;
            } else if (!hitTriangle && hitSquare) {
                selectedObject = 1;
                std::cout << "[Pick] Selected square" << std::endl;
            } else if (hitTriangle && hitSquare) {
                // If both are hit (overlap), prefer the square for now.
                selectedObject = 1;
                std::cout << "[Pick] Selected square (overlap)" << std::endl;
            }
        }
    }

    leftMousePressedLastFrame = (leftState == GLFW_PRESS);
}

void handleProbeSelection(GLFWwindow* window,
                          const std::vector<DataPoint>& dataset,
                          UiState& ui,
                          bool& leftMousePressedLastFrame,
                          bool mouseOverGui)
{
    if (!window) {
        return;
    }

    int leftState = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    if (leftState == GLFW_PRESS && !leftMousePressedLastFrame && !mouseOverGui) {
        double mouseX = 0.0;
        double mouseY = 0.0;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        int winWidth = 0;
        int winHeight = 0;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        if (winWidth > 0 && winHeight > 0) {
            float xNdc =  2.0f * static_cast<float>(mouseX) / static_cast<float>(winWidth) - 1.0f;
            float yNdc =  1.0f - 2.0f * static_cast<float>(mouseY) / static_cast<float>(winHeight);

            int   bestIndex = -1;
            float bestDist2 = 0.0f;
            const float pickRadius = 0.15f; // enlarged radius for easier selection
            const float maxDist2   = pickRadius * pickRadius;

            for (std::size_t i = 0; i < dataset.size(); ++i) {
                float dx = dataset[i].x - xNdc;
                float dy = dataset[i].y - yNdc;
                float d2 = dx * dx + dy * dy;
                if (d2 <= maxDist2 && (bestIndex < 0 || d2 < bestDist2)) {
                    bestIndex = static_cast<int>(i);
                    bestDist2 = d2;
                }
            }

            if (bestIndex >= 0) {
                ui.probeEnabled       = true;
                ui.probeX             = dataset[bestIndex].x;
                ui.probeY             = dataset[bestIndex].y;
                ui.hasSelectedPoint   = true;
                ui.selectedPointIndex = bestIndex;
            }
        }
    }

    leftMousePressedLastFrame = (leftState == GLFW_PRESS);
}
