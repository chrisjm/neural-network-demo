#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_GLCOREARB
#include <GLFW/glfw3.h>
#else
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#endif

#include <iostream>
#include <cmath>

#include "App.h"
#include "ShaderProgram.h"
#include "TriangleMesh.h"
#include "GeometryUtils.h"
#include "GLUtils.h"
#include "Object2D.h"

// ==========================================
// 1. THE SHADER SOURCE CODE (The "Recipe")
// ==========================================
// Usually these are in separate text files, but to show you they are just strings,
// we put them right here in the C++ code.

// VERTEX SHADER: The "Where" step.
// It takes a 3D position (aPos), applies scale and rotation around the triangle's center, then adds an offset.
const char *vertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "uniform vec2  uOffset;\n"
    "uniform float uScale;\n"
    "uniform float uRotation;\n"
    "void main()\n"
    "{\n"
    "   // Build a 2D transformation matrix from scale and rotation.\n"
    "   float c = cos(uRotation);\n"
    "   float s = sin(uRotation);\n"
    "   mat3 scaleMat = mat3(\n"
    "       uScale, 0.0,   0.0,\n"
    "       0.0,   uScale, 0.0,\n"
    "       0.0,   0.0,    1.0\n"
    "   );\n"
    "   mat3 rotMat = mat3(\n"
    "       c,  -s,  0.0,\n"
    "       s,   c,  0.0,\n"
    "       0.0, 0.0, 1.0\n"
    "   );\n"
    "   mat3 transform = rotMat * scaleMat;\n"
    "   // The triangle's center in its original coordinates is at (0, -1/6).\n"
    "   vec2 pivot = vec2(0.0, -0.1666667);\n"
    "   // Move into pivot space so we rotate around the center.\n"
    "   vec3 localPos = vec3(aPos.xy - pivot, 0.0);\n"
    "   vec3 rotatedScaled = transform * localPos;\n"
    "   // Move back out of pivot space and then apply the user-controlled offset.\n"
    "   vec2 worldPos2D = rotatedScaled.xy + pivot + uOffset;\n"
    "   gl_Position = vec4(worldPos2D, aPos.z, 1.0);\n"
    "}\0";

// FRAGMENT SHADER: The "Color" step.
// It outputs a color (FragColor) driven by a uniform (uColor) set from the CPU.
const char *fragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "uniform vec3 uColor;\n"
    "void main()\n"
    "{\n"
    "   // RGB comes from a uniform so we can change it with keyboard input.\n"
    "   FragColor = vec4(uColor, 1.0f);\n"
    "}\n\0";

// Geometry-related helpers such as worldToLocal, pointInTriangle, and
// pointInUnitSquare are provided by GeometryUtils.h/.cpp.

App::App()
    : m_window(nullptr) {
}

bool App::init() {
    // ==========================================
    // 2. INITIALIZATION (The OS Layer)
    // ==========================================
    // Initialize GLFW (Window Manager)
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "[Init] Failed to initialize GLFW" << std::endl;
        return false;
    }
    std::cout << "[Init] GLFW initialized" << std::endl;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    // macOS requires this for forward-compatible core profiles 3.2+
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create a Window
    std::cout << "[Init] Creating window 800x600..." << std::endl;
    m_window = glfwCreateWindow(800, 600, "My First Shader", NULL, NULL);
    if (m_window == NULL) {
        std::cerr << "[Init] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    std::cout << "[Init] Window created" << std::endl;
    glfwMakeContextCurrent(m_window);
    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);

    std::cout << "[Init] OpenGL context is now current" << std::endl;

    // Query and log basic OpenGL information
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    std::cout << "[OpenGL] Renderer: " << (renderer ? reinterpret_cast<const char*>(renderer) : "<null>") << std::endl;
    std::cout << "[OpenGL] Version : " << (version ? reinterpret_cast<const char*>(version) : "<null>") << std::endl;

    int major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    std::cout << "[OpenGL] Detected version " << major << "." << minor << std::endl;

    check_gl_error("After context creation");

    return true;
}

int App::run() {
    if (!m_window) {
        if (!init()) {
            return -1;
        }
    }

    GLFWwindow* window = m_window;

    // ==========================================
    // 3. BUILD SHADERS (Compiling Logic)
    // ==========================================

    // A. Compile Vertex Shader
    // B. Compile Fragment Shader
    // C. Link Shaders into a "Program"
    // This links the Vertex and Fragment stages into a single executable pipeline.
    ShaderProgram shaderProgram(vertexShaderSource, fragmentShaderSource);

    check_gl_error("After shader program link");

    // Query uniform locations so we can drive them from the CPU
    int offsetLocation   = glGetUniformLocation(shaderProgram.getId(), "uOffset");
    int colorLocation    = glGetUniformLocation(shaderProgram.getId(), "uColor");
    int scaleLocation    = glGetUniformLocation(shaderProgram.getId(), "uScale");
    int rotationLocation = glGetUniformLocation(shaderProgram.getId(), "uRotation");
    if (offsetLocation == -1) {
        std::cout << "[Uniform] Warning: uOffset not found (might be optimized out if unused)" << std::endl;
    } else {
        std::cout << "[Uniform] uOffset location = " << offsetLocation << std::endl;
    }
    if (colorLocation == -1) {
        std::cout << "[Uniform] Warning: uColor not found (might be optimized out if unused)" << std::endl;
    } else {
        std::cout << "[Uniform] uColor location  = " << colorLocation << std::endl;
    }
    if (scaleLocation == -1) {
        std::cout << "[Uniform] Warning: uScale not found (might be optimized out if unused)" << std::endl;
    } else {
        std::cout << "[Uniform] uScale location  = " << scaleLocation << std::endl;
    }
    if (rotationLocation == -1) {
        std::cout << "[Uniform] Warning: uRotation not found (might be optimized out if unused)" << std::endl;
    } else {
        std::cout << "[Uniform] uRotation location  = " << rotationLocation << std::endl;
    }

    // ==========================================
    // 4. LOAD ASSETS (Sending Mesh to VRAM)
    // ==========================================

    // The data: 3 vertices (X, Y, Z)
    float vertices[] = {
        -0.5f, -0.5f, 0.0f, // Left  (Bottom-Left)
         0.5f, -0.5f, 0.0f, // Right (Bottom-Right)
         0.0f,  0.5f, 0.0f  // Top   (Top-Center)
    };

    // A second mesh: a square made of two triangles (6 vertices)
    float squareVertices[] = {
        // First triangle
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.5f,  0.5f, 0.0f,
        // Second triangle
        -0.5f, -0.5f, 0.0f,
         0.5f,  0.5f, 0.0f,
        -0.5f,  0.5f, 0.0f
    };

    unsigned int VBO, VAO;
    unsigned int squareVBO, squareVAO;
    // Generate IDs for buffers
    // 1. Bind the Vertex Array Object first
    // 2. Copy our vertices array in a buffer for OpenGL to use
    // THIS IS THE MOMENT we move data from RAM -> VRAM
    // 3. Set the vertex attribute pointers
    // Tell the GPU how to interpret the raw binary data (It's 3 floats per vertex)
    TriangleMesh triangle(vertices, 3, VAO, VBO);
    TriangleMesh square(squareVertices, 6, squareVAO, squareVBO);

    check_gl_error("After VAO/VBO setup");

    // Initial CPU-side state for uniforms, now expressed per object.
    Object2D objects[2] = {
        { &triangle, 0.0f, 0.0f, 1.0f, 0.0f, {1.0f, 0.5f, 0.2f} }, // triangle
        { &square,   0.6f, 0.0f, 0.6f, 0.0f, {0.0f, 0.8f, 0.2f} }  // square
    };

    std::cout << "[State] Triangle initial offset   = (" << objects[0].offsetX << ", " << objects[0].offsetY << ")" << std::endl;
    std::cout << "[State] Triangle initial scale    = " << objects[0].scale << std::endl;
    std::cout << "[State] Triangle initial rotation = " << objects[0].rotation << " radians" << std::endl;
    std::cout << "[State] Triangle initial color    = (" << objects[0].color[0] << ", " << objects[0].color[1] << ", " << objects[0].color[2] << ")" << std::endl;

    std::cout << "[State] Square   initial offset   = (" << objects[1].offsetX << ", " << objects[1].offsetY << ")" << std::endl;
    std::cout << "[State] Square   initial scale    = " << objects[1].scale << std::endl;
    std::cout << "[State] Square   initial rotation = " << objects[1].rotation << " radians" << std::endl;
    std::cout << "[State] Square   initial color    = (" << objects[1].color[0] << ", " << objects[1].color[1] << ", " << objects[1].color[2] << ")" << std::endl;

    // Selection state: 0 = triangle, 1 = square.
    int selectedObject = 0;
    bool leftMousePressedLastFrame = false;
    bool tabPressedLastFrame = false;

    // ==========================================
    // 5. THE GAME LOOP
    // ==========================================
    std::cout << "[Loop] Entering render loop" << std::endl;
    while (!glfwWindowShouldClose(window)) {
        // Input
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        const float moveSpeed     = 0.01f; // how fast the triangle moves per frame when key is held
        const float scaleStep     = 0.01f; // how much scale changes per key press
        const float rotationStep  = 0.05f; // radians per key press (~3 degrees)

        // Allow cycling the active object with TAB.
        int tabState = glfwGetKey(window, GLFW_KEY_TAB);
        if (tabState == GLFW_PRESS && !tabPressedLastFrame) {
            selectedObject = (selectedObject + 1) % 2;
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

        // Mouse picking: on left-click, convert the mouse position from
        // window coordinates into world/clip space, then into each object's
        // local coordinates, and test against the original mesh shape.
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
                    vertices[0], vertices[1],
                    vertices[3], vertices[4],
                    vertices[6], vertices[7]
                );

                // Then test square.
                float squareLocalX = 0.0f, squareLocalY = 0.0f;
                worldToLocal(xNdc, yNdc,
                             objects[1].offsetX, objects[1].offsetY,
                             objects[1].scale, objects[1].rotation,
                             squareLocalX, squareLocalY);

                bool hitSquare = pointInUnitSquare(squareLocalX, squareLocalY);

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

        // Render Command 1: Clear the screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render Command 2: Use our compiled shader program
        shaderProgram.use();

        // Update uniforms every frame so the GPU sees the latest CPU-side values
        // Triangle color: brighten slightly if selected.
        float triColor[3] = { objects[0].color[0], objects[0].color[1], objects[0].color[2] };
        if (selectedObject == 0) {
            triColor[0] = std::min(1.0f, triColor[0] + 0.2f);
            triColor[1] = std::min(1.0f, triColor[1] + 0.2f);
            triColor[2] = std::min(1.0f, triColor[2] + 0.2f);
        }

        shaderProgram.setVec2(offsetLocation, objects[0].offsetX, objects[0].offsetY);
        shaderProgram.setVec3(colorLocation, triColor[0], triColor[1], triColor[2]);
        shaderProgram.setFloat(scaleLocation, objects[0].scale);
        shaderProgram.setFloat(rotationLocation, objects[0].rotation);

        // Render Command 3: Bind the mesh
        triangle.bind();

        // Render Command 4: DRAW!
        // "Draw 3 vertices starting from index 0"
        triangle.draw();

        // Second object: square (shares the same shader, but has its own
        // transform and color by using a second set of uniform values).
        float sqColor[3] = { objects[1].color[0], objects[1].color[1], objects[1].color[2] };
        if (selectedObject == 1) {
            sqColor[0] = std::min(1.0f, sqColor[0] + 0.2f);
            sqColor[1] = std::min(1.0f, sqColor[1] + 0.2f);
            sqColor[2] = std::min(1.0f, sqColor[2] + 0.2f);
        }

        shaderProgram.setVec2(offsetLocation, objects[1].offsetX, objects[1].offsetY);
        shaderProgram.setVec3(colorLocation, sqColor[0], sqColor[1], sqColor[2]);
        shaderProgram.setFloat(scaleLocation, objects[1].scale);
        shaderProgram.setFloat(rotationLocation, objects[1].rotation);

        square.bind();
        square.draw();

        // Swap buffers (Double buffering prevents flickering)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &squareVAO);
    glDeleteBuffers(1, &squareVBO);

    glfwTerminate();
    return 0;
}
