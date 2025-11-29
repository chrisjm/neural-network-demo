// Use GLAD as the OpenGL loader and prevent GLFW from including system GL headers.
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cmath>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "App.h"
#include "ShaderProgram.h"
#include "TriangleMesh.h"
#include "GeometryUtils.h"
#include "GLUtils.h"
#include "Object2D.h"
#include "Input.h"

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

    // Load OpenGL function pointers via GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[Init] Failed to initialize GLAD" << std::endl;
        return false;
    }

    check_gl_error("After GLAD initialization");

    // Query and log basic OpenGL information (now that GLAD is initialized)
    const GLubyte* renderer = glGetString(GL_RENDERER);
    const GLubyte* version = glGetString(GL_VERSION);
    std::cout << "[OpenGL] Renderer: " << (renderer ? reinterpret_cast<const char*>(renderer) : "<null>") << std::endl;
    std::cout << "[OpenGL] Version : " << (version ? reinterpret_cast<const char*>(version) : "<null>") << std::endl;

    int major = 0, minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);
    std::cout << "[OpenGL] Detected version " << major << "." << minor << std::endl;

    // Initialize Dear ImGui after OpenGL/GLAD are ready
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");  // matches your GL version

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

    Object2D initialObjects[2] = { objects[0], objects[1] };

    MouseDebugState mouseDebug{};

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

        // Keyboard and mouse input are handled in a separate module so we
        // can evolve interactions (e.g., drag-and-drop) independently of
        // the rendering and app structure.
        handleKeyboardInput(window, objects, 2, selectedObject, tabPressedLastFrame);
        handleMouseInput(window, objects, 2, selectedObject, leftMousePressedLastFrame, vertices, &mouseDebug);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Debug window with per-object state and controls.
        ImGui::Begin("Debug");

        const char* objectNames[2] = { "Triangle", "Square" };
        ImGui::Text("Selected: %s (%d)", objectNames[selectedObject], selectedObject);

        // Allow changing the selected object from the UI.
        if (ImGui::RadioButton("Triangle", selectedObject == 0)) {
            selectedObject = 0;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Square", selectedObject == 1)) {
            selectedObject = 1;
        }

        ImGui::Separator();

        for (int i = 0; i < 2; ++i) {
            Object2D& obj = objects[i];
            ImGui::PushID(i); // ensure widget IDs under this scope are unique per object
            if (ImGui::TreeNode(objectNames[i])) {
                if (ImGui::Button("Reset")) {
                    obj = initialObjects[i];
                }
                ImGui::Separator();
                ImGui::SliderFloat2("Offset", &obj.offsetX, -1.0f, 1.0f);
                ImGui::SliderFloat("Scale", &obj.scale, 0.1f, 2.0f);
                ImGui::SliderAngle("Rotation", &obj.rotation, -180.0f, 180.0f);
                ImGui::ColorEdit3("Color", obj.color);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }

        ImGui::Separator();

        ImGui::Text("Mouse debug");
        if (!mouseDebug.hasClick) {
            ImGui::Text("No click yet");
        } else {
            ImGui::Text("Window: (%.1f, %.1f)", mouseDebug.mouseX, mouseDebug.mouseY);
            ImGui::Text("NDC:    (%.3f, %.3f)", mouseDebug.xNdc, mouseDebug.yNdc);
            ImGui::Text("Triangle local: (%.3f, %.3f) %s",
                        mouseDebug.triLocalX, mouseDebug.triLocalY,
                        mouseDebug.hitTriangle ? "[hit]" : "[miss]");
            ImGui::Text("Square   local: (%.3f, %.3f) %s",
                        mouseDebug.squareLocalX, mouseDebug.squareLocalY,
                        mouseDebug.hitSquare ? "[hit]" : "[miss]");
        }

        ImGui::End();

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

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers (Double buffering prevents flickering)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &squareVAO);
    glDeleteBuffers(1, &squareVBO);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}
