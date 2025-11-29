// Use GLAD as the OpenGL loader and prevent GLFW from including system GL headers.
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <cmath>
#include <vector>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include "App.h"
#include "ShaderProgram.h"
#include "GLUtils.h"
#include "DataPoint.h"
#include "DatasetGenerator.h"
#include "FieldVisualizer.h"
#include "PlotGeometry.h"
#include "Trainer.h"
#include "ControlPanel.h"

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

// Point sprite shaders for scatter plot rendering.
const char *pointVertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec2 aPos;\n"
    "layout (location = 1) in float aLabel;\n"
    "flat out int vLabel;\n"
    "uniform float uPointSize;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "    vLabel = int(aLabel + 0.5);\n"
    "    gl_PointSize = uPointSize;\n"
    "}\n\0";

const char *pointFragmentShaderSource = "#version 330 core\n"
    "flat in int vLabel;\n"
    "out vec4 FragColor;\n"
    "uniform vec3 uColorClass0;\n"
    "uniform vec3 uColorClass1;\n"
    "void main()\n"
    "{\n"
    "    vec2 d = gl_PointCoord - vec2(0.5);\n"
    "    if (dot(d, d) > 0.25) discard;\n"
    "    vec3 color = (vLabel == 0) ? uColorClass0 : uColorClass1;\n"
    "    FragColor = vec4(color, 1.0);\n"
    "}\n\0";

const char *gridVertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec2 aPos;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "}\n\0";

const char *gridFragmentShaderSource = "#version 330 core\n"
    "out vec4 FragColor;\n"
    "uniform vec3 uColor;\n"
    "void main()\n"
    "{\n"
    "    FragColor = vec4(uColor, 1.0);\n"
    "}\n\0";

const char *fieldVertexShaderSource = "#version 330 core\n"
    "layout (location = 0) in vec2 aPos;\n"
    "out vec2 vPos;\n"
    "void main()\n"
    "{\n"
    "    vPos = aPos;\n"
    "    gl_Position = vec4(aPos, 0.0, 1.0);\n"
    "}\n\0";

const char *fieldFragmentShaderSource = "#version 330 core\n"
    "in vec2 vPos;\n"
    "out vec4 FragColor;\n"
    "const int INPUT_DIM  = 2;\n"
    "const int HIDDEN1    = 4;\n"
    "const int HIDDEN2    = 4;\n"
    "const int OUTPUT_DIM = 2;\n"
    "uniform float u_W1[HIDDEN1 * INPUT_DIM];\n"
    "uniform float u_b1[HIDDEN1];\n"
    "uniform float u_W2[HIDDEN2 * HIDDEN1];\n"
    "uniform float u_b2[HIDDEN2];\n"
    "uniform float u_W3[OUTPUT_DIM * HIDDEN2];\n"
    "uniform float u_b3[OUTPUT_DIM];\n"
    "void main()\n"
    "{\n"
    "    float a0[INPUT_DIM];\n"
    "    a0[0] = vPos.x;\n"
    "    a0[1] = vPos.y;\n"
    "    float a1[HIDDEN1];\n"
    "    for (int j = 0; j < HIDDEN1; ++j) {\n"
    "        float sum = u_b1[j];\n"
    "        for (int i = 0; i < INPUT_DIM; ++i) {\n"
    "            int idx = j * INPUT_DIM + i;\n"
    "            sum += u_W1[idx] * a0[i];\n"
    "        }\n"
    "        a1[j] = max(sum, 0.0);\n"
    "    }\n"
    "    float a2[HIDDEN2];\n"
    "    for (int j = 0; j < HIDDEN2; ++j) {\n"
    "        float sum = u_b2[j];\n"
    "        for (int i = 0; i < HIDDEN1; ++i) {\n"
    "            int idx = j * HIDDEN1 + i;\n"
    "            sum += u_W2[idx] * a1[i];\n"
    "        }\n"
    "        a2[j] = max(sum, 0.0);\n"
    "    }\n"
    "    float logits[OUTPUT_DIM];\n"
    "    float maxLogit = -1e30;\n"
    "    for (int k = 0; k < OUTPUT_DIM; ++k) {\n"
    "        float sum = u_b3[k];\n"
    "        for (int j = 0; j < HIDDEN2; ++j) {\n"
    "            int idx = k * HIDDEN2 + j;\n"
    "            sum += u_W3[idx] * a2[j];\n"
    "        }\n"
    "        logits[k] = sum;\n"
    "        if (sum > maxLogit) maxLogit = sum;\n"
    "    }\n"
    "    float expSum = 0.0;\n"
    "    float probs[OUTPUT_DIM];\n"
    "    for (int k = 0; k < OUTPUT_DIM; ++k) {\n"
    "        float e = exp(logits[k] - maxLogit);\n"
    "        probs[k] = e;\n"
    "        expSum += e;\n"
    "    }\n"
    "    if (expSum <= 0.0) {\n"
    "        FragColor = vec4(0.5, 0.5, 0.5, 0.4);\n"
    "        return;\n"
    "    }\n"
    "    for (int k = 0; k < OUTPUT_DIM; ++k) {\n"
    "        probs[k] /= expSum;\n"
    "    }\n"
    "    float p1 = probs[1];\n"
    "    vec3 c0 = vec3(0.2, 0.6, 1.0);\n"
    "    vec3 c1 = vec3(1.0, 0.5, 0.2);\n"
    "    vec3 color = mix(c0, c1, p1);\n"
    "    FragColor = vec4(color, 0.4);\n"
    "}\n\0";

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
    m_window = glfwCreateWindow(1024, 768, "Neural Net Demo", NULL, NULL);
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

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

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

    // Point sprite shader program for scatter plot rendering.
    ShaderProgram pointShader(pointVertexShaderSource, pointFragmentShaderSource);

    check_gl_error("After point shader program link");

    int pointSizeLocation    = glGetUniformLocation(pointShader.getId(), "uPointSize");
    int colorClass0Location  = glGetUniformLocation(pointShader.getId(), "uColorClass0");
    int colorClass1Location  = glGetUniformLocation(pointShader.getId(), "uColorClass1");

    // Simple line shader for grid and axes.
    ShaderProgram gridShader(gridVertexShaderSource, gridFragmentShaderSource);
    int gridColorLocation = glGetUniformLocation(gridShader.getId(), "uColor");

    // Shader for decision-boundary background field.
    ShaderProgram fieldShader(fieldVertexShaderSource, fieldFragmentShaderSource);
    int fieldW1Location = glGetUniformLocation(fieldShader.getId(), "u_W1");
    int fieldB1Location = glGetUniformLocation(fieldShader.getId(), "u_b1");
    int fieldW2Location = glGetUniformLocation(fieldShader.getId(), "u_W2");
    int fieldB2Location = glGetUniformLocation(fieldShader.getId(), "u_b2");
    int fieldW3Location = glGetUniformLocation(fieldShader.getId(), "u_W3");
    int fieldB3Location = glGetUniformLocation(fieldShader.getId(), "u_b3");

    // ==========================================
    // 4. LOAD ASSETS (Sending Mesh to VRAM)
    // ==========================================

    // Dataset of 2D points with class labels.
    std::vector<DataPoint> dataset;

    const int maxPoints = 5000;
    PointCloud pointCloud;
    pointCloud.init(maxPoints);

    DatasetType currentDataset = DatasetType::TwoBlobs;

    UiState ui;
    ui.datasetIndex = static_cast<int>(currentDataset);
    ui.numPoints    = 1000;
    ui.spread       = 0.25f;
    ui.pointSize    = 6.0f;

    generateDataset(currentDataset, ui.numPoints, ui.spread, dataset);
    pointCloud.upload(dataset);

    const float gridStep = 0.25f;
    GridAxes gridAxes;
    gridAxes.init(gridStep);

    const int fieldResolution = 64;
    FieldVisualizer fieldVis;
    fieldVis.init(fieldResolution);

    Trainer trainer;

    // ==========================================
    // 5. THE GAME LOOP
    // ==========================================
    std::cout << "[Loop] Entering render loop" << std::endl;
    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        bool regenerate = false;
        bool stepTrainRequested = false;

        drawControlPanel(ui,
                         trainer,
                         dataset.size(),
                         regenerate,
                         stepTrainRequested);

        if (regenerate) {
            if (ui.numPoints < 10) ui.numPoints = 10;
            if (ui.numPoints > maxPoints) ui.numPoints = maxPoints;
            currentDataset = static_cast<DatasetType>(ui.datasetIndex);
            generateDataset(currentDataset, ui.numPoints, ui.spread, dataset);
            pointCloud.upload(dataset);

            trainer.resetForNewDataset();
            fieldVis.setDirty();
        }

        if (stepTrainRequested) {
            trainer.stepOnce(dataset);
            fieldVis.setDirty();
        }

        if (trainer.autoTrain) {
            if (trainer.stepAuto(dataset)) {
                fieldVis.setDirty();
            }
        }

        if (fieldVis.isDirty()) {
            fieldVis.update();
        }

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        fieldShader.use();
        fieldShader.setFloatArray(fieldW1Location, trainer.net.W1.data(), static_cast<int>(trainer.net.W1.size()));
        fieldShader.setFloatArray(fieldB1Location, trainer.net.b1.data(), static_cast<int>(trainer.net.b1.size()));
        fieldShader.setFloatArray(fieldW2Location, trainer.net.W2.data(), static_cast<int>(trainer.net.W2.size()));
        fieldShader.setFloatArray(fieldB2Location, trainer.net.b2.data(), static_cast<int>(trainer.net.b2.size()));
        fieldShader.setFloatArray(fieldW3Location, trainer.net.W3.data(), static_cast<int>(trainer.net.W3.size()));
        fieldShader.setFloatArray(fieldB3Location, trainer.net.b3.data(), static_cast<int>(trainer.net.b3.size()));
        fieldVis.draw();

        gridShader.use();
        if (gridColorLocation != -1) {
            gridShader.setVec3(gridColorLocation, 0.15f, 0.15f, 0.15f);
        }
        gridAxes.drawGrid();

        if (gridColorLocation != -1) {
            gridShader.setVec3(gridColorLocation, 0.8f, 0.8f, 0.8f);
        }
        gridAxes.drawAxes();

        pointShader.use();

        if (pointSizeLocation != -1) {
            pointShader.setFloat(pointSizeLocation, ui.pointSize);
        }
        if (colorClass0Location != -1) {
            pointShader.setVec3(colorClass0Location, 0.2f, 0.6f, 1.0f);
        }
        if (colorClass1Location != -1) {
            pointShader.setVec3(colorClass1Location, 1.0f, 0.5f, 0.2f);
        }

        pointCloud.draw(dataset.size());

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    pointCloud.shutdown();
    gridAxes.shutdown();
    fieldVis.shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}
