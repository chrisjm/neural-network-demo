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
#include "Input.h"

// ==========================================
// 1. THE SHADER SOURCE CODE (The "Recipe")
// ==========================================
// Shader programs are loaded from external text files in the shaders/ directory.
// See shaders/*.vert and shaders/*.frag for the GLSL source.

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
    std::string pointVertexSrc   = loadTextFile("shaders/point.vert");
    std::string pointFragmentSrc = loadTextFile("shaders/point.frag");
    ShaderProgram pointShader(pointVertexSrc.c_str(), pointFragmentSrc.c_str());

    check_gl_error("After point shader program link");

    int pointSizeLocation     = glGetUniformLocation(pointShader.getId(), "uPointSize");
    int colorClass0Location   = glGetUniformLocation(pointShader.getId(), "uColorClass0");
    int colorClass1Location   = glGetUniformLocation(pointShader.getId(), "uColorClass1");
    int selectedIndexLocation = glGetUniformLocation(pointShader.getId(), "uSelectedIndex");

    // Simple line shader for grid and axes.
    std::string gridVertexSrc   = loadTextFile("shaders/grid.vert");
    std::string gridFragmentSrc = loadTextFile("shaders/grid.frag");
    ShaderProgram gridShader(gridVertexSrc.c_str(), gridFragmentSrc.c_str());
    int gridColorLocation = glGetUniformLocation(gridShader.getId(), "uColor");

    // Shader for decision-boundary background field.
    std::string fieldVertexSrc   = loadTextFile("shaders/field.vert");
    std::string fieldFragmentSrc = loadTextFile("shaders/field.frag");
    ShaderProgram fieldShader(fieldVertexSrc.c_str(), fieldFragmentSrc.c_str());
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
    ui.datasetIndex       = static_cast<int>(currentDataset);
    ui.numPoints          = 1000;
    ui.spread             = 0.25f;
    ui.pointSize          = 6.0f;
    ui.probeEnabled       = true;
    ui.probeX             = 0.0f;
    ui.probeY             = 0.0f;
    ui.hasSelectedPoint   = false;
    ui.selectedPointIndex = -1;
    ui.selectedLabel      = -1;

    generateDataset(currentDataset, ui.numPoints, ui.spread, dataset);
    pointCloud.upload(dataset);

    const float gridStep = 0.25f;
    GridAxes gridAxes;
    gridAxes.init(gridStep);

    const int fieldResolution = 64;
    FieldVisualizer fieldVis;
    fieldVis.init(fieldResolution);

    Trainer trainer;

    bool leftMousePressedLastFrame = false;

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

        ImGuiIO& io = ImGui::GetIO();

        bool regenerate = false;
        bool stepTrainRequested = false;

        drawControlPanel(ui,
                         trainer,
                         dataset.size(),
                         regenerate,
                         stepTrainRequested);

        handleProbeSelection(window, dataset, ui, leftMousePressedLastFrame, io.WantCaptureMouse);

        if (regenerate) {
            if (ui.numPoints < 10) ui.numPoints = 10;
            if (ui.numPoints > maxPoints) ui.numPoints = maxPoints;
            currentDataset = static_cast<DatasetType>(ui.datasetIndex);
            generateDataset(currentDataset, ui.numPoints, ui.spread, dataset);
            pointCloud.upload(dataset);

            ui.hasSelectedPoint   = false;
            ui.selectedPointIndex = -1;
            ui.selectedLabel      = -1;

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
        const auto& W1 = trainer.net.getW1();
        const auto& B1 = trainer.net.getB1();
        const auto& W2 = trainer.net.getW2();
        const auto& B2 = trainer.net.getB2();
        const auto& W3 = trainer.net.getW3();
        const auto& B3 = trainer.net.getB3();

        fieldShader.setFloatArray(fieldW1Location, W1.data(), static_cast<int>(W1.size()));
        fieldShader.setFloatArray(fieldB1Location, B1.data(), static_cast<int>(B1.size()));
        fieldShader.setFloatArray(fieldW2Location, W2.data(), static_cast<int>(W2.size()));
        fieldShader.setFloatArray(fieldB2Location, B2.data(), static_cast<int>(B2.size()));
        fieldShader.setFloatArray(fieldW3Location, W3.data(), static_cast<int>(W3.size()));
        fieldShader.setFloatArray(fieldB3Location, B3.data(), static_cast<int>(B3.size()));
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
        if (selectedIndexLocation != -1) {
            int selIndex = -1;
            if (ui.hasSelectedPoint && ui.selectedPointIndex >= 0 &&
                ui.selectedPointIndex < static_cast<int>(dataset.size())) {
                selIndex = ui.selectedPointIndex;
            }
            pointShader.setInt(selectedIndexLocation, selIndex);
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
