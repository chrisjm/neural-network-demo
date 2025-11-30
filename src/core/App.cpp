// Use GLES3 headers on Emscripten.
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define GLFW_INCLUDE_ES3
#include <GLFW/glfw3.h>
#else
#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#endif

#include <iostream>
#include <cmath>
#include <vector>
#include <optional>
#include <memory>

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

#ifdef __EMSCRIPTEN__
// Persistent render state for the WebAssembly build. This lives in App.cpp so
// that App::run() can return after registering the main loop, while the data
// stays alive for subsequent frames.
struct WasmSceneState {
    UiState ui;
    std::vector<DataPoint> dataset;
    PointCloud pointCloud;
    GridAxes gridAxes;
    FieldVisualizer fieldVis;
    Trainer trainer;

    bool leftMousePressedLastFrame = false;
    int  maxPoints                 = 0;

    std::unique_ptr<ShaderProgram> pointShader;
    std::unique_ptr<ShaderProgram> gridShader;
    std::unique_ptr<ShaderProgram> fieldShader;

    int pointSizeLocation     = -1;
    int colorClass0Location   = -1;
    int colorClass1Location   = -1;
    int selectedIndexLocation = -1;

    int gridColorLocation = -1;

    int fieldW1Location = -1;
    int fieldB1Location = -1;
    int fieldW2Location = -1;
    int fieldB2Location = -1;
    int fieldW3Location = -1;
    int fieldB3Location = -1;
};

static WasmSceneState g_wasmState;
static App* g_wasmApp = nullptr;

static void wasm_main_loop();
#endif

struct FrameContext;
static void updateAndRenderFrame(FrameContext& ctx);

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

    // Load OpenGL function pointers via GLAD on native platforms.
#ifndef __EMSCRIPTEN__
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[Init] Failed to initialize GLAD" << std::endl;
        return false;
    }

    glEnable(GL_PROGRAM_POINT_SIZE);
#endif
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
#ifdef __EMSCRIPTEN__
    ImGui_ImplOpenGL3_Init("#version 300 es");
#else
    ImGui_ImplOpenGL3_Init("#version 330");  // matches your GL version
#endif

    return true;
}

void App::shutdownScene(PointCloud& pointCloud,
                        GridAxes& gridAxes,
                        FieldVisualizer& fieldVis) {
    pointCloud.shutdown();
    gridAxes.shutdown();
    fieldVis.shutdown();
}

void App::shutdownApp() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    m_window = nullptr;
}

static void initSceneCommon(DatasetType currentDataset,
                            UiState& ui,
                            int& maxPoints,
                            std::vector<DataPoint>& dataset,
                            PointCloud& pointCloud,
                            GridAxes& gridAxes,
                            FieldVisualizer& fieldVis,
                            bool& leftMousePressedLastFrame) {
    dataset.clear();

    maxPoints = 5000;
    pointCloud.init(maxPoints);

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
    gridAxes.init(gridStep);

    const int fieldResolution = 64;
    fieldVis.init(fieldResolution);

    leftMousePressedLastFrame = false;
}

#ifdef __EMSCRIPTEN__
static bool initShadersWasm(WasmSceneState& state) {
    auto pointVertexSrc   = loadTextFile("shaders/point_es.vert");
    auto pointFragmentSrc = loadTextFile("shaders/point_es.frag");
    if (!pointVertexSrc || !pointFragmentSrc) {
        std::cerr << "[Init] Failed to load point shader sources" << std::endl;
        return false;
    }
    state.pointShader = std::make_unique<ShaderProgram>(pointVertexSrc->c_str(), pointFragmentSrc->c_str());

    check_gl_error("After point shader program link");

    state.pointSizeLocation     = glGetUniformLocation(state.pointShader->getId(), "uPointSize");
    state.colorClass0Location   = glGetUniformLocation(state.pointShader->getId(), "uColorClass0");
    state.colorClass1Location   = glGetUniformLocation(state.pointShader->getId(), "uColorClass1");
    state.selectedIndexLocation = glGetUniformLocation(state.pointShader->getId(), "uSelectedIndex");

    auto gridVertexSrc   = loadTextFile("shaders/grid_es.vert");
    auto gridFragmentSrc = loadTextFile("shaders/grid_es.frag");
    if (!gridVertexSrc || !gridFragmentSrc) {
        std::cerr << "[Init] Failed to load grid shader sources" << std::endl;
        return false;
    }
    state.gridShader = std::make_unique<ShaderProgram>(gridVertexSrc->c_str(), gridFragmentSrc->c_str());
    state.gridColorLocation = glGetUniformLocation(state.gridShader->getId(), "uColor");

    auto fieldVertexSrc   = loadTextFile("shaders/field_es.vert");
    auto fieldFragmentSrc = loadTextFile("shaders/field_es.frag");
    if (!fieldVertexSrc || !fieldFragmentSrc) {
        std::cerr << "[Init] Failed to load field shader sources" << std::endl;
        return false;
    }
    state.fieldShader = std::make_unique<ShaderProgram>(fieldVertexSrc->c_str(), fieldFragmentSrc->c_str());
    state.fieldW1Location = glGetUniformLocation(state.fieldShader->getId(), "u_W1");
    state.fieldB1Location = glGetUniformLocation(state.fieldShader->getId(), "u_b1");
    state.fieldW2Location = glGetUniformLocation(state.fieldShader->getId(), "u_W2");
    state.fieldB2Location = glGetUniformLocation(state.fieldShader->getId(), "u_b2");
    state.fieldW3Location = glGetUniformLocation(state.fieldShader->getId(), "u_W3");
    state.fieldB3Location = glGetUniformLocation(state.fieldShader->getId(), "u_b3");

    return true;
}
#else
static bool initShadersDesktop(std::unique_ptr<ShaderProgram>& pointShader,
                               std::unique_ptr<ShaderProgram>& gridShader,
                               std::unique_ptr<ShaderProgram>& fieldShader,
                               int& pointSizeLocation,
                               int& colorClass0Location,
                               int& colorClass1Location,
                               int& selectedIndexLocation,
                               int& gridColorLocation,
                               int& fieldW1Location,
                               int& fieldB1Location,
                               int& fieldW2Location,
                               int& fieldB2Location,
                               int& fieldW3Location,
                               int& fieldB3Location) {
    auto pointVertexSrc   = loadTextFile("shaders/point.vert");
    auto pointFragmentSrc = loadTextFile("shaders/point.frag");
    if (!pointVertexSrc || !pointFragmentSrc) {
        std::cerr << "[Init] Failed to load point shader sources" << std::endl;
        return false;
    }
    pointShader = std::make_unique<ShaderProgram>(pointVertexSrc->c_str(), pointFragmentSrc->c_str());

    check_gl_error("After point shader program link");

    pointSizeLocation     = glGetUniformLocation(pointShader->getId(), "uPointSize");
    colorClass0Location   = glGetUniformLocation(pointShader->getId(), "uColorClass0");
    colorClass1Location   = glGetUniformLocation(pointShader->getId(), "uColorClass1");
    selectedIndexLocation = glGetUniformLocation(pointShader->getId(), "uSelectedIndex");

    auto gridVertexSrc   = loadTextFile("shaders/grid.vert");
    auto gridFragmentSrc = loadTextFile("shaders/grid.frag");
    if (!gridVertexSrc || !gridFragmentSrc) {
        std::cerr << "[Init] Failed to load grid shader sources" << std::endl;
        return false;
    }
    gridShader = std::make_unique<ShaderProgram>(gridVertexSrc->c_str(), gridFragmentSrc->c_str());
    gridColorLocation = glGetUniformLocation(gridShader->getId(), "uColor");

    auto fieldVertexSrc   = loadTextFile("shaders/field.vert");
    auto fieldFragmentSrc = loadTextFile("shaders/field.frag");
    if (!fieldVertexSrc || !fieldFragmentSrc) {
        std::cerr << "[Init] Failed to load field shader sources" << std::endl;
        return false;
    }
    fieldShader = std::make_unique<ShaderProgram>(fieldVertexSrc->c_str(), fieldFragmentSrc->c_str());
    fieldW1Location = glGetUniformLocation(fieldShader->getId(), "u_W1");
    fieldB1Location = glGetUniformLocation(fieldShader->getId(), "u_b1");
    fieldW2Location = glGetUniformLocation(fieldShader->getId(), "u_W2");
    fieldB2Location = glGetUniformLocation(fieldShader->getId(), "u_b2");
    fieldW3Location = glGetUniformLocation(fieldShader->getId(), "u_W3");
    fieldB3Location = glGetUniformLocation(fieldShader->getId(), "u_b3");

    return true;
}
#endif

int App::run() {
    if (!m_window) {
        if (!init()) {
            return -1;
        }
    }

    GLFWwindow* window = m_window;

#ifdef __EMSCRIPTEN__
    // ==========================================
    // 3. BUILD SHADERS (Compiling Logic)
    // ==========================================

    if (!initShadersWasm(g_wasmState)) {
        shutdownApp();
        return -1;
    }

    // ==========================================
    // 4. LOAD ASSETS (Sending Mesh to VRAM)
    // ==========================================

    DatasetType currentDataset = DatasetType::TwoBlobs;
    initSceneCommon(currentDataset,
                    g_wasmState.ui,
                    g_wasmState.maxPoints,
                    g_wasmState.dataset,
                    g_wasmState.pointCloud,
                    g_wasmState.gridAxes,
                    g_wasmState.fieldVis,
                    g_wasmState.leftMousePressedLastFrame);

    g_wasmApp = this;
    emscripten_set_main_loop(wasm_main_loop, 0, 1);
    return 0;
#else
    // ==========================================
    // 3. BUILD SHADERS (Compiling Logic)
    // ==========================================

    std::unique_ptr<ShaderProgram> pointShader;
    std::unique_ptr<ShaderProgram> gridShader;
    std::unique_ptr<ShaderProgram> fieldShader;

    int pointSizeLocation     = -1;
    int colorClass0Location   = -1;
    int colorClass1Location   = -1;
    int selectedIndexLocation = -1;
    int gridColorLocation     = -1;
    int fieldW1Location       = -1;
    int fieldB1Location       = -1;
    int fieldW2Location       = -1;
    int fieldB2Location       = -1;
    int fieldW3Location       = -1;
    int fieldB3Location       = -1;

    if (!initShadersDesktop(pointShader,
                            gridShader,
                            fieldShader,
                            pointSizeLocation,
                            colorClass0Location,
                            colorClass1Location,
                            selectedIndexLocation,
                            gridColorLocation,
                            fieldW1Location,
                            fieldB1Location,
                            fieldW2Location,
                            fieldB2Location,
                            fieldW3Location,
                            fieldB3Location)) {
        shutdownApp();
        return -1;
    }

    // ==========================================
    // 4. LOAD ASSETS (Sending Mesh to VRAM)
    // ==========================================

    // Dataset of 2D points with class labels.
    std::vector<DataPoint> dataset;

    int maxPoints = 0;
    PointCloud pointCloud;
    GridAxes gridAxes;
    FieldVisualizer fieldVis;
    Trainer trainer;
    bool leftMousePressedLastFrame = false;

    DatasetType currentDataset = DatasetType::TwoBlobs;
    UiState ui;

    initSceneCommon(currentDataset,
                    ui,
                    maxPoints,
                    dataset,
                    pointCloud,
                    gridAxes,
                    fieldVis,
                    leftMousePressedLastFrame);

    // ==========================================
    // 5. THE GAME LOOP
    // ==========================================
    renderLoop(window,
               *pointShader,
               *gridShader,
               *fieldShader,
               pointSizeLocation,
               colorClass0Location,
               colorClass1Location,
               selectedIndexLocation,
               gridColorLocation,
               fieldW1Location,
               fieldB1Location,
               fieldW2Location,
               fieldB2Location,
               fieldW3Location,
               fieldB3Location,
               ui,
               dataset,
               pointCloud,
               gridAxes,
               fieldVis,
               trainer,
               leftMousePressedLastFrame,
               maxPoints);

    shutdownScene(pointCloud, gridAxes, fieldVis);
    shutdownApp();
    return 0;
#endif
}

 #ifdef __EMSCRIPTEN__
 static void wasm_main_loop() {
     if (!g_wasmApp || !g_wasmApp->getWindow()) {
         return;
     }

     GLFWwindow* window = g_wasmApp->getWindow();

     FrameContext ctx{
         window,
         *g_wasmState.pointShader,
         *g_wasmState.gridShader,
         *g_wasmState.fieldShader,
         g_wasmState.pointSizeLocation,
         g_wasmState.colorClass0Location,
         g_wasmState.colorClass1Location,
         g_wasmState.selectedIndexLocation,
         g_wasmState.gridColorLocation,
         g_wasmState.fieldW1Location,
         g_wasmState.fieldB1Location,
         g_wasmState.fieldW2Location,
         g_wasmState.fieldB2Location,
         g_wasmState.fieldW3Location,
         g_wasmState.fieldB3Location,
         g_wasmState.ui,
         g_wasmState.dataset,
         g_wasmState.pointCloud,
         g_wasmState.gridAxes,
         g_wasmState.fieldVis,
         g_wasmState.trainer,
         g_wasmState.leftMousePressedLastFrame,
         g_wasmState.maxPoints};

     updateAndRenderFrame(ctx);
 }
 #endif

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

 static void updateAndRenderFrame(FrameContext& ctx) {
     ImGui_ImplOpenGL3_NewFrame();
     ImGui_ImplGlfw_NewFrame();
     ImGui::NewFrame();

     ImGuiIO& io = ImGui::GetIO();

     bool regenerate = false;
     bool stepTrainRequested = false;

     drawControlPanel(ctx.ui,
                      ctx.trainer,
                      ctx.dataset.size(),
                      regenerate,
                      stepTrainRequested);

     handleProbeSelection(ctx.window,
                          ctx.dataset,
                          ctx.ui,
                          ctx.leftMousePressedLastFrame,
                          io.WantCaptureMouse);

     if (regenerate) {
         if (ctx.ui.numPoints < 10) ctx.ui.numPoints = 10;
         if (ctx.ui.numPoints > ctx.maxPoints) ctx.ui.numPoints = ctx.maxPoints;
         DatasetType currentDataset = static_cast<DatasetType>(ctx.ui.datasetIndex);
         generateDataset(currentDataset,
                         ctx.ui.numPoints,
                         ctx.ui.spread,
                         ctx.dataset);
         ctx.pointCloud.upload(ctx.dataset);

         ctx.ui.hasSelectedPoint   = false;
         ctx.ui.selectedPointIndex = -1;
         ctx.ui.selectedLabel      = -1;

         ctx.trainer.resetForNewDataset();
         ctx.fieldVis.setDirty();
     }

     if (stepTrainRequested) {
         ctx.trainer.stepOnce(ctx.dataset);
         ctx.fieldVis.setDirty();
     }

     if (ctx.trainer.autoTrain) {
         if (ctx.trainer.stepAuto(ctx.dataset)) {
             ctx.fieldVis.setDirty();
         }
     }

     if (ctx.fieldVis.isDirty()) {
         ctx.fieldVis.update();
     }

     glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
     glClear(GL_COLOR_BUFFER_BIT);

     ctx.fieldShader.use();
     const auto& W1 = ctx.trainer.net.getW1();
     const auto& B1 = ctx.trainer.net.getB1();
     const auto& W2 = ctx.trainer.net.getW2();
     const auto& B2 = ctx.trainer.net.getB2();
     const auto& W3 = ctx.trainer.net.getW3();
     const auto& B3 = ctx.trainer.net.getB3();

     ctx.fieldShader.setFloatArray(ctx.fieldW1Location, W1.data(), static_cast<int>(W1.size()));
     ctx.fieldShader.setFloatArray(ctx.fieldB1Location, B1.data(), static_cast<int>(B1.size()));
     ctx.fieldShader.setFloatArray(ctx.fieldW2Location, W2.data(), static_cast<int>(W2.size()));
     ctx.fieldShader.setFloatArray(ctx.fieldB2Location, B2.data(), static_cast<int>(B2.size()));
     ctx.fieldShader.setFloatArray(ctx.fieldW3Location, W3.data(), static_cast<int>(W3.size()));
     ctx.fieldShader.setFloatArray(ctx.fieldB3Location, B3.data(), static_cast<int>(B3.size()));
     ctx.fieldVis.draw();

     ctx.gridShader.use();
     if (ctx.gridColorLocation != -1) {
         ctx.gridShader.setVec3(ctx.gridColorLocation, 0.15f, 0.15f, 0.15f);
     }
     ctx.gridAxes.drawGrid();

     if (ctx.gridColorLocation != -1) {
         ctx.gridShader.setVec3(ctx.gridColorLocation, 0.8f, 0.8f, 0.8f);
     }
     ctx.gridAxes.drawAxes();

     ctx.pointShader.use();

     if (ctx.pointSizeLocation != -1) {
         ctx.pointShader.setFloat(ctx.pointSizeLocation, ctx.ui.pointSize);
     }
     if (ctx.colorClass0Location != -1) {
         ctx.pointShader.setVec3(ctx.colorClass0Location, 0.2f, 0.6f, 1.0f);
     }
     if (ctx.colorClass1Location != -1) {
         ctx.pointShader.setVec3(ctx.colorClass1Location, 1.0f, 0.5f, 0.2f);
     }
     if (ctx.selectedIndexLocation != -1) {
         int selIndex = -1;
         if (ctx.ui.hasSelectedPoint &&
             ctx.ui.selectedPointIndex >= 0 &&
             ctx.ui.selectedPointIndex < static_cast<int>(ctx.dataset.size())) {
             selIndex = ctx.ui.selectedPointIndex;
         }
         ctx.pointShader.setInt(ctx.selectedIndexLocation, selIndex);
     }

     ctx.pointCloud.draw(ctx.dataset.size());

     ImGui::Render();
     ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

     glfwSwapBuffers(ctx.window);
     glfwPollEvents();
 }

void App::renderLoop(GLFWwindow* window,
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
                     int maxPoints) {
    std::cout << "[Loop] Entering render loop" << std::endl;

    FrameContext ctx{
        window,
        pointShader,
        gridShader,
        fieldShader,
        pointSizeLocation,
        colorClass0Location,
        colorClass1Location,
        selectedIndexLocation,
        gridColorLocation,
        fieldW1Location,
        fieldB1Location,
        fieldW2Location,
        fieldB2Location,
        fieldW3Location,
        fieldB3Location,
        ui,
        dataset,
        pointCloud,
        gridAxes,
        fieldVis,
        trainer,
        leftMousePressedLastFrame,
        maxPoints};

    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        updateAndRenderFrame(ctx);
    }
}
