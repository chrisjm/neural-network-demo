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
#include "ToyNet.h"

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

    glEnable(GL_PROGRAM_POINT_SIZE);

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

    // ==========================================
    // 4. LOAD ASSETS (Sending Mesh to VRAM)
    // ==========================================

    // Dataset of 2D points with class labels.
    std::vector<DataPoint> dataset;

    auto generateTwoBlobs = [](int numPoints, float spread, std::vector<DataPoint>& out) {
        out.clear();
        out.reserve(static_cast<size_t>(numPoints));

        int half = numPoints / 2;
        auto rand01 = []() {
            return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        };

        for (int i = 0; i < half; ++i) {
            float angle = rand01() * 2.0f * static_cast<float>(M_PI);
            float radius = spread * rand01();
            float cx = -0.5f;
            float cy = 0.0f;
            float x = cx + std::cos(angle) * radius;
            float y = cy + std::sin(angle) * radius;
            out.push_back({x, y, 0});
        }

        for (int i = half; i < numPoints; ++i) {
            float angle = rand01() * 2.0f * static_cast<float>(M_PI);
            float radius = spread * rand01();
            float cx = 0.5f;
            float cy = 0.0f;
            float x = cx + std::cos(angle) * radius;
            float y = cy + std::sin(angle) * radius;
            out.push_back({x, y, 1});
        }
    };

    unsigned int pointVAO = 0;
    unsigned int pointVBO = 0;
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);

    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);

    int maxPoints = 5000;
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(maxPoints * 3 * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(0));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    glBindVertexArray(0);

    auto uploadDatasetToGPU = [&](const std::vector<DataPoint>& data) {
        std::vector<float> buffer;
        buffer.reserve(data.size() * 3);
        for (const auto& p : data) {
            buffer.push_back(p.x);
            buffer.push_back(p.y);
            buffer.push_back(static_cast<float>(p.label));
        }
        glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(buffer.size() * sizeof(float)), buffer.data());
    };

    int   uiNumPoints = 1000;
    float uiSpread    = 0.25f;
    float uiPointSize = 6.0f;

    generateTwoBlobs(uiNumPoints, uiSpread, dataset);
    uploadDatasetToGPU(dataset);

    std::vector<float> gridVertices;
    const float gridStep = 0.25f;

    for (float x = -1.0f; x <= 1.0001f; x += gridStep) {
        gridVertices.push_back(x);
        gridVertices.push_back(-1.0f);
        gridVertices.push_back(x);
        gridVertices.push_back(1.0f);
    }

    for (float y = -1.0f; y <= 1.0001f; y += gridStep) {
        gridVertices.push_back(-1.0f);
        gridVertices.push_back(y);
        gridVertices.push_back(1.0f);
        gridVertices.push_back(y);
    }

    std::vector<float> axisVertices = {
        -1.0f,  0.0f,  1.0f,  0.0f,
         0.0f, -1.0f,  0.0f,  1.0f
    };

    unsigned int gridVAO = 0;
    unsigned int gridVBO = 0;
    glGenVertexArrays(1, &gridVAO);
    glGenBuffers(1, &gridVBO);

    glBindVertexArray(gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, gridVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(gridVertices.size() * sizeof(float)),
                 gridVertices.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0));
    glBindVertexArray(0);

    unsigned int axisVAO = 0;
    unsigned int axisVBO = 0;
    glGenVertexArrays(1, &axisVAO);
    glGenBuffers(1, &axisVBO);

    glBindVertexArray(axisVAO);
    glBindBuffer(GL_ARRAY_BUFFER, axisVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(axisVertices.size() * sizeof(float)),
                 axisVertices.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0));
    glBindVertexArray(0);

    ToyNet net;
    float uiLearningRate   = 0.1f;
    int   uiBatchSize      = 64;
    bool  uiAutoTrain      = false;
    int   uiStepCount      = 0;
    float uiLastLoss       = 0.0f;
    float uiLastAccuracy   = 0.0f;
    int   uiAutoMaxSteps   = 2000;
    float uiAutoTargetLoss = 0.01f;

    std::vector<DataPoint> trainBatch;
    trainBatch.reserve(ToyNet::MaxBatch);
    int dataCursor = 0;

    auto makeBatch = [&](int batchSize) {
        trainBatch.clear();
        if (dataset.empty()) return;
        if (batchSize > ToyNet::MaxBatch) batchSize = ToyNet::MaxBatch;
        for (int i = 0; i < batchSize; ++i) {
            trainBatch.push_back(dataset[dataCursor]);
            dataCursor = (dataCursor + 1) % static_cast<int>(dataset.size());
        }
    };

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

        ImGui::Begin("Neural Net Demo");

        bool regenerate = false;
        regenerate |= ImGui::SliderInt("Points", &uiNumPoints, 100, 5000);
        regenerate |= ImGui::SliderFloat("Spread", &uiSpread, 0.01f, 0.5f);
        if (ImGui::Button("Regenerate Data")) {
            regenerate = true;
        }

        if (regenerate) {
            if (uiNumPoints < 10) uiNumPoints = 10;
            if (uiNumPoints > maxPoints) uiNumPoints = maxPoints;
            generateTwoBlobs(uiNumPoints, uiSpread, dataset);
            uploadDatasetToGPU(dataset);
        }

        ImGui::Separator();
        ImGui::SliderFloat("Point Size", &uiPointSize, 2.0f, 12.0f);

        ImGui::Separator();
        ImGui::SliderFloat("Learning Rate", &uiLearningRate, 0.0001f, 1.0f, "%.5f");
        ImGui::SliderInt("Batch Size", &uiBatchSize, 1, ToyNet::MaxBatch);

        if (ImGui::Button("Step Train")) {
            makeBatch(uiBatchSize);
            uiLastLoss = net.trainBatch(trainBatch, uiLastAccuracy);
            ++uiStepCount;
        }
        ImGui::SameLine();
        ImGui::Checkbox("Auto Train", &uiAutoTrain);

        ImGui::Text("Step: %d", uiStepCount);
        ImGui::Text("Loss: %.4f", uiLastLoss);
        ImGui::Text("Accuracy: %.3f", uiLastAccuracy);

        ImGui::Separator();
        ImGui::SliderInt("Auto Max Steps", &uiAutoMaxSteps, 1, 50000);
        ImGui::SliderFloat("Auto Target Loss", &uiAutoTargetLoss, 0.00001f, 1.0f, "%.5f");
        ImGui::Text("Auto stops when step >= %d or loss <= %.5f", uiAutoMaxSteps, uiAutoTargetLoss);

        ImGui::Text("Current points: %d", static_cast<int>(dataset.size()));

        ImGui::End();

        net.learningRate = uiLearningRate;

        if (uiAutoTrain) {
            makeBatch(uiBatchSize);
            uiLastLoss = net.trainBatch(trainBatch, uiLastAccuracy);
            ++uiStepCount;

            if (uiStepCount >= uiAutoMaxSteps || uiLastLoss <= uiAutoTargetLoss) {
                uiAutoTrain = false;
            }
        }

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        gridShader.use();
        if (gridColorLocation != -1) {
            gridShader.setVec3(gridColorLocation, 0.15f, 0.15f, 0.15f);
        }
        glBindVertexArray(gridVAO);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(gridVertices.size() / 2));
        glBindVertexArray(0);

        if (gridColorLocation != -1) {
            gridShader.setVec3(gridColorLocation, 0.8f, 0.8f, 0.8f);
        }
        glBindVertexArray(axisVAO);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(axisVertices.size() / 2));
        glBindVertexArray(0);

        pointShader.use();

        if (pointSizeLocation != -1) {
            pointShader.setFloat(pointSizeLocation, uiPointSize);
        }
        if (colorClass0Location != -1) {
            pointShader.setVec3(colorClass0Location, 0.2f, 0.6f, 1.0f);
        }
        if (colorClass1Location != -1) {
            pointShader.setVec3(colorClass1Location, 1.0f, 0.5f, 0.2f);
        }

        glBindVertexArray(pointVAO);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(dataset.size()));
        glBindVertexArray(0);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &pointVAO);
    glDeleteBuffers(1, &pointVBO);
    glDeleteVertexArrays(1, &gridVAO);
    glDeleteBuffers(1, &gridVBO);
    glDeleteVertexArrays(1, &axisVAO);
    glDeleteBuffers(1, &axisVBO);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
    return 0;
}
