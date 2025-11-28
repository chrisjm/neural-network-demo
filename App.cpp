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

#include "App.h"
#include "ShaderProgram.h"
#include "TriangleMesh.h"

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

// GLFW will call this when something goes wrong at the windowing/OS level
void glfw_error_callback(int error, const char* description) {
    std::cerr << "[GLFW ERROR] code=" << error << ", description=" << description << std::endl;
}

// Helper to dump OpenGL errors with a label so you can see *where* they came from
void check_gl_error(const char* label) {
    GLenum err;
    bool hadError = false;
    while ((err = glGetError()) != GL_NO_ERROR) {
        hadError = true;
        std::cerr << "[GL ERROR] (" << label << ") code=0x" << std::hex << err << std::dec << std::endl;
    }
    if (!hadError) {
        std::cout << "[GL OK] (" << label << ")" << std::endl;
    }
}

// Helper function to resize the viewport if the user resizes the window
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    std::cout << "[Callback] framebuffer_size_callback: width=" << width << ", height=" << height << std::endl;
    glViewport(0, 0, width, height);
}

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

    unsigned int VBO, VAO;
    // Generate IDs for buffers
    // 1. Bind the Vertex Array Object first
    // 2. Copy our vertices array in a buffer for OpenGL to use
    // THIS IS THE MOMENT we move data from RAM -> VRAM
    // 3. Set the vertex attribute pointers
    // Tell the GPU how to interpret the raw binary data (It's 3 floats per vertex)
    TriangleMesh triangle(vertices, 3, VAO, VBO);

    check_gl_error("After VAO/VBO setup");

    // Initial CPU-side state for uniforms
    float offsetX = 0.0f;
    float offsetY = 0.0f;
    float scale   = 1.0f;
    float rotation = 0.0f; // in radians
    float color[3] = {1.0f, 0.5f, 0.2f}; // start with the original orange-ish color
    std::cout << "[State] Initial offset   = (" << offsetX << ", " << offsetY << ")" << std::endl;
    std::cout << "[State] Initial scale    = " << scale << std::endl;
    std::cout << "[State] Initial rotation = " << rotation << " radians" << std::endl;
    std::cout << "[State] Initial color    = (" << color[0] << ", " << color[1] << ", " << color[2] << ")" << std::endl;

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

        // Arrow keys move the triangle by changing the offset uniform
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
            offsetY += moveSpeed;
            std::cout << "[Input] UP    -> offset = (" << offsetX << ", " << offsetY << ")" << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
            offsetY -= moveSpeed;
            std::cout << "[Input] DOWN  -> offset = (" << offsetX << ", " << offsetY << ")" << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
            offsetX -= moveSpeed;
            std::cout << "[Input] LEFT  -> offset = (" << offsetX << ", " << offsetY << ")" << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
            offsetX += moveSpeed;
            std::cout << "[Input] RIGHT -> offset = (" << offsetX << ", " << offsetY << ")" << std::endl;
        }

        // Scale controls (Z/X)
        if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS) {
            scale -= scaleStep;
            if (scale < 0.1f) scale = 0.1f; // avoid inverting/vanishing
            std::cout << "[Input] Z -> scale = " << scale << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS) {
            scale += scaleStep;
            std::cout << "[Input] X -> scale = " << scale << std::endl;
        }

        // Rotation controls (Q/E)
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            rotation -= rotationStep;
            std::cout << "[Input] Q -> rotation = " << rotation << " radians" << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            rotation += rotationStep;
            std::cout << "[Input] E -> rotation = " << rotation << " radians" << std::endl;
        }

        // Number keys change the color uniform
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            color[0] = 1.0f; color[1] = 0.0f; color[2] = 0.0f; // Red
            std::cout << "[Input] 1 -> color = RED" << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
            color[0] = 0.0f; color[1] = 1.0f; color[2] = 0.0f; // Green
            std::cout << "[Input] 2 -> color = GREEN" << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
            color[0] = 0.0f; color[1] = 0.0f; color[2] = 1.0f; // Blue
            std::cout << "[Input] 3 -> color = BLUE" << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
            color[0] = 1.0f; color[1] = 1.0f; color[2] = 1.0f; // White
            std::cout << "[Input] 4 -> color = WHITE" << std::endl;
        }
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
            color[0] = 1.0f; color[1] = 0.5f; color[2] = 0.2f; // Back to original orange
            std::cout << "[Input] 5 -> color = ORANGE" << std::endl;
        }

        // Render Command 1: Clear the screen
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render Command 2: Use our compiled shader program
        shaderProgram.use();

        // Update uniforms every frame so the GPU sees the latest CPU-side values
        shaderProgram.setVec2(offsetLocation, offsetX, offsetY);
        shaderProgram.setVec3(colorLocation, color[0], color[1], color[2]);
        shaderProgram.setFloat(scaleLocation, scale);
        shaderProgram.setFloat(rotationLocation, rotation);

        // Render Command 3: Bind the mesh
        triangle.bind();

        // Render Command 4: DRAW!
        // "Draw 3 vertices starting from index 0"
        triangle.draw();

        // Swap buffers (Double buffering prevents flickering)
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glfwTerminate();
    return 0;
}
