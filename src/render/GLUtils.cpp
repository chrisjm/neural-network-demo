// Use GLAD as the OpenGL loader and prevent GLFW from including system GL headers.
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "GLUtils.h"

void glfw_error_callback(int error, const char* description) {
    std::cerr << "[GLFW ERROR] code=" << error << ", description=" << description << std::endl;
}

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

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    std::cout << "[Callback] framebuffer_size_callback: width=" << width << ", height=" << height << std::endl;
    glViewport(0, 0, width, height);
}

std::string loadTextFile(const char* path) {
    if (!path) {
        std::cerr << "[File] Failed to open text file: <null path>" << std::endl;
        return std::string();
    }

    // First, try the path as-is (e.g., when running from project root).
    std::ifstream file(path);
    if (!file) {
        // Fallback: try looking one directory up (common when running from build/).
        std::string altPath = std::string("../") + path;
        file.open(altPath);
        if (!file) {
            std::cerr << "[File] Failed to open text file: " << path
                      << " or " << altPath << std::endl;
            return std::string();
        }
    }

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}
