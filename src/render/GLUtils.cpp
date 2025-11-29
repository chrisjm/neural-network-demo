// Use GLAD as the OpenGL loader and prevent GLFW from including system GL headers.
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#endif

#define GLFW_INCLUDE_NONE
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

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
