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

#include "ShaderProgram.h"

ShaderProgram::ShaderProgram(const char* vertexSrc, const char* fragmentSrc)
    : m_id(0) {
    int success = 0;
    char infoLog[512];

    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSrc, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    } else {
        std::cout << "[Shader] Vertex shader compiled successfully" << std::endl;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSrc, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    } else {
        std::cout << "[Shader] Fragment shader compiled successfully" << std::endl;
    }

    m_id = glCreateProgram();
    glAttachShader(m_id, vertexShader);
    glAttachShader(m_id, fragmentShader);
    glLinkProgram(m_id);

    glGetProgramiv(m_id, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(m_id, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    } else {
        std::cout << "[Shader] Program linked successfully" << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

ShaderProgram::~ShaderProgram() {
    if (m_id != 0) {
        glDeleteProgram(m_id);
    }
}

ShaderProgram::ShaderProgram(ShaderProgram&& other) noexcept
    : m_id(other.m_id) {
    other.m_id = 0;
}

ShaderProgram& ShaderProgram::operator=(ShaderProgram&& other) noexcept {
    if (this != &other) {
        if (m_id != 0) {
            glDeleteProgram(m_id);
        }
        m_id = other.m_id;
        other.m_id = 0;
    }
    return *this;
}

void ShaderProgram::use() const {
    glUseProgram(m_id);
}

void ShaderProgram::setVec2(int location, float x, float y) const {
    if (location == -1) {
#ifndef NDEBUG
        std::cerr << "[ShaderProgram] Warning: setVec2 called with location == -1" << std::endl;
#endif
        return;
    }
    glUniform2f(location, x, y);
}

void ShaderProgram::setVec3(int location, float x, float y, float z) const {
    if (location == -1) {
#ifndef NDEBUG
        std::cerr << "[ShaderProgram] Warning: setVec3 called with location == -1" << std::endl;
#endif
        return;
    }
    glUniform3f(location, x, y, z);
}

void ShaderProgram::setInt(int location, int value) const {
    if (location == -1) {
#ifndef NDEBUG
        std::cerr << "[ShaderProgram] Warning: setInt called with location == -1" << std::endl;
#endif
        return;
    }
    glUniform1i(location, value);
}

void ShaderProgram::setFloat(int location, float value) const {
    if (location == -1) {
#ifndef NDEBUG
        std::cerr << "[ShaderProgram] Warning: setFloat called with location == -1" << std::endl;
#endif
        return;
    }
    glUniform1f(location, value);
}

void ShaderProgram::setFloatArray(int location, const float* data, int count) const {
    if (location == -1) {
#ifndef NDEBUG
        std::cerr << "[ShaderProgram] Warning: setFloatArray called with location == -1" << std::endl;
#endif
        return;
    }
    if (data != nullptr && count > 0) {
        glUniform1fv(location, count, data);
    }
}
