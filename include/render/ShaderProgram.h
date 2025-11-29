#pragma once

#include <string>

class ShaderProgram {
public:
    ShaderProgram(const char* vertexSrc, const char* fragmentSrc);
    ~ShaderProgram();

    void use() const;
    unsigned int getId() const { return m_id; }

    void setVec2(int location, float x, float y) const;
    void setVec3(int location, float x, float y, float z) const;
    void setFloat(int location, float value) const;

    ShaderProgram(const ShaderProgram&) = delete;
    ShaderProgram& operator=(const ShaderProgram&) = delete;

    ShaderProgram(ShaderProgram&& other) noexcept;
    ShaderProgram& operator=(ShaderProgram&& other) noexcept;

private:
    unsigned int m_id;
};
