#pragma once

class TriangleMesh {
public:
    TriangleMesh(const float* vertices, int vertexCount, unsigned int& outVAO, unsigned int& outVBO);

    void bind() const;
    void draw() const;

    TriangleMesh(const TriangleMesh&) = delete;
    TriangleMesh& operator=(const TriangleMesh&) = delete;

private:
    unsigned int m_vao;
    unsigned int m_vbo;
    int m_vertexCount;
};
