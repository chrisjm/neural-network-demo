#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_GLCOREARB
#include <GLFW/glfw3.h>
#else
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#endif

#include "TriangleMesh.h"

TriangleMesh::TriangleMesh(const float* vertices, int vertexCount, unsigned int& outVAO, unsigned int& outVBO)
    : m_vao(0), m_vbo(0), m_vertexCount(vertexCount) {
    // Generate IDs for buffers
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    // 1. Bind the Vertex Array Object first
    glBindVertexArray(m_vao);

    // 2. Copy our vertices array in a buffer for OpenGL to use
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    // THIS IS THE MOMENT we move data from RAM -> VRAM
    glBufferData(GL_ARRAY_BUFFER, m_vertexCount * 3 * sizeof(float), vertices, GL_STATIC_DRAW);

    // 3. Set the vertex attribute pointers
    // Tell the GPU how to interpret the raw binary data (It's 3 floats per vertex)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Unbind (cleanup)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    outVAO = m_vao;
    outVBO = m_vbo;
}

void TriangleMesh::bind() const {
    glBindVertexArray(m_vao);
}

void TriangleMesh::draw() const {
    glDrawArrays(GL_TRIANGLES, 0, m_vertexCount);
}
