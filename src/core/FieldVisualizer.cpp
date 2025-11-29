#include "FieldVisualizer.h"

#include <glad/glad.h>

#include <cmath>

FieldVisualizer::FieldVisualizer()
    : m_resolution(0)
    , m_quads(0)
    , m_verts(0)
    , m_vao(0)
    , m_vbo(0)
    , m_dirty(false)
{
}

void FieldVisualizer::init(int resolution)
{
    m_resolution = resolution;
    if (m_resolution < 2) {
        m_resolution = 2;
    }

    m_quads = (m_resolution - 1) * (m_resolution - 1);
    m_verts = m_quads * 6;

    // 2 floats per vertex: position only. Color is computed in the fragment shader.
    m_vertexData.resize(static_cast<std::size_t>(m_verts * 2));

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    const GLsizeiptr bufferSize = static_cast<GLsizeiptr>(m_vertexData.size() * sizeof(float));
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0));

    glBindVertexArray(0);

    m_dirty = true;
}

void FieldVisualizer::setDirty()
{
    m_dirty = true;
}

bool FieldVisualizer::isDirty() const
{
    return m_dirty;
}

void FieldVisualizer::update()
{
    if (!m_vao || !m_vbo) {
        return;
    }

    const float step = 2.0f / static_cast<float>(m_resolution - 1);

    std::size_t v = 0;

    for (int j = 0; j < m_resolution - 1; ++j) {
        float y0 = -1.0f + step * static_cast<float>(j);
        float y1 = -1.0f + step * static_cast<float>(j + 1);
        for (int i = 0; i < m_resolution - 1; ++i) {
            float x0 = -1.0f + step * static_cast<float>(i);
            float x1 = -1.0f + step * static_cast<float>(i + 1);

            // First triangle
            m_vertexData[v++] = x0;
            m_vertexData[v++] = y0;
            m_vertexData[v++] = x1;
            m_vertexData[v++] = y0;
            m_vertexData[v++] = x1;
            m_vertexData[v++] = y1;

            // Second triangle
            m_vertexData[v++] = x0;
            m_vertexData[v++] = y0;
            m_vertexData[v++] = x1;
            m_vertexData[v++] = y1;
            m_vertexData[v++] = x0;
            m_vertexData[v++] = y1;
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    const GLsizeiptr bufferSize = static_cast<GLsizeiptr>(m_vertexData.size() * sizeof(float));
    glBufferSubData(GL_ARRAY_BUFFER, 0, bufferSize, m_vertexData.data());

    m_dirty = false;
}

void FieldVisualizer::draw() const
{
    if (!m_vao) {
        return;
    }

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_verts));
    glBindVertexArray(0);
}

void FieldVisualizer::shutdown()
{
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }

    m_vertexData.clear();
    m_dirty = false;
}
