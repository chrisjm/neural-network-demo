#include "FieldVisualizer.h"

#include "ToyNet.h"

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

    m_vertexData.resize(static_cast<std::size_t>(m_verts * 5));
    m_probs.resize(static_cast<std::size_t>(m_resolution * m_resolution));

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    const GLsizeiptr bufferSize = static_cast<GLsizeiptr>(m_vertexData.size() * sizeof(float));
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

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

void FieldVisualizer::update(const ToyNet& net)
{
    if (!m_vao || !m_vbo) {
        return;
    }

    const float step = 2.0f / static_cast<float>(m_resolution - 1);

    for (int j = 0; j < m_resolution; ++j) {
        float y = -1.0f + step * static_cast<float>(j);
        for (int i = 0; i < m_resolution; ++i) {
            float x = -1.0f + step * static_cast<float>(i);
            float p0 = 0.0f;
            float p1 = 0.0f;
            net.forwardSingle(x, y, p0, p1);
            m_probs[j * m_resolution + i] = p1;
        }
    }

    const float c0r = 0.2f;
    const float c0g = 0.6f;
    const float c0b = 1.0f;
    const float c1r = 1.0f;
    const float c1g = 0.5f;
    const float c1b = 0.2f;

    std::size_t v = 0;

    auto addVertex = [&](float x, float y, float p1) {
        float r = (1.0f - p1) * c0r + p1 * c1r;
        float g = (1.0f - p1) * c0g + p1 * c1g;
        float b = (1.0f - p1) * c0b + p1 * c1b;

        m_vertexData[v++] = x;
        m_vertexData[v++] = y;
        m_vertexData[v++] = r;
        m_vertexData[v++] = g;
        m_vertexData[v++] = b;
    };

    for (int j = 0; j < m_resolution - 1; ++j) {
        float y0 = -1.0f + step * static_cast<float>(j);
        float y1 = -1.0f + step * static_cast<float>(j + 1);
        for (int i = 0; i < m_resolution - 1; ++i) {
            float x0 = -1.0f + step * static_cast<float>(i);
            float x1 = -1.0f + step * static_cast<float>(i + 1);

            int idx00 = j * m_resolution + i;
            int idx10 = j * m_resolution + (i + 1);
            int idx01 = (j + 1) * m_resolution + i;
            int idx11 = (j + 1) * m_resolution + (i + 1);

            float p00 = m_probs[idx00];
            float p10 = m_probs[idx10];
            float p01 = m_probs[idx01];
            float p11 = m_probs[idx11];

            addVertex(x0, y0, p00);
            addVertex(x1, y0, p10);
            addVertex(x1, y1, p11);

            addVertex(x0, y0, p00);
            addVertex(x1, y1, p11);
            addVertex(x0, y1, p01);
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
    m_probs.clear();
    m_dirty = false;
}
