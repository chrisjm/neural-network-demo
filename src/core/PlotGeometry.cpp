#include "PlotGeometry.h"

#include <glad/glad.h>

PointCloud::PointCloud()
    : m_vao(0)
    , m_vbo(0)
    , m_maxPoints(0)
{
}

void PointCloud::init(int maxPoints)
{
    m_maxPoints = maxPoints;

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    const GLsizeiptr bufferSize = static_cast<GLsizeiptr>(m_maxPoints * 3 * sizeof(float));
    glBufferData(GL_ARRAY_BUFFER, bufferSize, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(0));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    glBindVertexArray(0);
}

void PointCloud::upload(const std::vector<DataPoint>& data)
{
    if (!m_vbo) {
        return;
    }

    std::vector<float> buffer;
    buffer.reserve(data.size() * 3);
    for (const auto& p : data) {
        buffer.push_back(p.x);
        buffer.push_back(p.y);
        buffer.push_back(static_cast<float>(p.label));
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    static_cast<GLsizeiptr>(buffer.size() * sizeof(float)),
                    buffer.data());
}

void PointCloud::draw(std::size_t pointCount) const
{
    if (!m_vao) {
        return;
    }

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(pointCount));
    glBindVertexArray(0);
}

void PointCloud::shutdown()
{
    if (m_vao) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_vbo) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }

    m_maxPoints = 0;
}

GridAxes::GridAxes()
    : m_gridVAO(0)
    , m_gridVBO(0)
    , m_axisVAO(0)
    , m_axisVBO(0)
    , m_gridVertexCount(0)
    , m_axisVertexCount(0)
{
}

void GridAxes::init(float gridStep)
{
    m_gridVertices.clear();
    m_axisVertices.clear();

    for (float x = -1.0f; x <= 1.0001f; x += gridStep) {
        m_gridVertices.push_back(x);
        m_gridVertices.push_back(-1.0f);
        m_gridVertices.push_back(x);
        m_gridVertices.push_back(1.0f);
    }

    for (float y = -1.0f; y <= 1.0001f; y += gridStep) {
        m_gridVertices.push_back(-1.0f);
        m_gridVertices.push_back(y);
        m_gridVertices.push_back(1.0f);
        m_gridVertices.push_back(y);
    }

    m_axisVertices = {
        -1.0f,  0.0f,  1.0f,  0.0f,
         0.0f, -1.0f,  0.0f,  1.0f
    };

    m_gridVertexCount = m_gridVertices.size() / 2;
    m_axisVertexCount = m_axisVertices.size() / 2;

    glGenVertexArrays(1, &m_gridVAO);
    glGenBuffers(1, &m_gridVBO);

    glBindVertexArray(m_gridVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_gridVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(m_gridVertices.size() * sizeof(float)),
                 m_gridVertices.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0));
    glBindVertexArray(0);

    glGenVertexArrays(1, &m_axisVAO);
    glGenBuffers(1, &m_axisVBO);

    glBindVertexArray(m_axisVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_axisVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(m_axisVertices.size() * sizeof(float)),
                 m_axisVertices.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), reinterpret_cast<void*>(0));
    glBindVertexArray(0);
}

void GridAxes::drawGrid() const
{
    if (!m_gridVAO) {
        return;
    }

    glBindVertexArray(m_gridVAO);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_gridVertexCount));
    glBindVertexArray(0);
}

void GridAxes::drawAxes() const
{
    if (!m_axisVAO) {
        return;
    }

    glBindVertexArray(m_axisVAO);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_axisVertexCount));
    glBindVertexArray(0);
}

void GridAxes::shutdown()
{
    if (m_gridVAO) {
        glDeleteVertexArrays(1, &m_gridVAO);
        m_gridVAO = 0;
    }
    if (m_gridVBO) {
        glDeleteBuffers(1, &m_gridVBO);
        m_gridVBO = 0;
    }

    if (m_axisVAO) {
        glDeleteVertexArrays(1, &m_axisVAO);
        m_axisVAO = 0;
    }
    if (m_axisVBO) {
        glDeleteBuffers(1, &m_axisVBO);
        m_axisVBO = 0;
    }

    m_gridVertices.clear();
    m_axisVertices.clear();
    m_gridVertexCount = 0;
    m_axisVertexCount = 0;
}
