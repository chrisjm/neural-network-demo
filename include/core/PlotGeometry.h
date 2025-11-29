#pragma once

#include <cstddef>

#include <vector>

#include "DataPoint.h"

class PointCloud {
public:
    PointCloud();

    void init(int maxPoints);
    void upload(const std::vector<DataPoint>& data);
    void draw(std::size_t pointCount) const;
    void shutdown();

private:
    unsigned int m_vao;
    unsigned int m_vbo;
    int          m_maxPoints;
};

class GridAxes {
public:
    GridAxes();

    void init(float gridStep = 0.25f);
    void drawGrid() const;
    void drawAxes() const;
    void shutdown();

private:
    std::vector<float> m_gridVertices;
    std::vector<float> m_axisVertices;

    unsigned int m_gridVAO;
    unsigned int m_gridVBO;
    unsigned int m_axisVAO;
    unsigned int m_axisVBO;

    std::size_t m_gridVertexCount; // number of vec2 vertices
    std::size_t m_axisVertexCount; // number of vec2 vertices
};
