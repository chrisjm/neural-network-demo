#pragma once

#include <vector>

class ToyNet;

class FieldVisualizer {
public:
    FieldVisualizer();

    void init(int resolution);
    void update(const ToyNet& net);
    void draw() const;
    void setDirty();
    bool isDirty() const;
    void shutdown();

private:
    int m_resolution;
    int m_quads;
    int m_verts;
    unsigned int m_vao;
    unsigned int m_vbo;
    bool m_dirty;
    std::vector<float> m_vertexData;
    std::vector<float> m_probs;
};
