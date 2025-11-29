#include "DatasetGenerator.h"

#include <cmath>
#include <cstdlib>

namespace {

float rand01()
{
    return static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
}

void generateTwoBlobs(int numPoints, float spread, std::vector<DataPoint>& out)
{
    out.clear();
    out.reserve(static_cast<std::size_t>(numPoints));

    int half = numPoints / 2;

    for (int i = 0; i < half; ++i) {
        float angle = rand01() * 2.0f * static_cast<float>(M_PI);
        float radius = spread * rand01();
        float cx = -0.5f;
        float cy = 0.0f;
        float x = cx + std::cos(angle) * radius;
        float y = cy + std::sin(angle) * radius;
        out.push_back({x, y, 0});
    }

    for (int i = half; i < numPoints; ++i) {
        float angle = rand01() * 2.0f * static_cast<float>(M_PI);
        float radius = spread * rand01();
        float cx = 0.5f;
        float cy = 0.0f;
        float x = cx + std::cos(angle) * radius;
        float y = cy + std::sin(angle) * radius;
        out.push_back({x, y, 1});
    }
}

void generateConcentricCircles(int numPoints, float noise, std::vector<DataPoint>& out)
{
    out.clear();
    out.reserve(static_cast<std::size_t>(numPoints));

    int half = numPoints / 2;

    float innerR = 0.3f;
    float outerR = 0.75f;
    float noiseScale = noise;

    for (int i = 0; i < half; ++i) {
        float angle = rand01() * 2.0f * static_cast<float>(M_PI);
        float r = innerR + noiseScale * (rand01() - 0.5f);
        float x = r * std::cos(angle);
        float y = r * std::sin(angle);
        out.push_back({x, y, 0});
    }

    for (int i = half; i < numPoints; ++i) {
        float angle = rand01() * 2.0f * static_cast<float>(M_PI);
        float r = outerR + noiseScale * (rand01() - 0.5f);
        float x = r * std::cos(angle);
        float y = r * std::sin(angle);
        out.push_back({x, y, 1});
    }
}

void generateTwoMoons(int numPoints, float noise, std::vector<DataPoint>& out)
{
    out.clear();
    out.reserve(static_cast<std::size_t>(numPoints));

    int half = numPoints / 2;

    float radius = 0.8f;
    float offsetX = 0.5f;
    float offsetY = 0.25f;
    float noiseScale = noise;

    for (int i = 0; i < half; ++i) {
        float t = rand01() * static_cast<float>(M_PI);
        float x = std::cos(t) * radius - offsetX;
        float y = std::sin(t) * radius * 0.5f;
        x += noiseScale * (rand01() - 0.5f);
        y += noiseScale * (rand01() - 0.5f);
        out.push_back({x, y, 0});
    }

    for (int i = half; i < numPoints; ++i) {
        float t = rand01() * static_cast<float>(M_PI);
        float x = std::cos(t) * radius + offsetX;
        float y = -std::sin(t) * radius * 0.5f + offsetY;
        x += noiseScale * (rand01() - 0.5f);
        y += noiseScale * (rand01() - 0.5f);
        out.push_back({x, y, 1});
    }
}

void generateXORQuads(int numPoints, float spread, std::vector<DataPoint>& out)
{
    out.clear();
    out.reserve(static_cast<std::size_t>(numPoints));

    int quarter = numPoints / 4;
    float r = spread;

    auto sampleAround = [&](float cx, float cy, int label, int count) {
        for (int i = 0; i < count; ++i) {
            float angle = rand01() * 2.0f * static_cast<float>(M_PI);
            float rad = r * rand01();
            float x = cx + std::cos(angle) * rad;
            float y = cy + std::sin(angle) * rad;
            out.push_back({x, y, label});
        }
    };

    sampleAround(-0.5f, -0.5f, 0, quarter);
    sampleAround( 0.5f,  0.5f, 0, quarter);
    sampleAround(-0.5f,  0.5f, 1, quarter);
    sampleAround( 0.5f, -0.5f, 1, numPoints - 3 * quarter);
}

void generateSpirals(int numPoints, float noise, std::vector<DataPoint>& out)
{
    out.clear();
    out.reserve(static_cast<std::size_t>(numPoints));

    int half = numPoints / 2;

    float maxT = 3.5f * static_cast<float>(M_PI);
    float a = 0.1f;
    float b = 0.05f;
    float noiseScale = noise;

    auto sampleSpiral = [&](int label, float angleOffset, int count) {
        for (int i = 0; i < count; ++i) {
            float t = rand01() * maxT;
            float r = a + b * t;
            float x = r * std::cos(t + angleOffset);
            float y = r * std::sin(t + angleOffset);
            x += noiseScale * (rand01() - 0.5f);
            y += noiseScale * (rand01() - 0.5f);
            out.push_back({x, y, label});
        }
    };

    sampleSpiral(0, 0.0f, half);
    sampleSpiral(1, static_cast<float>(M_PI), numPoints - half);
}

} // namespace

void generateDataset(DatasetType type,
                     int numPoints,
                     float spread,
                     std::vector<DataPoint>& out)
{
    switch (type) {
        case DatasetType::TwoBlobs:
            generateTwoBlobs(numPoints, spread, out);
            break;
        case DatasetType::ConcentricCircles:
            generateConcentricCircles(numPoints, spread, out);
            break;
        case DatasetType::TwoMoons:
            generateTwoMoons(numPoints, spread, out);
            break;
        case DatasetType::XORQuads:
            generateXORQuads(numPoints, spread, out);
            break;
        case DatasetType::Spirals:
            generateSpirals(numPoints, spread, out);
            break;
    }
}
