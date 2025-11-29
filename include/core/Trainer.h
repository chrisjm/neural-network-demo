#pragma once

#include <vector>

#include "DataPoint.h"
#include "ToyNet.h"

struct Trainer {
    ToyNet net;

    float learningRate;
    int   batchSize;
    bool  autoTrain;
    int   autoMaxSteps;
    float autoTargetLoss;

    int   stepCount;
    float lastLoss;
    float lastAccuracy;

    static constexpr int HistorySize = 512;
    float lossHistory[HistorySize];
    float accuracyHistory[HistorySize];
    int   historyCount;

    Trainer();

    void resetForNewDataset();

    void stepOnce(const std::vector<DataPoint>& dataset);

    bool stepAuto(const std::vector<DataPoint>& dataset);

private:
    std::vector<DataPoint> m_batch;
    int m_dataCursor;

    void makeBatch(const std::vector<DataPoint>& dataset);
};
