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

    OptimizerType optimizerType;
    float         momentum;
    float         adamBeta1;
    float         adamBeta2;
    float         adamEps;

    InitMode initMode;

    int   stepCount;
    float lastLoss;
    float lastAccuracy;

    static constexpr int HistorySize = 4096;
    std::vector<float> lossHistory;
    std::vector<float> accuracyHistory;
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
