#pragma once

#include <vector>

#include "DataPoint.h"
#include "ToyNet.h"

struct Trainer {
    ToyNet net;

    float learningRate;
    int   batchSize;
    bool  autoTrain;
    int   autoMaxEpochs;
    float autoTargetLoss;

    OptimizerType optimizerType;
    float         momentum;
    float         adamBeta1;
    float         adamBeta2;
    float         adamEps;

    InitMode initMode;

    int   epochCount;
    float lastLoss;
    float lastAccuracy;

    static constexpr int HistorySize = 4096;
    std::vector<float> lossHistory;
    std::vector<float> accuracyHistory;
    int   historyCount;

    Trainer();

    void resetForNewDataset();

    void trainOneEpoch(const std::vector<DataPoint>& dataset);

    bool autoTrainEpochs(const std::vector<DataPoint>& dataset);

private:
    std::vector<DataPoint> m_batch;
    int m_dataCursor;

    void makeBatch(const std::vector<DataPoint>& dataset);
};
