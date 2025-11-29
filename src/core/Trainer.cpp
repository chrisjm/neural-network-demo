#include "Trainer.h"

Trainer::Trainer()
    : learningRate(0.1f)
    , batchSize(64)
    , autoTrain(false)
    , autoMaxSteps(2000)
    , autoTargetLoss(0.01f)
    , stepCount(0)
    , lastLoss(0.0f)
    , lastAccuracy(0.0f)
    , historyCount(0)
    , m_dataCursor(0)
{
    m_batch.reserve(ToyNet::MaxBatch);
    for (int i = 0; i < HistorySize; ++i) {
        lossHistory[i] = 0.0f;
        accuracyHistory[i] = 0.0f;
    }
    net.resetParameters();
}

void Trainer::resetForNewDataset()
{
    net.resetParameters();
    stepCount    = 0;
    lastLoss     = 0.0f;
    lastAccuracy = 0.0f;
    autoTrain    = false;
    m_dataCursor = 0;

    historyCount = 0;
    for (int i = 0; i < HistorySize; ++i) {
        lossHistory[i] = 0.0f;
        accuracyHistory[i] = 0.0f;
    }
}

void Trainer::makeBatch(const std::vector<DataPoint>& dataset)
{
    m_batch.clear();
    if (dataset.empty()) {
        return;
    }

    int size = batchSize;
    if (size < 1) {
        size = 1;
    }
    if (size > ToyNet::MaxBatch) {
        size = ToyNet::MaxBatch;
    }

    const int dataCount = static_cast<int>(dataset.size());

    for (int i = 0; i < size; ++i) {
        m_batch.push_back(dataset[m_dataCursor]);
        m_dataCursor = (m_dataCursor + 1) % dataCount;
    }
}

void Trainer::stepOnce(const std::vector<DataPoint>& dataset)
{
    if (dataset.empty()) {
        return;
    }

    net.learningRate = learningRate;

    makeBatch(dataset);

    lastLoss = net.trainBatch(m_batch, lastAccuracy);
    ++stepCount;

    if (historyCount < HistorySize) {
        lossHistory[historyCount]     = lastLoss;
        accuracyHistory[historyCount] = lastAccuracy;
        ++historyCount;
    } else {
        for (int i = 1; i < HistorySize; ++i) {
            lossHistory[i - 1]     = lossHistory[i];
            accuracyHistory[i - 1] = accuracyHistory[i];
        }
        lossHistory[HistorySize - 1]     = lastLoss;
        accuracyHistory[HistorySize - 1] = lastAccuracy;
    }
}

bool Trainer::stepAuto(const std::vector<DataPoint>& dataset)
{
    if (!autoTrain) {
        return false;
    }

    stepOnce(dataset);

    if (stepCount >= autoMaxSteps || lastLoss <= autoTargetLoss) {
        autoTrain = false;
    }

    return true;
}
