# TBrain-ETF

由趨勢科技所舉辦的ETF股票預測競賽。

## Tool

* Tensorflow 1.3.0
* Python 3.5.4

## Model

v1: Input Hidden layer --> LstmRNNCell --> Output Hidden layer (Dense)
v2: Input Hidden layer --> LstmRNNCell --> Attention layer --> Output Hidden layer (Dense)

## Directory

* data: dataset
* event: directory for tensorboard
* check: directory for store model

## Competition

2018/04/03~2018/06/22