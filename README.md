# Self_supervised model

## Python version I used

Python 3.8.5

## 3rd party Python libraries used
tqdm 4.59.0

numpy 1.19.2

pytorch 1.4.0

torchvision 0.5.0

torch_scatter 2.0.4

cudatoolkit 10.1

## Train a model
~~~
sh train.sh
~~~

note this repo is not refactored, the destination folder can be specified in train/self_supervised_train.py around line 115

## Inference
For use the model, please check out the inference.ipynb file, it has all the information.