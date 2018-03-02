## Packages
* [pysc2](https://github.com/deepmind/pysc2) v1.2
* [sc2client-proto](https://github.com/Blizzard/s2client-proto) v3.18.0
* [pytorch](https://github.com/pytorch/pytorch) v0.3.1
* [StarCraft II (Linux)](https://github.com/Blizzard/s2client-proto#downloads) v.3.16.1
* [tensorboardX (Tensorboard for Pytorch)](https://github.com/lanpa/tensorboard-pytorch) v1.0

Python3 is required to resolve [multiprocessing issue](https://github.com/ikostrikov/pytorch-a3c/issues/37).

Detail installation [steps](#installation).

## Training
To train agent in "FindAndDefeatZerglings" mini game with 8 different worker threads:
```bash
python main.py --map-name FindAndDefeatZerglings --num-processes 8
```
Use `python main.py --help` to see all available options.

## <a id='installation'></a> Installation
#### pytorch
Follow instruction [here](http://pytorch.org) and chose OS, Package Manager, Python version and CUDA version accordingly.
- Linux
```bash
# check cuda version
nvcc --version
# use CUDA 9.1 as example
pip3 install http://download.pytorch.org/whl/cu91/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 
```
- OS X
```bash
# no GPU as example
pip3 install http://download.pytorch.org/whl/torch-0.3.1-cp35-cp35m-macosx_10_6_x86_64.whl  
```
#### pysc2
```bash
pip install pysc2
```
#### tensorboardX
```bash
pip install git+https://github.com/lanpa/tensorboard-pytorch
# TensorFlow is required
pip install tensorflow
```
