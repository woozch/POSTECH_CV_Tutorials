# POSTECH Computer Vision Tutorials
## TensorFlow Tutorials

TensorFlow Tutorial for Computer Vision (CV) tasks.
It is suitable for students who want to find clear and concise examples about TensorFlow.
You may need to study Linear algebra, Probability and statistics.
Most of the contents of this tutorial is about Deep Learning in CV.

### Environment

After installing [Anaconda](https://www.continuum.io/downloads), you should create a [conda environment](http://conda.pydata.org/docs/using/envs.html)
so you do not destroy your main installation in case you make a mistake somewhere:
```bash
conda create --name tensorflow-py3 python=3.5
```
Now you can switch to the new environment in your terminal by running the following (on Linux terminal):
```bash
source activate tensorflow-py3
```

### Required Python Packages

The tutorials require several Python packages to be installed. The packages are listed below:
```
numpy
scipy
jupyter
matplotlib
pillow
scikit-learn
seaborn
```

you can install these packages by typing:
```bash
conda install numpy scipy jupyter matplotlib pillow scikit-learn seaborn
```


### Installation of Tensorflow

There are different ways of installing and executing TensorFlow depending on your machine.
Please follow this website: [Tensorflow Installation Guide](https://www.tensorflow.org/install/)

Note that the GPU-version of TensorFlow also requires the installation of various
NVIDIA drivers(nvidia drivers, cuda, cudnn, etc), which is not described here.
We strongly recommend you to use gpu version of TensorFlow, otherwise, you may need to fix the code.

### Dataset
##### MNIST
Some examples require MNIST dataset for training and testing.
This dataset will be downloaded automatically when executing example codes (with input_data.py).
MNIST is a database of handwritten digits, which is most popular dataset in deep learning tutorial.
You can check the data in official website [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### Tutorial index



# Reference
* [TensorFlow Examples](https://github.com/aymericdamien/TensorFlow-Examples)
* [TensorFlow Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
