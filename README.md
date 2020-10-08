<img width="50%" align="middle" src="docs/airlab_logo.png" alt="Airlab logo" />

<img width="10%" align="middle" src="https://readthedocs.org/projects/airlab/badge/?version=latest&style=plastic" alt="docs status" />



# Autograd Image Registration Laboratory
AirLab is an open laboratory for medical image registration. It provides an environment
for rapid prototyping and reproduction of registration algorithms. The unique feature of AirLab is, that the analytic gradients of the objective function are computed automatically with fosters rapid prototyping. In addition, the device on which the computations are performed, on a CPU or a GPU,
is transparent.
AirLab is implemented in Python using [PyTorch](https://pytorch.org/) as tensor and optimization library and SimpleITK for basic image IO. It profits therefore from recent advances made by the machine learning community.

AirLab is not meant to replace existing registration frameworks nor it implements deep learning methods only. It is rather a laboratory for image registration algorithms for rapid prototyping and reproduction. Furthermore, it borrows key functionality from PyTorch (autograd and optimization) which is of course not limited to deep learning methods.

We refer to our arXiv preprint [2018](https://arxiv.org/abs/1806.09907) for a detailed introduction of AirLab and its feature.

Authors: Robin Sandkuehler and Christoph Jud

[Documentation](https://airlab.readthedocs.io/en/latest/index.html)

Follow us on Twitter [<img width="2%" src="docs/twitter.png">](https://twitter.com/AirLab6?ref_src=twsrc%5Etfw) to get informed about the most recent features, achievements and bugfixes.

## Getting Started
1. Clone git repository: `git clone https://github.com/airlab-unibas/airlab.git`
2. Make sure that following python libraries are installed:
  1. pytorch
  2. numpy
  3. SimpleITK
  4. matplotlib
They can be installed with `pip`.

We recommend to start with the example applications provided in the `example` folder.


##### A Note on CPU Performance
The convolution operation, which is frequently used in AirLab, is performed in PyTorch. Currently, its CPU implementation is quite memory consuming. In order to process larger image data a GPU is required.


## Dependencies
The project depends on following libraries: 
* [PyTorch](https://pytorch.org/)
* [NumPy](www.numpy.org/)
* [SimpleITK](www.simpleitk.org/)
* [Matplotlib](https://matplotlib.org/)


## History
The project started in the [Center for medical Image Analysis & Navigation](https://dbe.unibas.ch/en/research/imaging-modelling-diagnosis/center-for-medical-image-analysis-navigation/) research group of the [University of Basel](http://www.unibas.ch). 


##### Authors and Contributors
* **Robin Sandkuehler** - *initial work* _(robin.sandkuehler@unibas.ch)_
* **Christoph Jud** - *initial work* _(christoph.jud@unibas.ch)_
* **Simon Andermatt** - *project support*
* **Alina Giger** - *presentation support*
* **Reinhard Wendler** - *logo design support*
* **Philippe C. Cattin** - *project support*


## Tutorial
<a href="https://airlab-unibas.github.io/MICCAITutorial2019/">
<img width="100%" align="middle" src="https://www.miccai2019.org/wp-content/uploads/2018/10/Web-Header.png" alt="miccai" /></a>

Check out our AIRLab tutorial at MICCAI 2019 in Shenzhen: https://airlab-unibas.github.io/MICCAITutorial2019/


## License
AirLab is licensed under the Apache 2.0 license. For details, consider the LICENSE and NOTICE file.

If you can use this software in any way, please cite us in your publications:

[2018] Robin Sandkuehler,  Christoph Jud, Simon Andermatt, and Philippe C. Cattin. "AirLab: Autograd Image Registration Laboratory". arXiv preprint arXiv:1806.09907, 2018. [link](https://arxiv.org/abs/1806.09907)


### Contributing
We released AirLab to contribute to the community. Thus, if you find and/or fix bugs or extend the software please contribute as well and let us know or make a pull request. 

We deeply appreciate the help of the following people:
* Iain Carmichael
* Benjamin Sugerman

### Other Open Source Projects
AirLab depends on several third party open source project which are included as library. For details, consider the `NOTICE` file.

