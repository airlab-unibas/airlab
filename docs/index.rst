.. Airlab documentation master file, created by
   sphinx-quickstart on Wed Jun 20 22:07:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/airlab-unibas/airlab

Autograd Image Registration Laboratory
======================================

AirLab is an open laboratory for medical image registration. It provides an environment for rapid prototyping
and reproduction of registration algorithms. The unique feature of AirLab is, that the analytic gradients 
of the objective function are computed automatically with fosters rapid prototyping. In addition,
the device on which the computations are performed, on a CPU or a GPU, is transparent.
AirLab is implemented in Python using PyTorch as tensor and optimization library and
SimpleITK for basic image IO. It profits therefore from recent advances made by the machine learning community.

Authors: Robin Sandkuehler and Christoph Jud


.. toctree::
   :maxdepth: 1
   :caption: Building Blocks:

   registration
   transformation
   loss
   regulariser
   utils




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`


