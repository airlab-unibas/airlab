# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import SimpleITK as sitk
import multiprocessing as mp
import numpy as np
import torch as th

from .image import Image

def AutoCropImageFilter(image):

    vals = (image.squeeze()>0).argmax(dim=0).nonzero()
    xmin = vals[:, 0].min(dim=0)
    xmax = vals[:, 0].max(dim=0)
    ymin = vals[:, 1].min(dim=0)
    ymax = vals[:, 1].max(dim=0)

    pass

def RemoveBedFilter(image, cropping=True):

    # define parameters
    houndsfield_min = -300
    houndsfield_max = 3071
    houndsfield_default = -1024

    radius_opening = 3
    radius_closing = 40


    image_itk = image.itk()

    # resample image
    workingSize = np.array(image.size)
    workingSize[0] /= 3
    workingSize[1] /= 3
    workingSpacing = np.array(image.spacing, dtype=float) * np.array(image.size, dtype=float) / np.array(workingSize, dtype=float)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(image.origin)
    resampler.SetSize(workingSize.tolist())
    resampler.SetOutputSpacing(workingSpacing.tolist())
    resampler.SetInterpolator(2) # linear interpolation
    resampler.SetNumberOfThreads(mp.cpu_count())

    image_small = resampler.Execute(image_itk)


    # threshold image
    thresholder = sitk.BinaryThresholdImageFilter()
    thresholder.SetOutsideValue(0)
    thresholder.SetInsideValue(1)
    thresholder.SetLowerThreshold(houndsfield_min)
    thresholder.SetUpperThreshold(houndsfield_max)
    thresholder.SetNumberOfThreads(mp.cpu_count())

    image_thresh = thresholder.Execute(image_small)


    # morphological opening with ball as structuring element
    # removes thin structures as the bed
    opening = sitk.BinaryMorphologicalOpeningImageFilter()
    opening.SetKernelType(sitk.sitkBall)
    opening.SetKernelRadius(radius_opening)
    opening.SetForegroundValue(1)
    opening.SetNumberOfThreads(mp.cpu_count())

    image_opened = opening.Execute(image_thresh)


    # crop mask and image
    if cropping:
        image_opened = AutoCropImageFilter(Image(image_opened)).itk()


    # morphological closing with ball as structuring element
    # fills up the lungs
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    closing.SetKernelRadius(sitk.sitkBall)
    closing.SetKernelRadius(radius_closing)
    closing.SetForegroundValue(1)
    closing.SetNumberOfThreads(mp.cpu_count())

    image_closed = closing.Execute(image_opened)


    # resample mask to original size
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(image.origin)
    resampler.SetSize(image.size)
    resampler.SetOutputSpacing(image.spacing)
    resampler.SetInterpolator(1) # nearest neighbor interpolation
    resampler.SetNumberOfThreads(mp.cpu_count())

    bodyMask = resampler.Execute(image_closed)


    # mask image with found label map
    masking = sitk.MaskImageFilter()
    masking.SetMaskingValue(0)
    masking.SetOutsideValue(houndsfield_default)
    masking.SetNumberOfThreads(mp.cpu_count())

    outImage = masking.Execute(image_itk, bodyMask)

    return (Image(outImage), Image(bodyMask))
