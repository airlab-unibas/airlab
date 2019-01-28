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

import os
import multiprocessing as mp
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(mp.cpu_count())

import SimpleITK as sitk
import numpy as np
import torch as th

from .image import Image


def auto_crop_image_filter(image, boundary_value=0):
    """
    Performs an auto cropping of values on boundary
    image (Image): image which has to be cropped
    boundary_value (float|int): specifies the boundary value which will be cropped
    return (Image): a new image with cropped boundary
    """
    msk = 1 - (image.image.squeeze() == boundary_value)

    rminmax = []

    for d in range(len(msk.shape)):
        region = msk.argmax(dim=d).nonzero()
        rminmax.append((region.min(dim=0)[0], region.max(dim=0)[0]))
        #print(rminmax[-1])

    if image.ndim == 2:
        cropped = image.image.squeeze()[rminmax[1][0]:rminmax[1][1], rminmax[0][0]:rminmax[0][1]]
        origin = image.origin + th.Tensor(image.spacing) * th.Tensor([rminmax[1][0], rminmax[0][0]])
    elif image.ndim == 3:
        cropped = image.image.squeeze()[rminmax[1][0][0]:rminmax[1][1][0], \
                                        rminmax[0][0][0]:rminmax[0][1][0], \
                                        rminmax[0][0][1]:rminmax[0][1][1]]
        #print(cropped.shape)
        origin = th.Tensor(image.origin) + th.Tensor(image.spacing) * th.Tensor([rminmax[1][0][0], rminmax[0][0][0],rminmax[0][0][1]])
    else:
        raise Exception("Only 2 and 3 space dimensions supported")

    size = tuple(cropped.shape)
    cropped.unsqueeze_(0).unsqueeze_(0)

    return Image(cropped, size, image.spacing, origin.tolist())


def normalize_images(fixed_image, moving_image):
    """
    Noramlize image intensities by extracting joint minimum and dividing by joint maximum

    Note: the function is inplace

    fixed_image (Image): fixed image
    moving_image (Image): moving image
    return (Image, Image): normalized images
    """
    fixed_min = fixed_image.image.min()
    moving_min = moving_image.image.min()

    min_val = min(fixed_min, moving_min)

    fixed_image.image -= min_val
    moving_image.image -= min_val

    moving_max = moving_image.image.max()
    fixed_max = fixed_image.image.max()
    max_val = max(fixed_max, moving_max)

    fixed_image.image /= max_val
    moving_image.image /= max_val

    return (fixed_image, moving_image)



def remove_bed_filter(image, cropping=True):
    """
    Removes fine structures from the image using morphological operators. It can be used to remove the bed structure
    usually present in CT images. The resulting image and the respective body mask can be cropped with the cropping
    option.

    Note: the morphological operations are performed on a downsampled version of the image

    image (Image): image of interest
    cropping (bool): specifies if the image should be cropped after bed removal
    return (Image, Image): bed-free image and a body mask
    """

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

    image_tmp = resampler.Execute(image_itk)


    # threshold image
    thresholder = sitk.BinaryThresholdImageFilter()
    thresholder.SetOutsideValue(0)
    thresholder.SetInsideValue(1)
    thresholder.SetLowerThreshold(houndsfield_min)
    thresholder.SetUpperThreshold(houndsfield_max)
    thresholder.SetNumberOfThreads(mp.cpu_count())

    image_tmp = thresholder.Execute(image_tmp)


    # morphological opening with ball as structuring element
    # removes thin structures as the bed
    opening = sitk.BinaryMorphologicalOpeningImageFilter()
    opening.SetKernelType(sitk.sitkBall)
    opening.SetKernelRadius(radius_opening)
    opening.SetForegroundValue(1)
    opening.SetNumberOfThreads(mp.cpu_count())

    image_tmp = opening.Execute(image_tmp)


    # crop zero values from mask boundary
    if cropping:
        image_tmp = auto_crop_image_filter(Image(image_tmp).to(device=image.device)).itk()


    # morphological closing with ball as structuring element
    # fills up the lungs
    closing = sitk.BinaryMorphologicalClosingImageFilter()
    closing.SetKernelRadius(sitk.sitkBall)
    closing.SetKernelRadius(radius_closing)
    closing.SetForegroundValue(1)
    closing.SetNumberOfThreads(mp.cpu_count())

    image_tmp = closing.Execute(image_tmp)


    # resample mask to original spacing
    mask_size = np.array(np.array(image_tmp.GetSpacing(), dtype=float)*np.array(image_tmp.GetSize(),dtype=float)/np.array(image.spacing, dtype=float), dtype=int).tolist()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(image_tmp.GetOrigin())
    resampler.SetSize(mask_size)
    resampler.SetOutputSpacing(image.spacing)
    resampler.SetInterpolator(1) # nearest neighbor interpolation
    resampler.SetNumberOfThreads(mp.cpu_count())

    bodyMask = resampler.Execute(image_tmp)

    # resample also original image
    resampler.SetInterpolator(2)
    image_itk = resampler.Execute(image_itk)


    # mask image with found label map
    masking = sitk.MaskImageFilter()
    masking.SetMaskingValue(0)
    masking.SetOutsideValue(houndsfield_default)
    masking.SetNumberOfThreads(mp.cpu_count())

    outImage = masking.Execute(image_itk, bodyMask)

    return (Image(outImage).to(device=image.device), Image(bodyMask).to(device=image.device))
