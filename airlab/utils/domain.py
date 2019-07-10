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

import numpy as np
import torch as th
import multiprocessing as mp
import SimpleITK as sitk
from .image import Image
from .image import create_image_from_image

"""
Create a two dimensional coordinate grid
"""
def compute_coordinate_grid_2d(image):

    x = np.linspace(0, image.size[0] - 1, num=image.size[0])
    y = np.linspace(0, image.size[1] - 1, num=image.size[1])

    y_m, x_m = np.meshgrid(y, x)

    return [x_m, y_m]

"""
Create a three dimensional coordinate grid
"""
def compute_coordinate_grid_3d(image):

    x = np.linspace(0, image.size[0] - 1, num=image.size[0])
    y = np.linspace(0, image.size[1] - 1, num=image.size[1])
    z = np.linspace(0, image.size[2] - 1, num=image.size[2])

    y_m, x_m, z_m = np.meshgrid(y, x, z)

    return [x_m, y_m, z_m]


def get_center_of_mass(image):
    """
    Returns the center of mass of the image (weighted average of coordinates where the intensity values serve as weights)

    image (Image): input is an airlab image
    return (array): coordinates of the center of mass
    """

    num_points = np.prod(image.size)
    coordinate_value_array = np.zeros([num_points, len(image.size)+1])  # allocate coordinate value array

    values = image.image.squeeze().cpu().numpy().reshape(num_points)  # vectorize image
    coordinate_value_array[:, 0] = values

    if len(image.size)==2:
        X, Y = compute_coordinate_grid_2d(image)
        coordinate_value_array[:, 1] = X.reshape(num_points)
        coordinate_value_array[:, 2] = Y.reshape(num_points)

    elif len(image.size)==3:
        X, Y, Z = compute_coordinate_grid_3d(image)
        coordinate_value_array[:, 1] = X.reshape(num_points)
        coordinate_value_array[:, 2] = Y.reshape(num_points)
        coordinate_value_array[:, 3] = Z.reshape(num_points)

    else:
        raise Exception("Only 2 and 3 space dimensions supported")

    # compared to the itk implementation the
    #   center of gravity for the 2d lenna should be [115.626, 91.9961]
    #   center of gravity for 3d image it should be [2.17962, 5.27883, -1.81531]

    cm = np.average(coordinate_value_array[:, 1:], axis=0, weights=coordinate_value_array[:, 0])
    cm = cm * image.spacing + image.origin

    return cm


def get_joint_domain_images(fixed_image, moving_image, default_value=0, interpolator=2, cm_alignment=False, compute_masks=False):
    """
    The method brings the fixed and moving image in a common image domain in order to be compatible with the
    registration framework of airlab. Different from the ITK convention, the registration in airlab is performed
    on pixels and not on points. This allows an efficient evaluation of the image metrics, the synthesis of
    displacement fields and warp of the moving image.

    If the images already have the same image domain (after a possible center of mass alignment) no resampling is
    performed and only masks are generated for return.

    Step 1: The moving image is aligned to the fixed image by matching the center of mass of the two images.
    Step 2: The new image domain is the smallest possible domain where both images are contained completely.
            The minimum spacing is taken as new spacing. This second step can increase the amount of pixels.
    Step 3: Fixed and moving image are resampled on this new domain.
    Step 4: Masks are built which defines in which region the respective image is not defined on this new domain.

    Note: The minimum possible value of the fixed image type is used as placeholder when resampling.
          Hence, this value should not be present in the images

    fixed_image (Image): fixed image provided as airlab image
    moving_image (Image): moving image provided as airlab image
    default_value (float|int): default value which defines the value which is set where the images are not defined in the new domain
    interpolator (int):  nn=1, linear=2, bspline=3
    cm_alignment (bool): defines whether the center of mass refinement should be performed prior to the resampling
    compute_masks (bool): defines whether the masks should be created. otherwise, None is returned as masks.
    return (tuple): resampled fixed image, fixed mask, resampled moving image, moving mask
    """
    f_mask = None
    m_mask = None

    cm_displacement = None

    # align images using center of mass
    if cm_alignment:
        cm_displacement = get_center_of_mass(fixed_image) - get_center_of_mass(moving_image)
        moving_image.origin = moving_image.origin + cm_displacement

    # check if domains are equal, as then nothing has to be resampled
    if np.all(fixed_image.origin == moving_image.origin) and\
            np.all(fixed_image.spacing == moving_image.spacing) and\
            np.all(fixed_image.size == moving_image.size):
        if compute_masks:
            f_mask = th.ones_like(fixed_image.image)
            m_mask = th.ones_like(moving_image.image)

            f_mask = Image(f_mask, fixed_image.size, fixed_image.spacing, fixed_image.origin)
            m_mask = Image(m_mask, moving_image.size, moving_image.spacing, moving_image.origin)
        return fixed_image, f_mask, moving_image, m_mask, None

    # common origin
    origin = np.minimum(fixed_image.origin, moving_image.origin)

    # common extent
    f_extent = np.array(fixed_image.origin) + (np.array(fixed_image.size)-1)*np.array(fixed_image.spacing)
    m_extent = np.array(moving_image.origin) + (np.array(moving_image.size)-1)*np.array(moving_image.spacing)
    extent = np.maximum(f_extent, m_extent)

    # common spacing
    spacing = np.minimum(fixed_image.spacing, moving_image.spacing)

    # common size
    size = np.ceil(((extent-origin)/spacing)+1).astype(int)

    # Resample images
    # fixed and moving image are resampled in new domain
    # the default value for resampling is set to a predefined value
    # (minimum possible value of the fixed image type) to use it
    # to create masks. At the end, default values are replaced with
    # the provided default value
    minimum_value = default_value
    if compute_masks:
        minimum_value = float(np.finfo(fixed_image.image.cpu().numpy().dtype).tiny)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size.tolist())
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetDefaultPixelValue(minimum_value)
    resampler.SetInterpolator(interpolator)
    resampler.SetNumberOfThreads(mp.cpu_count())

    # resample fixed and moving image
    f_image = Image(resampler.Execute(fixed_image.itk()))
    m_image = Image(resampler.Execute(moving_image.itk()))

    f_image.to(dtype=fixed_image.dtype, device=fixed_image.device)
    m_image.to(dtype=moving_image.dtype, device=moving_image.device)

    # create masks
    if compute_masks:
        f_mask = th.ones_like(f_image.image)
        m_mask = th.ones_like(m_image.image)

        f_mask[f_image.image == minimum_value] = 0
        m_mask[m_image.image == minimum_value] = 0

        f_mask = Image(f_mask, size, spacing, origin)
        m_mask = Image(m_mask, size, spacing, origin)

        # reset default value in images
        f_image.image[f_image.image == minimum_value] = default_value
        m_image.image[m_image.image == minimum_value] = default_value

    return f_image, f_mask, m_image, m_mask, cm_displacement


