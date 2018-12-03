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

from .image import Image, Displacement, read_image_as_tensor, create_image_from_image, image_from_numpy, \
                   create_displacement_image_from_image, create_tensor_image_from_itk_image, create_image_pyramid

from .points import Points

from .graph import Graph

from .matrix import MatrixDiagonalElement, LaplaceMatrix, band_mv, expm_eig, expm_krylov
from .kernelFunction import gaussian_kernel_1d, gaussian_kernel_2d, gaussian_kernel_3d, gaussian_kernel,\
                            wendland_kernel_1d, wendland_kernel_2d, wendland_kernel_3d, wendland_kernel, \
                            bspline_kernel_1d, bspline_kernel_2d, bspline_kernel_3d, bspline_kernel

from .domain import compute_coordinate_grid_2d, compute_coordinate_grid_3d, get_center_of_mass,\
                    get_joint_domain_images

from .imageLoader import ImageLoader

from .imageFilters import auto_crop_image_filter, normalize_images, remove_bed_filter


__all__ = ['Image', 'Displacement', 'read_image_as_tensor', 'create_image_from_image', 'image_from_numpy',\
           'create_displacement_image_from_image', 'create_tensor_image_from_itk_image', 'create_image_pyramid',\
           'Points', 'Graph', 'MatrixDiagonalElement', 'LaplaceMatrix', 'band_mv', 'expm_eig', 'expm_krylov',\
           'gaussian_kernel_1d', 'gaussian_kernel_2d', 'gaussian_kernel_3d', 'gaussian_kernel',\
           'wendland_kernel_1d', 'wendland_kernel_2d', 'wendland_kernel_3d', 'wendland_kernel',\
           'bspline_kernel_1d', 'bspline_kernel_2d', 'bspline_kernel_3d', 'bspline_kernel',\
           'compute_coordinate_grid_2d', 'compute_coordinate_grid_3d', 'get_center_of_mass', 'get_joint_domain_images',\
           'ImageLoader', 'auto_crop_image_filter', 'remove_bed_filter']
