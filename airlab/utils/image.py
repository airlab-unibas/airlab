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
import torch as th
import torch.nn.functional as F
import numpy as np
import sys

from . import kernelFunction

"""
    Object representing an image
"""
class Image:
    def __init__(self, tensor_image, image_size, image_spacing, image_origin):
        self.image = tensor_image
        self.size = image_size
        self.spacing = image_spacing
        self.origin = image_origin
        self.dtype = self.image.dtype
        self.device = self.image.device

    def to(self, dtype=th.float32, device='cpu'):
        self.image = self.image.to(dtype=dtype, device=device)
        self.dtype = self.image.dtype
        self.device = self.image.device

    def itk(self):
        itk_image = sitk.GetImageFromArray(self.image.cpu().numpy()[0, 0, ...])
        itk_image.SetSpacing(spacing=self.spacing)
        itk_image.SetOrigin(origin=self.origin)
        return itk_image

    def numpy(self):
        return self.image.cpu().numpy()[0, 0, ...]


"""
    Object representing a displacement image
"""
class Displacement(Image):
    def __init__(self, tensor_image, image_size, image_spacing):
        super(Displacement, self).__init__(tensor_image, image_size, image_spacing)

    def itk(self):
        if len(self.size) == 2:
            numpy_disp = self.image.cpu().numpy()
            itk_displacement = sitk.GetImageFromArray(numpy_disp, isVector=True)
        elif len(self.size) == 3:
            itk_displacement = sitk.GetImageFromArray(self.image.cpu().numpy())

        itk_displacement.SetSpacing(spacing=self.spacing)
        return itk_displacement

    def magnitude(self):
        # tmp = self.image.pow(2)
        # tmp = th.sum(self.image.pow(2), len(self.size))
        return Image(th.sqrt(th.sum(self.image.pow(2),  len(self.size))).unsqueeze(0).unsqueeze(0),
                     self.size, self.spacing)

    def numpy(self):
        return self.image.cpu().numpy()


"""
    Convert an image to tensor representation
"""
def read_image_as_tensor(filename, dtype=th.float32, device='cpu'):

    itk_image = sitk.ReadImage(filename, sitk.sitkFloat32)

    return create_tensor_image_from_itk_image(itk_image, dtype=dtype, device=device)


"""
    Convert an image to tensor representation
"""
def create_image_from_image(tensor_image, image):
    return Image(tensor_image, image.size, image.spacing)



"""
    Convert numpy image to AirlLab image format
"""
def image_from_numpy(image, pixel_spacing, dtype=th.float32, device='cpu'):
    tensor_image = th.from_numpy(image).unsqueeze_(0).unsqueeze_(0)
    tensor_image = tensor_image.to(dtype=dtype, device=device)
    return Image(tensor_image, image.shape, pixel_spacing)


"""
    Convert an image to tensor representation
"""
def create_displacement_image_from_image(tensor_displacement, image):
    return Displacement(tensor_displacement, image.size, image.spacing)


"""
    Create tensor image representation
"""
def create_tensor_image_from_itk_image(itk_image, dtype=th.float32, device='cpu'):

    # transform image in a unit direction
    image_dim = itk_image.GetDimension()
    if image_dim == 2:
        itk_image.SetDirection(sitk.VectorDouble([1, 0, 0, 1]))
    else:
        itk_image.SetDirection(sitk.VectorDouble([1, 0, 0, 0, 1, 0, 0, 0, 1]))

    image_spacing = itk_image.GetSpacing()
    image_origin = itk_image.GetOrigin()

    np_image = np.squeeze(sitk.GetArrayFromImage(itk_image))
    image_size = np_image.shape

    # adjust image spacing vector size if image contains empty dimension
    if len(image_size) != image_dim:
        image_spacing = image_spacing[0:len(image_size)]

    tensor_image = th.tensor(np_image, dtype=dtype, device=device).unsqueeze_(0).unsqueeze_(0)


    return Image(tensor_image, image_size, image_spacing, image_origin)


"""
    Create an image pyramide  
"""
def create_image_pyramide(image, down_sample_factor):

    image_dim = len(image.size)
    image_pyramide = []
    if image_dim == 2:
        for level in down_sample_factor:
            sigma = (th.tensor(level)/2).to(dtype=th.float32)

            kernel = kernelFunction.gaussian_kernel_2d(sigma.numpy(), asTensor=True)
            padding = np.array([(x - 1)/2 for x in kernel.size()], dtype=int).tolist()
            kernel.unsqueeze_(0).unsqueeze_(0)
            kernel = kernel.to(dtype=image.dtype, device=image.device)

            image_sample = F.conv2d(image.image, kernel, stride=level, padding=padding)
            image_size = image_sample.size()[-image_dim:]
            image_spacing = [x*y for x, y in zip(image.spacing, level)]
            image_origin = image.origin
            image_pyramide.append(Image(image_sample, image_size, image_spacing, image_origin))

        image_pyramide.append(image)
    elif image_dim == 3:
        for level in down_sample_factor:
            sigma = (th.tensor(level)/2).to(dtype=th.float32)

            kernel = kernelFunction.gaussian_kernel_3d(sigma.numpy(), asTensor=True)
            padding = np.array([(x - 1) / 2 for x in kernel.size()], dtype=int).tolist()
            kernel.unsqueeze_(0).unsqueeze_(0)
            kernel = kernel.to(dtype=image.dtype, device=image.device)

            image_sample = F.conv3d(image.image, kernel, stride=level, padding=padding)
            image_size = image_sample.size()[-image_dim:]
            image_spacing = [x*y for x, y in zip(image.spacing, level)]
            image_origin = image.origin
            image_pyramide.append(Image(image_sample, image_size, image_spacing, image_origin))

        image_pyramide.append(image)

    else:
        print("Error: ", image_dim, " is not supported with create_image_pyramide()")
        sys.exit(-1)

    return image_pyramide
