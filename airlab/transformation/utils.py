## Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
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

import torch as th
import torch.nn.functional as F

from ..utils import image as iutils


def compute_grid(image_size, dtype=th.float32, device='cpu'):

    dim = len(image_size)

    if(dim == 2):
        nx = image_size[0]
        ny = image_size[1]

        x = th.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = th.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return th.cat((x, y), 3).to(dtype=dtype, device=device)

    elif(dim == 3):
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]

        x = th.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = th.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = th.linspace(-1, 1, steps=nz).to(dtype=dtype)

        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)

        return th.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        print("Error " + dim + "is not a valid grid type")


"""
    Upsample displacement field
"""
def upsample_displacement(displacement, new_size, interpolation="linear"):

    dim = displacement.size()[-1]
    if dim == 2:
        displacement = th.transpose(displacement.unsqueeze_(0), 0, 3).unsqueeze_(0)
        if interpolation == 'linear':
            interpolation = 'bilinear'
        else:
            interpolation = 'nearest'
    elif dim == 3:
        displacement = th.transpose(displacement.unsqueeze_(0), 0, 4).unsqueeze_(0)
        if interpolation == 'linear':
            interpolation = 'trilinear'
        else:
            interpolation = 'nearest'

    upsampled_displacement = F.interpolate(displacement[...,0], size=new_size, mode=interpolation, align_corners=False)

    if dim == 2:
        upsampled_displacement = th.transpose(upsampled_displacement.unsqueeze_(-1), 1, -1)
    elif dim == 3:
        upsampled_displacement = th.transpose(upsampled_displacement.unsqueeze_(-1), 1, -1)


    return upsampled_displacement[0, 0, ...]


"""
    Warp image with displacement
"""
def warp_image(image, displacement):

    image_size = image.size

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)

    # warp image
    warped_image = F.grid_sample(image.image, displacement + grid)

    return iutils.Image(warped_image, image_size, image.spacing, image.origin)


"""
    Convert displacement to a unit displacement
"""
def displacement_to_unit_displacement(displacement, pixel_spacing):
    # scale displacements from image
    # domain to 2square
    # - last dimension are displacements
    for dim in range(displacement.shape[-1]):
        displacement[..., dim] = 2.0 * displacement[..., dim] / float(displacement.shape[-dim - 2] - 1)
    return displacement


"""
    Convert a unit displacement to a  itk like displacement
"""
def unit_displacement_to_dispalcement(displacement, pixel_spacing):
    # scale displacements from 2square
    # domain to image domain
    # - last dimension are displacements
    for dim in range(displacement.shape[-1]):
        displacement[..., dim] = float(displacement.shape[-dim - 2] - 1) * displacement[..., dim] / 2.0
    return displacement

"""
    Create a 3d rotation matrix
"""
def rotation_matrix(phi_x, phi_y, phi_z, dtype=th.float32, device='cpu'):
    R_x = th.Tensor([[1, 0, 0], [0, th.cos(phi_x), -th.sin(phi_x)], [0, th.sin(phi_x), th.cos(phi_x)]])
    R_y = th.Tensor([[th.cos(phi_y), 0, th.sin(phi_y)], [0, 1, 0], [-th.sin(phi_y), 0, th.cos(phi_y)]])
    R_z = th.Tensor([[h.cos(phi_z), -th.sin(phi_z), 0], [th.sin(phi_z), th.cos(phi_z), 0], [0, 0, 1]])

    return th.mm(th.mm(R_z, R_y), R_x).to(dtype=dtype, device=device)



