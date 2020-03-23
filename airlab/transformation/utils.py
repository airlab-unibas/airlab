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

import SimpleITK as sitk

def compute_grid(image_size, dtype=th.float32, device='cpu'):

    dim = len(image_size)

    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]

        x = th.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = th.linspace(-1, 1, steps=nx).to(dtype=dtype)

        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return th.cat((x, y), 3).to(dtype=dtype, device=device)

    elif dim == 3:
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


def upsample_displacement(displacement, new_size, interpolation="linear"):
    """
        Upsample displacement field
    """
    dim = displacement.size()[-1]
    if dim == 2:
        displacement = th.transpose(displacement.unsqueeze(0), 0, 3).unsqueeze(0)
        if interpolation == 'linear':
            interpolation = 'bilinear'
        else:
            interpolation = 'nearest'
    elif dim == 3:
        displacement = th.transpose(displacement.unsqueeze(0), 0, 4).unsqueeze(0)
        if interpolation == 'linear':
            interpolation = 'trilinear'
        else:
            interpolation = 'nearest'

    upsampled_displacement = F.interpolate(displacement[..., 0], size=new_size, mode=interpolation, align_corners=False)

    if dim == 2:
        upsampled_displacement = th.transpose(upsampled_displacement.unsqueeze(-1), 1, -1)
    elif dim == 3:
        upsampled_displacement = th.transpose(upsampled_displacement.unsqueeze(-1), 1, -1)

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
def displacement_to_unit_displacement(displacement):
    # scale displacements from image
    # domain to 2square
    # - last dimension are displacements
    if type(displacement) == iutils.Displacement:
        df = displacement.image
    else:
        df = displacement

    for dim in range(df.shape[-1]):
        df[..., dim] = 2.0 * df[..., dim] / float(df.shape[-dim - 2] - 1)

    return displacement


"""
    Convert a unit displacement to a displacement field with the right spacing/scale
"""
def unit_displacement_to_displacement(displacement):
    # scale displacements from 2square
    # domain to image domain
    # - last dimension are displacements
    if type(displacement) == iutils.Displacement:
        df = displacement.image
    else:
        df = displacement

    # manipulate displacement field
    for dim in range(df.shape[-1]):
        df[..., dim] = float(df.shape[-dim - 2] - 1) * df[..., dim] / 2.0

    return displacement

def get_displacement_itk(displacement, refIm):
    displacement = displacement.detach().clone()
    dim = len(displacement.shape) - 1
    unit_displacement_to_displacement(displacement)
    dispIm = sitk.GetImageFromArray(
        displacement.cpu().numpy().astype('float64')\
        .transpose(list(range(dim-1, -1, -1)) + [dim])[..., ::-1],  # simpleitk image in numpy: D, H, W
        isVector=True
    )
    dispIm.CopyInformation(refIm)
    trans = sitk.DisplacementFieldTransform(dispIm)
    return trans

"""
    Create a 3d rotation matrix
"""
def rotation_matrix(phi_x, phi_y, phi_z, dtype=th.float32, device='cpu', homogene=False):
    R_x = th.Tensor([[1, 0, 0], [0, th.cos(phi_x), -th.sin(phi_x)], [0, th.sin(phi_x), th.cos(phi_x)]])
    R_y = th.Tensor([[th.cos(phi_y), 0, th.sin(phi_y)], [0, 1, 0], [-th.sin(phi_y), 0, th.cos(phi_y)]])
    R_z = th.Tensor([[th.cos(phi_z), -th.sin(phi_z), 0], [th.sin(phi_z), th.cos(phi_z), 0], [0, 0, 1]])

    matrix = th.mm(th.mm(R_z, R_y), R_x).to(dtype=dtype, device=device)

    if homogene:
        matrix_homogene = th.zeros(4, 4, dtype=dtype, device=device)
        matrix_homogene[3, 3] = 1
        matrix_homogene[0:3, 0:3] = matrix

        matrix = matrix_homogene

    return matrix


class Diffeomorphic():
    r"""
    Diffeomorphic transformation. This class computes the matrix exponential of a given flow field using the scaling
    and squaring algorithm according to:
              Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
              Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
              MICCAI 2018
              and
              Diffeomorphic Demons: Efficient Non-parametric Image Registration
              Tom Vercauterena et al., 2008

    """
    def __init__(self, image_size=None, scaling=10, dtype=th.float32, device='cpu'):

        self._dtype = dtype
        self._device = device
        self._dim = len(image_size)
        self._image_size = image_size
        self._scaling = scaling
        self._init_scaling = 8

        if image_size is not None:
            self._image_grid = compute_grid(image_size, dtype=dtype, device=device)
        else:
            self._image_grid = None

    def set_image_size(self, image_szie):
        self._image_size = image_szie
        self._image_grid = compute_grid(self._image_size, dtype=self._dtype, device=self._device)

    def calculate(self, displacement):
        if self._dim == 2:
            return Diffeomorphic.diffeomorphic_2D(displacement, self._image_grid, self._scaling)
        else:
            return Diffeomorphic.diffeomorphic_3D(displacement, self._image_grid, self._scaling)

    @staticmethod
    def _compute_scaling_value(displacement):

        with th.no_grad():
            scaling = 8
            norm = th.norm(displacement / (2 ** scaling))

            while norm > 0.5:
                scaling += 1
                norm = th.norm(displacement / (2 ** scaling))

        return scaling

    @staticmethod
    def diffeomorphic_2D(displacement, grid, scaling=-1):

        if scaling < 0:
            scaling = Diffeomorphic._compute_scaling_value(displacement)

        displacement = displacement / (2 ** scaling)

        displacement = displacement.transpose(2, 1).transpose(1, 0).unsqueeze(0)

        for i in range(scaling):
            displacement_trans = displacement.transpose(1, 2).transpose(2, 3)
            displacement = displacement + F.grid_sample(displacement, displacement_trans + grid)

        return displacement.transpose(1, 2).transpose(2, 3).squeeze()

    @staticmethod
    def diffeomorphic_3D(displacement, grid, scaling=-1):
        displacement = displacement / (2 ** scaling)

        displacement = displacement.transpose(3, 2).transpose(2, 1).transpose(0, 1).unsqueeze(0)

        for i in range(scaling):
            displacement_trans = displacement.transpose(1, 2).transpose(2, 3).transpose(3, 4)
            displacement = displacement + F.grid_sample(displacement, displacement_trans + grid)

        return displacement.transpose(1, 2).transpose(2, 3).transpose(3, 4).squeeze()



