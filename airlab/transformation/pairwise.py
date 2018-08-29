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

import torch as th
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

from ..utils import kernelFunction as utils

from . import utils as tu

"""
    Base class for a transformation
"""
class _Transformation(th.nn.Module):
    def __init__(self, image_size, dtype=th.float32, device='cpu'):
        super(_Transformation, self).__init__()

        self._dtype = dtype
        self._device = device
        self._dim = len(image_size)
        self._image_size = np.array(image_size)
        self._constant_displacement = None


    def get_displacement_numpy(self):

        if self._dim == 2:
            return th.unsqueeze(self().detach(), 0).cpu().numpy()
        elif self._dim == 3:
            return self().detach().cpu().numpy()

    def get_displacement(self):

            return self().detach()


    def get_current_displacement(self):

        if self._dim == 2:
            return th.unsqueeze(self().detach(), 0).cpu().numpy()
        elif self._dim == 3:
            return self().detach().cpu().numpy()


    def set_constant_displacement(self, displacement):

        self._constant_displacement = displacement


    def _return_displacement(self, displacement):

        if self._constant_displacement is None:
            return displacement
        else:
            return (displacement + self._constant_displacement)

"""
 Rigid Transformation
"""
class RigidTransformation(_Transformation):
    def __init__(self, image_size, dtype=th.float32, device='cpu'):
        super(RigidTransformation, self).__init__(image_size, dtype, device)

        self._grid = th.squeeze(tu.compute_grid(image_size, dtype=dtype, device=device))

        self._grid = th.cat((self._grid, th.ones(*[list(image_size) + [1]], dtype=dtype, device=device)), self._dim)

        if self._dim == 2:
            self.trans_parameters = Parameter(th.Tensor([0, 0, 0])) # phi tx ty

            self._compute_displacement = self._compute_displacement_2d
        else:
            self.trans_parameters = Parameter(th.Tensor([0, 0, 0, 0, 0, 0])) # phix phiy phiy tx ty tz
            self._compute_displacement = self._compute_displacement_3d

    def _compute_displacement_2d(self):

        matrix = th.zeros(2, 3, dtype=self._dtype, device=self._device)
        matrix[0, 0] = th.cos(self.trans_parameters[0])
        matrix[0, 1] = -th.sin(self.trans_parameters[0])
        matrix[1, 0] = -matrix[0, 1]
        matrix[1, 1] = th.cos(self.trans_parameters[0])

        matrix[0, 2] = self.trans_parameters[1]
        matrix[1, 2] = self.trans_parameters[2]

        return th.mm(self._grid.view(np.prod(self._image_size).tolist(), self._dim + 1), matrix.t()) \
                .view(*(self._image_size.tolist()), self._dim) - self._grid[:,:,:2]

    def _compute_displacement_3d(self):
        rot_matrix = tu.rotation_matrix(self.trans_parameters[0], self.trans_parameters[1],
                                        self.trans_parameters[2], dtype=self._dtype, device=self._device)

        matrix = th.zeros(3, 4, dtype=dtype, device=device)

        matrix[:3, :,3] = rot_matrix;
        matrix[:3, 4] = self.trans_parameters[-3:]

        return th.mm(self._grid.view(np.prod(self._image_size).tolist(), self._dim + 1), matrix.t()) \
            .view(*(self._image_size.tolist()), self._dim)

    def print(self):
        if self._dim == 2:
            print("phi   \tt_x   \tt_y")
        elif self._dim == 3:
            print("\tphi_x   \tphi_y   \tphi_z   \tt_x   \tt_y   \tt_z")

        for p in self.parameters():
            for parameter in p:
                print("%.3f" % parameter.item(), "\t", end='', flush=True)



    def forward(self):
        return self._return_displacement(self._compute_displacement())


"""
    None parametric transformation
"""
class NonParametricTransformation(_Transformation):
    def __init__(self, image_size, dtype=th.float32, device='cpu'):
        super(NonParametricTransformation, self).__init__(image_size, dtype, device)

        self._tensor_size = [self._dim] + self._image_size.tolist()

        self.trans_parameters = Parameter(th.Tensor(*self._tensor_size))
        self.trans_parameters.data.fill_(0)

        self.to(dtype=self._dtype, device=self._device)

        if self._dim == 2:
            self._compute_displacement = self._compute_displacement_2d
        else:
            self._compute_displacement = self._compute_displacement_3d


    def set_start_parameter(self, parameters):
        if self._dim == 2:
            self.trans_parameters = Parameter(th.tensor(parameters.transpose(0, 2)))
        elif self._dim == 3:
            self.trans_parameters = Parameter(th.tensor(parameters.transpose(0, 1)
                                                        .transpose(0, 2).transpose(0, 3)))


    def _compute_displacement_2d(self):
        return self.trans_parameters.transpose(0, 2)

    def _compute_displacement_3d(self):
        return self.trans_parameters.transpose(0, 3).transpose(0, 2).transpose(0, 1)



    def forward(self):
        return self._return_displacement(self._compute_displacement())

"""
    Base class for kernel transformations
"""
class _KernelTransformation(_Transformation):
    def __init__(self, image_size, dtype=th.float32, device='cpu'):
        super(_KernelTransformation, self).__init__(image_size, dtype, device)

        self._kernel = None
        self._stride = 1
        self._padding = 0
        self._displacement_tmp = None
        self._displacement = None

        assert self._dim == 2 or self._dim == 3

        if self._dim == 2:
            self._compute_displacement = self._compute_displacement_2d
        else:
            self._compute_displacement = self._compute_displacement_3d


    def get_current_displacement(self):

        if self._dim == 2:
            return th.unsqueeze(self._compute_displacement().detach(), 0).cpu().numpy()
        elif self._dim == 3:
            return self._compute_displacement().detach().cpu().numpy()


    def _initialize(self):

        cp_grid = np.ceil(np.divide(self._image_size, self._stride)).astype(dtype=int)

        # new image size after convolution
        inner_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)

        # add one control point at each side
        cp_grid = cp_grid + 2

        # image size with additional control points
        new_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)

        # center image between control points
        image_size_diff = inner_image_size - self._image_size
        image_size_diff_floor = np.floor((np.abs(image_size_diff)/2))*np.sign(image_size_diff)

        self._crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2)*np.sign(image_size_diff)
        self._crop_end = image_size_diff_floor

        cp_grid = [1, self._dim] + cp_grid.tolist()

        # create transformation parameters
        self.trans_parameters = Parameter(th.Tensor(*cp_grid))
        self.trans_parameters.data.fill_(0)

        # copy to gpu if needed
        self.to(dtype=self._dtype, device=self._device)

        # convert to integer
        self._padding = self._padding.astype(dtype=int).tolist()
        self._stride = self._stride.astype(dtype=int).tolist()

        self._crop_start = self._crop_start.astype(dtype=int)
        self._crop_end = self._crop_end.astype(dtype=int)

        size = [1,1] + new_image_size.astype(dtype=int).tolist()
        self._displacement_tmp = th.empty(*size, dtype=self._dtype, device=self._device)

        size = [1, 1] + self._image_size.astype(dtype=int).tolist()
        self._displacement = th.empty(*size, dtype=self._dtype, device=self._device)


    def _compute_displacement_2d(self):
        displacement_tmp = F.conv_transpose2d(self.trans_parameters, self._kernel,
                                          padding=self._padding, stride=self._stride, groups=2)

        # crop displacement
        return th.squeeze(displacement_tmp[:, :,
                       self._stride[0] + self._crop_start[0]:-self._stride[0] - self._crop_end[0],
                       self._stride[1] + self._crop_start[1]:-self._stride[1] - self._crop_end[1]].transpose_(1, 3))


    def _compute_displacement_3d(self):

        # compute dense displacement
        displacement = F.conv_transpose3d(self.trans_parameters, self._kernel,
                                          padding=self._padding, stride=self._stride, groups=3)

        # crop displacement
        return th.squeeze(displacement[:, :, self._stride[0] + self._crop_start[0]:-self._stride[0] - self._crop_end[0],
                                  self._stride[1] + self._crop_start[1]:-self._stride[1] - self._crop_end[1],
                                  self._stride[2] + self._crop_start[2]:-self._stride[2] - self._crop_end[2]
                                  ].transpose_(1,4).transpose_(1,3).transpose_(1,2))


    def forward(self):

        return self._return_displacement(self._compute_displacement())


"""
    bspline kernel transformation
"""
class BsplineTransformation(_KernelTransformation):
    def __init__(self, image_size, sigma, order=2, dtype=th.float32, device='cpu'):
        super(BsplineTransformation, self).__init__(image_size, dtype, device)

        self._stride = np.array(sigma)

        # compute bspline kernel
        self._kernel = utils.bspline_kernel(sigma, dim=self._dim, order=order, asTensor=True, dtype=dtype)

        self._padding = (np.array(self._kernel.size()) - 1) / 2

        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.expand(self._dim, *((np.ones(self._dim + 1, dtype=int)*-1).tolist()))
        self._kernel = self._kernel.to(dtype=dtype, device=self._device)

        self._initialize()


"""
    Wendland kernel transformation
"""
class WendlandKernelTransformation(_KernelTransformation):
    def __init__(self, image_size, sigma, cp_scale=2, ktype="C4", dtype=th.float32, device='cpu'):
        super(WendlandKernelTransformation, self).__init__(image_size, dtype, device)

        self._stride = np.array(sigma)

        # compute bspline kernel
        self._kernel = utils.wendland_kernel(np.array(sigma)*cp_scale, dim=self._dim, type=type, asTensor=True, dtype=dtype)

        self._padding = (np.array(self._kernel.size()) - 1) / 2

        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.expand(self._dim, *((np.ones(self._dim + 1) * -1).tolist()))
        self._kernel = self._kernel.to(dtype=dtype, device=self._device)

        self._initialize()
