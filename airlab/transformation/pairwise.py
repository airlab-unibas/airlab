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
    def __init__(self, image_size, diffeomorphic=False, dtype=th.float32, device='cpu'):
        super(_Transformation, self).__init__()

        self._dtype = dtype
        self._device = device
        self._dim = len(image_size)
        self._image_size = np.array(image_size)
        self._constant_displacement = None
        self._diffeomorphic = diffeomorphic
        self._constant_flow = None

        self._compute_flow = None

        if self._diffeomorphic:
            self._diffeomorphic_calculater = tu.Diffeomorphic(image_size, dtype=dtype, device=device)
        else:
            self._diffeomorphic_calculater = None

    def get_flow(self):

        if self._constant_flow is None:
            return self._compute_flow().detach()
        else:
            return self._compute_flow().detach() + self._constant_flow

    def set_constant_flow(self, flow):
        self._constant_flow = flow

    def get_displacement_numpy(self):

        if self._dim == 2:
            return th.unsqueeze(self().detach(), 0).cpu().numpy()
        elif self._dim == 3:
            return self().detach().cpu().numpy()

    def get_displacement(self):
            return self().detach()

    # def get_current_displacement(self):
    #
    #     if self._dim == 2:
    #         return th.unsqueeze(self().detach(), 0).cpu().numpy()
    #     elif self._dim == 3:
    #         return self().detach().cpu().numpy()

    # def set_constant_displacement(self, displacement):
    #
    #     self._constant_displacement = displacement

    # def get_inverse_transformation(self, displacement):
    #     if self._diffeomorphic:
    #         if self._dim == 2:
    #             inv_displacement = self._diffeomorphic_calculater.calculate(displacement * -1)
    #         else:
    #             inv_displacement = self._diffeomorphic_calculater.calculate(displacement * -1)
    #     else:
    #         print("error displacement ")
    #         inv_displacement = None
    #
    #     return inv_displacement

    def get_inverse_displacement(self):

        flow = self._concatenate_flows(self._compute_flow()).detach()

        if self._diffeomorphic:
                inv_displacement = self._diffeomorphic_calculater.calculate(flow * -1)
        else:
            print("error displacement ")
            inv_displacement = None

        return inv_displacement

    def _compute_diffeomorphic_displacement(self, flow):

        return self._diffeomorphic_calculater.calculate(flow)

    def _concatenate_flows(self, flow):

        if self._constant_flow is None:
            return flow
        else:
            return flow + self._constant_flow


class RigidTransformation(_Transformation):
    r"""
    Rigid centred transformation for 2D and 3D.

    Args:
        moving_image (Image): moving image for the registration
        opt_cm (bool): using center of as parameter for the optimisation
    """
    def __init__(self, moving_image, opt_cm=False):
        super(RigidTransformation, self).__init__(image_size=moving_image.size,
                                                  dtype=moving_image.dtype,
                                                  device=moving_image.device)

        self._opt_cm = opt_cm

        grid = th.squeeze(tu.compute_grid(moving_image.size, dtype=self._dtype))

        grid = th.cat((grid, th.ones(*[list(moving_image.size) + [1]], dtype=self._dtype)), self._dim)\
               .to(device=self._device)

        self.register_buffer("_grid", grid)

        # compute the initial center of mass of the moving image
        intensity_sum = th.sum(moving_image.image)

        self._center_mass_x = th.sum(moving_image.image.squeeze() * self._grid[..., 0]) / intensity_sum
        self._center_mass_y = th.sum(moving_image.image.squeeze() * self._grid[..., 1]) / intensity_sum


        self._phi_z = Parameter(th.tensor(0.0))
        self._t_x = Parameter(th.tensor(0.0))
        self._t_y = Parameter(th.tensor(0.0))

        self._trans_matrix_pos = None
        self._trans_matrix_cm = None
        self._trans_matrix_cm_rw = None
        self._rotation_matrix = None

        if self._opt_cm:
            self._center_mass_x = Parameter(self._center_mass_x)
            self._center_mass_y = Parameter(self._center_mass_y)

        if self._dim == 2:
            self._compute_transformation = self._compute_transformation_2d

        else:
            self._compute_transformation = self._compute_transformation_3d

            self._center_mass_z = th.sum(moving_image.image.squeeze() * self._grid[..., 2]) / intensity_sum

            self._t_z = Parameter(th.tensor(0.0))
            self._phi_x = Parameter(th.tensor(0.0))
            self._phi_y = Parameter(th.tensor(0.0))

            if self._opt_cm:
                self._center_mass_z = Parameter(self._center_mass_z)

    def init_translation(self, fixed_image):
        r"""
        Initialize the translation parameters with the difference between the center of mass of the
        fixed and the moving image

        Args:
            fixed_image (Image): Fixed image for the registration
        """
        intensity_sum = th.sum(fixed_image.image)

        fixed_image_center_mass_x = th.sum(fixed_image.image.squeeze() * self._grid[..., 0]) / intensity_sum
        fixed_image_center_mass_y = th.sum(fixed_image.image.squeeze() * self._grid[..., 1]) / intensity_sum

        self._t_x = Parameter(self._center_mass_x - fixed_image_center_mass_x)
        self._t_y = Parameter(self._center_mass_y - fixed_image_center_mass_y)

        if self._dim == 3:
            fixed_image_center_mass_z = th.sum(fixed_image.image.squeeze() * self._grid[..., 2]) / intensity_sum
            self._t_z = Parameter(self._center_mass_z - fixed_image_center_mass_z)
            
    @property
    def transformation_matrix(self):
        return self._compute_transformation_matrix()

    def set_parameters(self, t, phi, rotation_center=None):
        """
        Set parameters manually

        t (array): 2 or 3 dimensional array specifying the spatial translation
        phi (array): 1 or 3 dimensional array specifying the rotation angles
        rotation_center (array): 2 or 3 dimensional array specifying the rotation center (default is zeros)
        """
        self._t_x = Parameter(th.tensor(t[0]).to(dtype=self._dtype, device=self._device))
        self._t_y = Parameter(th.tensor(t[1]).to(dtype=self._dtype, device=self._device))
        self._phi_z = Parameter(th.tensor(phi[0]).to(dtype=self._dtype, device=self._device))

        if rotation_center is not None:
            self._center_mass_x = rotation_center[0]
            self._center_mass_y = rotation_center[1]

        if len(t) == 2:
            self._compute_transformation_2d()
        else:
            self._t_z = Parameter(th.tensor(t[2]).to(dtype=self._dtype, device=self._device))
            self._phi_x = Parameter(th.tensor(phi[1]).to(dtype=self._dtype, device=self._device))
            self._phi_y = Parameter(th.tensor(phi[2]).to(dtype=self._dtype, device=self._device))
            if rotation_center is not None:
                self._center_mass_z = rotation_center[1]
                
            self._compute_transformation_3d()


    def _compute_transformation_2d(self):

        self._trans_matrix_pos = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))
        self._trans_matrix_cm = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))
        self._trans_matrix_cm_rw = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))
        self._rotation_matrix = th.zeros(self._dim + 1, self._dim + 1, dtype=self._dtype, device=self._device)
        self._rotation_matrix[-1, -1] = 1

        self._trans_matrix_pos[0, 2] = self._t_x
        self._trans_matrix_pos[1, 2] = self._t_y

        self._trans_matrix_cm[0, 2] = -self._center_mass_x
        self._trans_matrix_cm[1, 2] = -self._center_mass_y

        self._trans_matrix_cm_rw[0, 2] = self._center_mass_x
        self._trans_matrix_cm_rw[1, 2] = self._center_mass_y

        self._rotation_matrix[0, 0] = th.cos(self._phi_z)
        self._rotation_matrix[0, 1] = -th.sin(self._phi_z)
        self._rotation_matrix[1, 0] = th.sin(self._phi_z)
        self._rotation_matrix[1, 1] = th.cos(self._phi_z)

    def _compute_transformation_3d(self):

        self._trans_matrix_pos = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))
        self._trans_matrix_cm = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))
        self._trans_matrix_cm_rw = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))

        self._trans_matrix_pos[0, 3] = self._t_x
        self._trans_matrix_pos[1, 3] = self._t_y
        self._trans_matrix_pos[2, 3] = self._t_z

        self._trans_matrix_cm[0, 3] = -self._center_mass_x
        self._trans_matrix_cm[1, 3] = -self._center_mass_y
        self._trans_matrix_cm[2, 3] = -self._center_mass_z

        self._trans_matrix_cm_rw[0, 3] = self._center_mass_x
        self._trans_matrix_cm_rw[1, 3] = self._center_mass_y
        self._trans_matrix_cm_rw[2, 3] = self._center_mass_z

        R_x = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))
        R_x[1, 1] = th.cos(self._phi_x)
        R_x[1, 2] = -th.sin(self._phi_x)
        R_x[2, 1] = th.sin(self._phi_x)
        R_x[2, 2] = th.cos(self._phi_x)

        R_y = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))
        R_y[0, 0] = th.cos(self._phi_y)
        R_y[0, 2] = th.sin(self._phi_y)
        R_y[2, 0] = -th.sin(self._phi_y)
        R_y[2, 2] = th.cos(self._phi_y)

        R_z = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))
        R_z[0, 0] = th.cos(self._phi_z)
        R_z[0, 1] = -th.sin(self._phi_z)
        R_z[1, 0] = th.sin(self._phi_z)
        R_z[1, 1] = th.cos(self._phi_z)
        
        self._rotation_matrix = th.mm(th.mm(R_z, R_y), R_x)

    def _compute_transformation_matrix(self):
        transformation_matrix = th.mm(th.mm(th.mm(self._trans_matrix_pos, self._trans_matrix_cm),
                                                  self._rotation_matrix), self._trans_matrix_cm_rw)[0:self._dim, :]
        return transformation_matrix

    def _compute_dense_flow(self, transformation_matrix):

        displacement = th.mm(self._grid.view(np.prod(self._image_size).tolist(), self._dim + 1),
                             transformation_matrix.t()).view(*(self._image_size.tolist()), self._dim) \
                       - self._grid[..., :self._dim]
        return displacement

    def print(self):
        for name, param in self.named_parameters():
            print(name, param.item())
            
    def compute_displacement(self, transformation_matrix):
        return self._compute_dense_flow(transformation_matrix)

    def forward(self):

        self._compute_transformation()
        transformation_matrix = self._compute_transformation_matrix()
        flow = self._compute_dense_flow(transformation_matrix)

        return self._concatenate_flows(flow)
    
    


class SimilarityTransformation(RigidTransformation):
    r"""
    Similarity centred transformation for 2D and 3D.
    Args:
        moving_image (Image): moving image for the registration
        opt_cm (bool): using center of as parameter for the optimisation
    """
    def __init__(self, moving_image, opt_cm=False):
        super(SimilarityTransformation, self).__init__(moving_image, opt_cm)

        self._scale_x = Parameter(th.tensor(1.0))
        self._scale_y = Parameter(th.tensor(1.0))

        self._scale_matrix = None

        if self._dim == 2:
            self._compute_transformation = self._compute_transformation_2d
        else:
            self._compute_transformation = self._compute_transformation_3d

            self._scale_z = Parameter(th.tensor(1.0))

    def set_parameters(self, t, phi, scale, rotation_center=None):
        """
        Set parameters manually

        t (array): 2 or 3 dimensional array specifying the spatial translation
        phi (array): 1 or 3 dimensional array specifying the rotation angles
        scale (array): 2 or 3 dimensional array specifying the scale in each dimension
        rotation_center (array): 2 or 3 dimensional array specifying the rotation center (default is zeros)
        """
        super(SimilarityTransformation, self).set_parameters(t, phi, rotation_center)

        self._scale_x = Parameter(th.tensor(scale[0]).to(dtype=self._dtype, device=self._device))
        self._scale_y = Parameter(th.tensor(scale[1]).to(dtype=self._dtype, device=self._device))

        if len(t) == 2:
            self._compute_transformation_2d()
        else:
            self._scale_z = Parameter(th.tensor(scale[2]).to(dtype=self._dtype, device=self._device))
            self._compute_transformation_3d()

    def _compute_transformation_2d(self):

        super(SimilarityTransformation, self)._compute_transformation_2d()

        self._scale_matrix = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))

        self._scale_matrix[0, 0] = self._scale_x
        self._scale_matrix[1, 1] = self._scale_y

    def _compute_transformation_3d(self):

        super(SimilarityTransformation, self)._compute_transformation_3d()

        self._scale_matrix = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))

        self._scale_matrix[0, 0] = self._scale_x
        self._scale_matrix[1, 1] = self._scale_y
        self._scale_matrix[2, 2] = self._scale_z

    def _compute_transformation_matrix(self):
        transformation_matrix = th.mm(th.mm(th.mm(th.mm(self._trans_matrix_pos, self._trans_matrix_cm),
                                                  self._rotation_matrix), self._scale_matrix),
                                      self._trans_matrix_cm_rw)[0:self._dim, :]

        return transformation_matrix

    def forward(self):

        self._compute_transformation()
        transformation_matrix = self._compute_transformation_matrix()
        flow = self._compute_dense_flow(transformation_matrix)

        return self._concatenate_flows(flow)


class AffineTransformation(SimilarityTransformation):
    """
    Affine centred transformation for 2D and 3D.

    Args:
        moving_image (Image): moving image for the registration
        opt_cm (bool): using center of as parameter for the optimisation
    """
    def __init__(self, moving_image, opt_cm=False):
        super(AffineTransformation, self).__init__(moving_image, opt_cm)

        self._shear_y_x = Parameter(th.tensor(0.0))
        self._shear_x_y = Parameter(th.tensor(0.0))

        self._shear_matrix = None

        if self._dim == 2:
            self._compute_displacement = self._compute_transformation_2d
        else:
            self._compute_displacement = self._compute_transformation_3d

            self._shear_z_x = Parameter(th.tensor(0.0))
            self._shear_z_y = Parameter(th.tensor(0.0))
            self._shear_x_z = Parameter(th.tensor(0.0))
            self._shear_y_z = Parameter(th.tensor(0.0))

    def set_parameters(self, t, phi, scale, shear, rotation_center=None):
        """
        Set parameters manually

        t (array): 2 or 3 dimensional array specifying the spatial translation
        phi (array): 1 or 3 dimensional array specifying the rotation angles
        scale (array): 2 or 3 dimensional array specifying the scale in each dimension
        shear (array): 2 or 6 dimensional array specifying the shear in each dimension: yx, xy, zx, zy, xz, yz
        rotation_center (array): 2 or 3 dimensional array specifying the rotation center (default is zeros)
        """
        super(AffineTransformation, self).set_parameters(t, phi, scale, rotation_center)

        self._shear_y_x = Parameter(th.tensor(shear[0]).to(dtype=self._dtype, device=self._device))
        self._shear_x_y = Parameter(th.tensor(shear[1]).to(dtype=self._dtype, device=self._device))

        if len(t) == 2:
            self._compute_transformation_2d()
        else:
            self._shear_z_x = Parameter(th.tensor(shear[2]).to(dtype=self._dtype, device=self._device))
            self._shear_z_y = Parameter(th.tensor(shear[3]).to(dtype=self._dtype, device=self._device))
            self._shear_x_z = Parameter(th.tensor(shear[4]).to(dtype=self._dtype, device=self._device))
            self._shear_y_z = Parameter(th.tensor(shear[5]).to(dtype=self._dtype, device=self._device))
            self._compute_transformation_3d()

    def _compute_transformation_2d(self):

        super(AffineTransformation, self)._compute_transformation_2d()

        self._shear_matrix = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))

        self._shear_matrix[0, 1] = self._shear_y_x
        self._shear_matrix[1, 0] = self._shear_x_y

    def _compute_transformation_3d(self):

        super(AffineTransformation, self)._compute_transformation_3d()

        self._shear_matrix = th.diag(th.ones(self._dim + 1, dtype=self._dtype, device=self._device))

        self._shear_matrix[0, 1] = self._shear_y_x
        self._shear_matrix[0, 2] = self._shear_z_x
        self._shear_matrix[1, 0] = self._shear_x_y
        self._shear_matrix[1, 2] = self._shear_z_y
        self._shear_matrix[2, 0] = self._shear_x_z
        self._shear_matrix[2, 1] = self._shear_y_z

    def _compute_transformation_matrix(self):
        transformation_matrix = th.mm(th.mm(th.mm(th.mm(th.mm(self._trans_matrix_pos, self._trans_matrix_cm),
                                                        self._rotation_matrix),self._scale_matrix), self._shear_matrix),
                                      self._trans_matrix_cm_rw)[0:self._dim, :]

        return transformation_matrix

    def forward(self):

        self._compute_transformation()
        transformation_matrix = self._compute_transformation_matrix()
        flow = self._compute_dense_flow(transformation_matrix)

        return self._concatenate_flows(flow)


class NonParametricTransformation(_Transformation):
    r"""
        None parametric transformation
    """
    def __init__(self, image_size,  diffeomorphic=False, dtype=th.float32, device='cpu'):
        super(NonParametricTransformation, self).__init__(image_size, diffeomorphic, dtype, device)

        self._tensor_size = [self._dim] + self._image_size.tolist()

        self.trans_parameters = Parameter(th.Tensor(*self._tensor_size))
        self.trans_parameters.data.fill_(0)

        self.to(dtype=self._dtype, device=self._device)

        if self._dim == 2:
            self._compute_flow = self._compute_flow_2d
        else:
            self._compute_flow = self._compute_flow_3d

    def set_start_parameter(self, parameters):
        if self._dim == 2:
            self.trans_parameters = Parameter(th.tensor(parameters.transpose(0, 2)))
        elif self._dim == 3:
            self.trans_parameters = Parameter(th.tensor(parameters.transpose(0, 1)
                                                        .transpose(0, 2).transpose(0, 3)))

    def _compute_flow_2d(self):
        return self.trans_parameters.transpose(0, 2).transpose(0, 1)

    def _compute_flow_3d(self):
        return self.trans_parameters.transpose(0, 3).transpose(0, 2).transpose(0, 1)

    def forward(self):
        flow = self._concatenate_flows(self._compute_flow())

        if self._diffeomorphic:
            displacement = self._compute_diffeomorphic_displacement(flow)
        else:
            displacement = flow

        return displacement

"""
    Base class for kernel transformations
"""
class _KernelTransformation(_Transformation):
    def __init__(self, image_size, diffeomorphic=False, dtype=th.float32, device='cpu'):
        super(_KernelTransformation, self).__init__(image_size, diffeomorphic, dtype, device)

        self._kernel = None
        self._stride = 1
        self._padding = 0
        self._displacement_tmp = None
        self._displacement = None

        assert self._dim == 2 or self._dim == 3

        if self._dim == 2:
            self._compute_flow = self._compute_flow_2d
        else:
            self._compute_flow = self._compute_flow_3d


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

        size = [1, 1] + new_image_size.astype(dtype=int).tolist()
        self._displacement_tmp = th.empty(*size, dtype=self._dtype, device=self._device)

        size = [1, 1] + self._image_size.astype(dtype=int).tolist()
        self._displacement = th.empty(*size, dtype=self._dtype, device=self._device)

    def _compute_flow_2d(self):
        displacement_tmp = F.conv_transpose2d(self.trans_parameters, self._kernel,
                                          padding=self._padding, stride=self._stride, groups=2)

        # crop displacement
        return th.squeeze(displacement_tmp[:, :,
                       self._stride[0] + self._crop_start[0]:-self._stride[0] - self._crop_end[0],
                       self._stride[1] + self._crop_start[1]:-self._stride[1] - self._crop_end[1]].transpose_(1, 3).transpose(1, 2))

    def _compute_flow_3d(self):

        # compute dense displacement
        displacement = F.conv_transpose3d(self.trans_parameters, self._kernel,
                                          padding=self._padding, stride=self._stride, groups=3)

        # crop displacement
        return th.squeeze(displacement[:, :, self._stride[0] + self._crop_start[0]:-self._stride[0] - self._crop_end[0],
                                  self._stride[1] + self._crop_start[1]:-self._stride[1] - self._crop_end[1],
                                  self._stride[2] + self._crop_start[2]:-self._stride[2] - self._crop_end[2]
                                  ].transpose_(1,4).transpose_(1,3).transpose_(1,2))

    def forward(self):

        flow = self._concatenate_flows(self._compute_flow())

        if self._diffeomorphic:
            displacement = self._compute_diffeomorphic_displacement(flow)
        else:
            displacement = flow

        return displacement

"""
    bspline kernel transformation
"""
class BsplineTransformation(_KernelTransformation):
    def __init__(self, image_size, sigma, diffeomorphic=False, order=2, dtype=th.float32, device='cpu'):
        super(BsplineTransformation, self).__init__(image_size, diffeomorphic, dtype, device)

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
    """
    Wendland Kernel Transform:

    Implements the kernel transform with the Wendland basis

    Parameters:
        sigma: specifies how many control points are used (each sigma pixels)
        cp_scale: specifies the extent of the kernel. how many control points are in the support of the kernel
    """
    def __init__(self, image_size, sigma, cp_scale=2, diffeomorphic=False, ktype="C4", dtype=th.float32, device='cpu'):
        super(WendlandKernelTransformation, self).__init__(image_size, diffeomorphic, dtype, device)

        self._stride = np.array(sigma)

        # compute bspline kernel
        self._kernel = utils.wendland_kernel(np.array(sigma)*cp_scale, dim=self._dim, type=ktype, asTensor=True, dtype=dtype)

        self._padding = (np.array(self._kernel.size()) - 1) / 2

        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.expand(self._dim, *((np.ones(self._dim + 1,dtype=int) * -1).tolist()))
        self._kernel = self._kernel.to(dtype=dtype, device=self._device)

        self._initialize()
