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
import torch.nn.functional as F
import numpy as np
from ..utils import graph as G
from ..utils import matrix as mat
from ..utils import kernelFunction as utils
from ..utils import image as iu

class _DemonsRegulariser():
    def __init__(self, pixel_spacing, dtype=th.float32, device='cpu'):
        super(_DemonsRegulariser, self).__init__()

        self._dtype = dtype
        self._device = device
        self._weight = 1
        self._dim = len(pixel_spacing)
        self._pixel_spacing = pixel_spacing
        self.name = "parent"



class GaussianRegulariser(_DemonsRegulariser):
    def __init__(self, pixel_spacing, sigma, dtype=th.float32, device='cpu'):
        super(GaussianRegulariser, self).__init__(pixel_spacing, dtype=dtype, device=device)

        sigma = np.array(sigma)

        if sigma.size != self._dim:
            sigma_app = sigma[-1]
            while sigma.size != self._dim:
                sigma = np.append(sigma, sigma_app)


        self._kernel = utils.gaussian_kernel(sigma, self._dim, asTensor=True, dtype=dtype, device=device)

        self._padding = (np.array(self._kernel.size()) - 1) / 2
        self._padding = self._padding.astype(dtype=int).tolist()

        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.expand(self._dim, *((np.ones(self._dim + 1, dtype=int) * -1).tolist()))
        self._kernel = self._kernel.to(dtype=dtype, device=self._device)

        if self._dim == 2:
            self._regulariser = self._regularise_2d
        elif self._dim == 3:
            self._regulariser = self._regularise_3d


    def _regularise_2d(self, data):

        data.data = data.data.unsqueeze(0)
        data.data = F.conv2d(data.data, self._kernel.contiguous(), padding=self._padding, groups=2)
        data.data = data.data.squeeze()


    def _regularise_3d(self, data):

        data.data = data.data.unsqueeze(0)
        data.data = F.conv3d(data.data, self._kernel, padding=self._padding, groups=3)
        data.data = data.data.squeeze()

    def regularise(self, data):
        for parameter in data:
            # no gradient calculation for the demons regularisation
            with th.no_grad():
                self._regulariser(parameter)



class _GraphEdgeWeightUpdater():
    def __init__(self, pixel_spacing, edge_window=0.9, edge_mean=False):

        self._edge_window = edge_window
        self._edge_mean = edge_mean
        self._laplace_matrix = None
        self._dim = len(pixel_spacing)
        self._pixel_spacing = pixel_spacing
        self._collapse_threshold = 0
        self._detect_node_collapse = True

    def detect_node_collapse(self, detect):
        self._detect_node_collapse = detect


    def set_laplace_matrix(self, laplace_matrix):
        self._laplace_matrix = laplace_matrix

    def remove_node_collapse(self):
            for i, diag in enumerate(self._laplace_matrix.diag_elements):
                node_value = self._laplace_matrix.main_diag[diag.edge_index[-1]]
                index = th.abs(node_value) < self._collapse_threshold
                diag.edge_values[index] = 1




class EdgeUpdaterIntensities(_GraphEdgeWeightUpdater):
    def __init__(self, pixel_spacing, image, scale=1, edge_window=0.9, edge_mean=False):
        super(EdgeUpdaterIntensities, self).__init__(pixel_spacing, edge_window, edge_mean)

        self._image = image
        self._scale = scale

    def set_scale(self, sale):
        self._scale = scale

    def update(self, data):

        if self._dim == 2:
            for i, diag in enumerate(self._laplace_matrix.diag_elements):
                one = th.zeros(self._dim, dtype=th.int64, device=self._image.device)
                one[i] = 1

                intensyties_A = self._image[0, 0, diag.edge_index[0], diag.edge_index[1]]
                intensyties_B = self._image[0, 0, diag.edge_index[0] - one[0], diag.edge_index[1] - one[1]]

                diag.edge_values = (th.exp(-self._scale*th.abs(intensyties_A - intensyties_B)))

        elif self._dim == 3:
            for i, diag in enumerate(self._laplace_matrix.diag_elements):
                one = th.zeros(self._dim, dtype=th.int64, device=self._image.device)
                one[i] = 1

                intensyties_A = self._image[0, 0, diag.edge_index[0], diag.edge_index[1], diag.edge_index[2]]
                intensyties_B = self._image[0, 0, diag.edge_index[0] - one[0], diag.edge_index[1] - one[1],
                                            diag.edge_index[2] - one[2]]

                diag.edge_values = (th.exp(-self._scale*th.abs(intensyties_A - intensyties_B)))

        # update the laplace matrix
        self._laplace_matrix.update()



class EdgeUpdaterDisplacementIntensities(_GraphEdgeWeightUpdater):
    def __init__(self, pixel_spacing, image, edge_window=0.9, edge_mean=False):
        super(EdgeUpdaterDisplacementIntensities, self).__init__(pixel_spacing, edge_window, edge_mean)

        self._image = image[0, 0, ...]

        self._image_gradient = None
        self._scale_int_diff = 1
        self._scale_disp_diff = 1
        self._scale_disp = 1

        if self._dim == 2:
            data_pad = th.nn.functional.pad(self._image, pad=(1, 0, 1, 0))  # , mode='replicate'
            dx = data_pad[1:, 1:] - data_pad[:-1, 1:]
            dy = data_pad[1:, 1:] - data_pad[:1, -1:]

            self._image_gradient = th.stack((dx, dy), 2)

    def update(self, data):

        if self._dim == 2:

            for i, diag in enumerate(self._laplace_matrix.diag_elements):
                one = th.zeros(self._dim, dtype=th.int64, device=self._image.device)
                one[i] = 1

                intensyties_A = self._image[diag.edge_index[0], diag.edge_index[1]]
                intensyties_B = self._image[diag.edge_index[0] - one[0], diag.edge_index[1] - one[1]]

                intensity_diff = th.exp(-th.abs(intensyties_A - intensyties_B)*self._scale_int_diff)

                del intensyties_A, intensyties_B

                displacement_A = data[:, diag.edge_index[0], diag.edge_index[1]]
                displacement_B = data[:, diag.edge_index[0] - one[0], diag.edge_index[1] - one[1]]

                displacement_diff = displacement_A - displacement_B
                displacement_diff = th.sqrt(displacement_diff[0, :]**2 + displacement_diff[1, :]**2)
                displacement_diff = th.exp(-self._scale_disp_diff*displacement_diff)

                norm_disp_A = th.sqrt(displacement_A[0, ...]**2 + displacement_A[1, ...]**2)
                norm_disp_B = th.sqrt(displacement_B[0, ...]**2 + displacement_B[1, ...]**2)

                image_gradient_A = self._image_gradient[diag.edge_index[0], diag.edge_index[1], :]
                image_gradient_B = self._image_gradient[diag.edge_index[0] - one[0], diag.edge_index[1] - one[1], :]

                norm_A = th.sqrt(image_gradient_A[..., 0]**2 + image_gradient_A[..., 1]**2)
                norm_B = th.sqrt(image_gradient_B[..., 0]**2 + image_gradient_B[..., 1]**2)

                max_norm = th.max(norm_A, norm_B)

                max_grad = th.zeros_like(image_gradient_A)
                index = (norm_A - max_norm) == 0
                max_grad[index] = image_gradient_A[index]
                max_grad[1 - index] = image_gradient_B[1 - index]

                del index, image_gradient_A, image_gradient_B

                phi_A = th.div(th.sum(th.mul(max_grad, displacement_A.t()), dim=1), th.mul(norm_disp_A, max_norm) + 1e-10)
                phi_B = th.div(th.sum(th.mul(max_grad, displacement_B.t()), dim=1), th.mul(norm_disp_B, max_norm) + 1e-10)

                weight = th.mul(intensity_diff,displacement_diff) + (1-intensity_diff)*((phi_A + phi_B)*0.5)
                weight = weight*(1 - self._scale_disp) + displacement_diff*self._scale_disp

                if self._edge_mean:
                    diag.edge_values = diag.edge_values*self._edge_window + th.round(weight)*(1. - self._edge_window)
                else:
                    diag.edge_values = weight

        elif self._dim == 3:
            for i, diag in enumerate(self._laplace_matrix.diag_elements):
                one = th.zeros(self._dim, dtype=th.int64, device=self._image.device)
                one[i] = 1

                intensyties_A = self._image[diag.edge_index[0], diag.edge_index[1]]
                intensyties_B = self._image[diag.edge_index[0] - one[0], diag.edge_index[1] - one[1],
                                            diag.edge_index[2] - one[2]]

                intensity_diff = th.exp(-th.abs(intensyties_A - intensyties_B)*self._scale_int_diff)

                del intensyties_A, intensyties_B

                displacement_A = data[:, diag.edge_index[0], diag.edge_index[1]]
                displacement_B = data[:, diag.edge_index[0] - one[0], diag.edge_index[1] - one[1],
                                      diag.edge_index[2] - one[2]]

                displacement_diff = displacement_A - displacement_B
                displacement_diff = th.sqrt(displacement_diff[0, :]**2 + displacement_diff[1, :]**2 + displacement_diff[2, :]**2)
                displacement_diff = th.exp(-self._scale_disp_diff*displacement_diff)

                norm_disp_A = th.sqrt(displacement_A[0, ...]**2 + displacement_A[1, ...]**2 + displacement_A[2, ...]**2)
                norm_disp_B = th.sqrt(displacement_B[0, ...]**2 + displacement_B[1, ...]**2 + displacement_B[2, ...]**2)

                image_gradient_A = self._image_gradient[diag.edge_index[0], diag.edge_index[1], :]
                image_gradient_B = self._image_gradient[diag.edge_index[0] - one[0], diag.edge_index[1] - one[1],
                                                        diag.edge_index[2] - one[2], :]

                norm_A = th.sqrt(image_gradient_A[..., 0]**2 + image_gradient_A[..., 1]**2 + image_gradient_A[..., 2]**2)
                norm_B = th.sqrt(image_gradient_B[..., 0]**2 + image_gradient_B[..., 1]**2 + image_gradient_A[..., 2]**2)

                max_norm = th.max(norm_A, norm_B)

                del norm_A, norm_B

                max_grad = th.zeros_like(image_gradient_A)
                index = (norm_A - max_norm) == 0
                max_grad[index] = image_gradient_A[index]
                max_grad[1 - index] = image_gradient_B[1 - index]

                del index, image_gradient_A, image_gradient_B

                phi_A = th.div(th.sum(th.mul(max_grad, displacement_A.t()), dim=1), th.mul(norm_disp_A, max_norm) + 1e-10)
                phi_B = th.div(th.sum(th.mul(max_grad, displacement_B.t()), dim=1), th.mul(norm_disp_B, max_norm) + 1e-10)

                weight = th.mul(intensity_diff,displacement_diff) + (1-intensity_diff)*((phi_A + phi_B)*0.5)
                weight = weight*(1 - self._scale_disp) + displacement_diff*self._scale_disp

                if self._edge_mean:
                    diag.edge_values = diag.edge_values*self._edge_window + th.round(weight)*(1. - self._edge_window)
                else:
                    diag.edge_values = weight

        # remove collapsed nodes
        if self._detect_node_collapse:
            self.remove_node_collapse()





class GraphDiffusionRegulariser(_DemonsRegulariser):
    def __init__(self, image_size, pixel_spacing, edge_updater, phi=1, dtype=th.float32, device='cpu'):
        super(GraphDiffusionRegulariser, self).__init__(pixel_spacing, dtype=dtype, device=device)

        self._graph = G.Graph(image_size, dtype=dtype, device=device)

        self._edge_updater = edge_updater
        self._edge_updater.set_laplace_matrix(self._graph.laplace_matrix)

        self._phi = phi

        self._krylov_dim = 30

        self._image_size = image_size

    def set_krylov_dim(self, krylov_dim):
        self._krylov_dim = krylov_dim

    def get_edge_image(self):
        main_diag_laplace = th.reshape(self._graph.laplace_matrix.main_diag, self._image_size)

        return iu.Image(main_diag_laplace.unsqueeze_(0).unsqueeze(0), self._image_size, self._pixel_spacing, th.zeros(len(self._image_size))) # only zero origin supported yet



    def regularise(self, data):
        for parameter in data:
            # no gradient calculation for the demons regularisation
            with th.no_grad():
                dim = parameter.size()[0]

                # compute the graph diffusion regularisation for each dimension
                for i in range(dim):
                    mat.expm_krylov(self._graph.laplace_matrix, parameter.data[i, ...].view(-1),
                                    phi=self._phi, krylov_dim=self._krylov_dim)

                # update the edge weights on the curren data
                self._edge_updater.update(parameter.data)













