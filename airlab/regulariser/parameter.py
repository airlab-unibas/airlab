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

# Regulariser base class (standard from PyTorch)
class _ParameterRegulariser(th.nn.modules.Module):
    def __init__(self, parameter_name, size_average=True, reduce=True):
        super(_ParameterRegulariser, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._weight = 1
        self.name = "parent"
        self._parameter_name = parameter_name

    def SetWeight(self, weight):
        print("SetWeight is deprecated. Use set_weight instead.")
        self.set_weight(weight)

    def set_weight(self, weight):
        self._weight = weight

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return self._weight*tensor.mean()
        if not self._size_average and self._reduce:
            return self._weight*tensor.sum()
        if not self._reduce:
            return self._weight*tensor


"""
    Base class for spatial parameter regulariser
"""
class _SpatialParameterRegulariser(_ParameterRegulariser):
    def __init__(self, parameter_name, scaling=[1], size_average=True, reduce=True):
        super(_SpatialParameterRegulariser, self).__init__(parameter_name, size_average, reduce)

        self._dim = len(scaling)
        self._scaling = scaling
        if len(scaling) == 1:
            self._scaling = np.ones(self._dim)*self._scaling[0]

        self.name = "parent"

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return self._weight*tensor.mean()
        if not self._size_average and self._reduce:
            return self._weight*tensor.sum()
        if not self._reduce:
            return self._weight*tensor

"""
    Isotropic TV regularisation
"""
class IsotropicTVRegulariser(_SpatialParameterRegulariser):
    def __init__(self, parameter_name, scaling=[1], size_average=True, reduce=True):
        super(IsotropicTVRegulariser, self).__init__(parameter_name, scaling, size_average, reduce)

        self.name = "param_isoTV"

        if self._dim == 2:
            self._regulariser = self._regulariser_2d # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(2)*self._scaling[0]
                dy = (parameter[:, 1:, 1:] - parameter[:,  1:, :-1]).pow(2)*self._scaling[1]

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:, 1:] - parameter[:, -1, 1:, 1:]).pow(2)*self._scaling[0]
                dy = (parameter[:, 1:, 1:, 1:] - parameter[:, 1:, :-1, 1:]).pow(2)*self._scaling[1]
                dz = (parameter[:, 1:, 1:, 1:] - parameter[:, 1:, 1:, :-1]).pow(2)*self._scaling[2]

                return dx + dy + dz

    def forward(self, parameters):

        # set the supgradient to zeros
        value = self._regulariser(parameters)
        mask = value > 0
        value[mask] = th.sqrt(value[mask])

        return self.return_loss(value)


"""
    TV regularisation 
"""
class TVRegulariser(_SpatialParameterRegulariser):
    def __init__(self, parameter_name, scaling=[1], size_average=True, reduce=True):
        super(TVRegulariser, self).__init__(parameter_name, scaling, size_average, reduce)

        self.name = "param_TV"

        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = th.abs(parameter[:, 1:, 1:] - parameter[:, :-1, 1:])*self._pixel_spacing[0]
                dy = th.abs(parameter[:, 1:, 1:] - parameter[:,  1:, :-1])*self._pixel_spacing[1]

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = th.abs(parameter[:, 1:, 1:, 1:] - parameter[:, -1, 1:, 1:])*self._pixel_spacing[0]
                dy = th.abs(parameter[:, 1:, 1:, 1:] - parameter[:, 1:, :-1, 1:])*self._pixel_spacing[1]
                dz = th.abs(parameter[:, 1:, 1:, 1:] - parameter[:, 1:, 1:, :-1])*self._pixel_spacing[2]

                return dx + dy + dz

    def forward(self, parameters):
        return self.return_loss(self._regulariser(parameters))

"""
    Diffusion regularisation 
"""
class DiffusionRegulariser(_SpatialParameterRegulariser):
    def __init__(self, pixel_spacing, size_average=True, reduce=True):
        super(DiffusionRegulariser, self).__init__(pixel_spacing, size_average, reduce)

        self.name = "param diff"

        if self._dim == 2:
            self._regulariser = self._regulariser_2d  # 2d regularisation
        elif self._dim == 3:
            self._regulariser = self._regulariser_3d  # 3d regularisation

    def _regulariser_2d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:] - parameter[:, :-1, 1:]).pow(2) * self._pixel_spacing[0]
                dy = (parameter[:, 1:, 1:] - parameter[:,  1:, :-1]).pow(2) * self._pixel_spacing[1]

                return dx + dy

    def _regulariser_3d(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                dx = (parameter[:, 1:, 1:, 1:] - parameter[:, -1, 1:, 1:]).pow(2) * self._pixel_spacing[0]
                dy = (parameter[:, 1:, 1:, 1:] - parameter[:, 1:, :-1, 1:]).pow(2) * self._pixel_spacing[1]
                dz = (parameter[:, 1:, 1:, 1:] - parameter[:, 1:, 1:, :-1]).pow(2) * self._pixel_spacing[2]

                return dx + dy + dz

    def forward(self, displacement):
        return self.return_loss(self._regulariser(displacement))

"""
    Sparsity regularisation 
"""
class SparsityRegulariser(_ParameterRegulariser):
    def __init__(self, parameter_name, size_average=True, reduce=True):
        super(SparsityRegulariser, self).__init__(parameter_name, size_average, reduce)

        self.name = "param_L1"

    def forward(self, parameters):
        for name, parameter in parameters:
            if self._parameter_name in name:
                return self.return_loss(th.abs(parameter))

