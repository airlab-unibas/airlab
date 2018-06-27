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

from .. import transformation as T
from ..transformation import utils as tu
from ..utils import kernelFunction as utils






# Loss base class (standard from PyTorch)
class _PairwiseImageLoss(th.nn.modules.Module):
    def __init__(self, fixed_image, moving_image, size_average=True, reduce=True):
        super(_PairwiseImageLoss, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self.name = "parent"

        self._warped_moving_image = None
        self._weight = 1

        self._moving_image = moving_image
        self._fixed_image = fixed_image
        self._grid = None

        assert self._moving_image != None and self._fixed_image != None
        # TODO allow different image size for each image in the future
        assert self._moving_image.size == self._fixed_image.size
        assert self._moving_image.device == self._fixed_image.device
        assert len(self._moving_image.size) == 2 or len(self._moving_image.size) == 3

        self._grid = T.compute_grid(self._moving_image.size, dtype=self._moving_image.dtype,
                                     device=self._moving_image.device)

        self._dtype = self._moving_image.dtype
        self._device = self._moving_image.device

    def GetWarpedImage(self):
        return self._warped_moving_image[0, 0, ...].detach().cpu()

    def set_loss_weight(self, weight):
        self._weight = weight

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return tensor.mean()*self._weight
        if not self._size_average and self._reduce:
            return tensor.sum()*self._weight
        if not self.reduce:
            return tensor*self._weight


class MSELoss(_PairwiseImageLoss):
    r""" The mean square error loss is a simple and fast to compute point-wise measure
    which is well suited for monomodal image registration.

    .. math::
         \mathcal{S}_{\text{MSE}} := \frac{1}{\vert \mathcal{X} \vert}\sum_{x\in\mathcal{X}}
          \Big(I_M\big(x+f(x)\big) - I_F\big(x\big)\Big)^2

     fixed_image: Fixed image
    :parameter: fixed_image: Fixed image

    """
    def __init__(self, fixed_image, moving_image, size_average=True, reduce=True):
        super(MSELoss, self).__init__(fixed_image, moving_image, size_average, reduce)

        self.name = "mse"

        self.warped_moving_image = None

    def forward(self, displacement):

        displacement = self._grid + displacement

        mask = th.zeros_like(self._fixed_image.image, dtype=th.uint8, device=self._device)
        for dim in range(displacement.size()[-1]):
            mask += displacement[..., dim].gt(1) + displacement[..., dim].lt(-1)

        mask = mask == 0

        self.warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        value = (self.warped_moving_image - self._fixed_image.image).pow(2)

        value = th.masked_select(value, mask)

        return self.return_loss(value)


"""
    Normalized Cross Correlation image loss
"""
class NCCLoss(_PairwiseImageLoss):
    def __init__(self, fixed_image, moving_image):
        super(NCCLoss, self).__init__(fixed_image, moving_image, False, False)

        self.name = "ncc"

        self.warped_moving_image = th.empty_like(self._moving_image.image, dtype=self._dtype, device=self._device)

    def forward(self, displacement):

        displacement = self._grid + displacement

        mask = th.zeros_like(self._fixed_image.image, dtype=th.uint8, device=self._device)
        for dim in range(displacement.size()[-1]):
            mask += displacement[..., dim].gt(1) + displacement[..., dim].lt(-1)

        mask = mask == 0

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        moving_image_valid = th.masked_select(self._warped_moving_image, mask)
        fixed_image_valid = th.masked_select(self._fixed_image.image, mask)


        value = -1.*th.sum((fixed_image_valid - th.mean(moving_image_valid))*(moving_image_valid - th.mean(moving_image_valid)))\
                /th.sqrt(th.sum((fixed_image_valid - th.mean(moving_image_valid))**2)*th.sum((moving_image_valid - th.mean(moving_image_valid))**2) + 1e-10)

        return value


"""
    Local Normaliced Cross Corelation Image Loss
"""
class LCCLoss(_PairwiseImageLoss):
    def __init__(self, fixed_image, moving_image, sigma=3, kernel_type="box", size_average=True, reduce=True):
        super(LCCLoss, self).__init__(fixed_image, moving_image, size_average, reduce)

        self.name = "lcc"
        self.warped_moving_image = th.empty_like(self._moving_image.image, dtype=self._dtype, device=self._device)
        self._kernel = None

        dim = len(self._moving_image.size)
        sigma = np.array(sigma)

        if sigma.size != dim:
            sigma_app = sigma[-1]
            while sigma.size != dim:
                sigma = np.append(sigma, sigma_app)

        if kernel_type == "box":
            kernel_size = sigma*2 + 1
            self._kernel = th.ones(*kernel_size.tolist(), dtype=self._dtype, device=self._device) \
                           / float(np.product(kernel_size)**2)
        elif kernel_type == "gaussian":
            self._kernel = utils.gaussian_kernel(sigma, dim, asTensor=True, dtype=self._dtype, device=self._device)

        self._kernel.unsqueeze_(0).unsqueeze_(0)

        if dim == 2:
            self._lcc_loss = self._lcc_loss_2d  # 2d lcc

            self._mean_fixed_image = F.conv2d(self._fixed_image.image, self._kernel)
            self._variance_fixed_image = F.conv2d(self._fixed_image.image.pow(2), self._kernel) \
                                         - (self._mean_fixed_image.pow(2))
        elif dim == 3:
            self._lcc_loss = self._lcc_loss_3d  # 3d lcc

            self._mean_fixed_image = F.conv3d(self._fixed_image.image, self._kernel)
            self._variance_fixed_image = F.conv3d(self._fixed_image.image.pow(2), self._kernel) \
                                         - (self._mean_fixed_image.pow(2))


    def _lcc_loss_2d(self, warped_image, mask):


        mean_moving_image = F.conv2d(warped_image, self._kernel)
        variance_moving_image = F.conv2d(warped_image.pow(2), self._kernel) - (mean_moving_image.pow(2))

        mean_fixed_moving_image = F.conv2d(self._fixed_image.image * warped_image, self._kernel)

        cc = (mean_fixed_moving_image - mean_moving_image*self._mean_fixed_image)**2 \
             / (variance_moving_image*self._variance_fixed_image + 1e-10)

        mask = F.conv2d(mask, self._kernel)
        mask = mask == 0

        return -1.0*th.masked_select(cc, mask)

    def _lcc_loss_3d(self, warped_image, mask):

        mean_moving_image = F.conv3d(warped_image, self._kernel)
        variance_moving_image = F.conv3d(warped_image.pow(2), self._kernel) - (mean_moving_image.pow(2))

        mean_fixed_moving_image = F.conv3d(self._fixed_image.image * warped_image, self._kernel)

        cc = (mean_fixed_moving_image - mean_moving_image*self._mean_fixed_image)**2\
             /(variance_moving_image*self._variance_fixed_image + 1e-10)

        mask = F.conv3d(mask, self._kernel)
        mask = mask == 0

        return -1.0 * th.masked_select(cc, mask)

    def forward(self, displacement):

        displacement = self._grid + displacement

        mask = th.zeros_like(self._fixed_image.image, dtype=th.uint8, device=self._device)
        for dim in range(displacement.size()[-1]):
            mask += displacement[..., dim].gt(1) + displacement[..., dim].lt(-1)

        mask = mask > 0
        mask = mask.to(dtype=self._dtype, device=self._device)

        self._warped_moving_image = F.grid_sample(self._moving_image.image, displacement)

        return self.return_loss(self._lcc_loss(self._warped_moving_image, mask))

