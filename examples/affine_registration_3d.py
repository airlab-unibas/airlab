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

import sys
import os
import time

import matplotlib.pyplot as plt
import torch as th

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al

def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    # create 3D image volume with two objects
    object_shift = 10

    fixed_image = th.zeros(64, 64, 64).to(device=device)
    fixed_image[16:32, 16:32, 16:32] = 1.0
    fixed_image = al.Image(fixed_image, [64, 64, 64], [1, 1, 1], [0, 0, 0])

    moving_image = th.zeros(64, 64, 64).to(device=device)
    moving_image[16 - object_shift:32 - object_shift, 16 - object_shift:32 - object_shift,
    16 - object_shift:32 - object_shift] = 1.0
    moving_image = al.Image(moving_image, [64, 64, 64], [1, 1, 1], [0, 0, 0])

    # create pairwise registration object
    registration = al.PairwiseRegistration()

    # choose the affine transformation model
    transformation = al.transformation.pairwise.RigidTransformation(moving_image, opt_cm=True)
    transformation.init_translation(fixed_image)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)

    registration.set_image_loss([image_loss])

    # choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.1)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(500)

    # start the registration
    registration.start()

    # set the intensities for the visualisation
    fixed_image.image = 1 - fixed_image.image
    moving_image.image = 1 - moving_image.image

    # warp the moving image with the final transformation result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)

    end = time.time()

    print("=================================================================")

    print("Registration done in: ", end - start, " s")
    print("Result parameters:")
    transformation.print()

    # sitk.WriteImage(warped_image.itk(), '/tmp/rigid_warped_image.vtk')
    # sitk.WriteImage(moving_image.itk(), '/tmp/rigid_moving_image.vtk')
    # sitk.WriteImage(fixed_image.itk(), '/tmp/rigid_fixed_image.vtk')

    # plot the results
    plt.subplot(131)
    plt.imshow(fixed_image.numpy()[16, :, :], cmap='gray')
    plt.title('Fixed Image Slice')

    plt.subplot(132)
    plt.imshow(moving_image.numpy()[16, :, :], cmap='gray')
    plt.title('Moving Image Slice')

    plt.subplot(133)
    plt.imshow(warped_image.numpy()[16, :, :], cmap='gray')
    plt.title('Warped Moving Image Slice')

    plt.show()

if __name__ == '__main__':
    main()