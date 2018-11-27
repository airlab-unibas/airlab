
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

import matplotlib.pyplot as plt
import time

import sys
import os

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
    #device = th.device("cuda:0")

    # load the image data and normalize to [0, 1]
    itkImg = sitk.ReadImage("./data/affine_test_image_2d_fixed.png", sitk.sitkFloat32)
    itkImg = sitk.RescaleIntensity(itkImg, 0, 1)
    fixed_image = al.create_tensor_image_from_itk_image(itkImg, dtype=dtype, device=device)

    itkImg = sitk.ReadImage("./data/affine_test_image_2d_moving.png", sitk.sitkFloat32)
    itkImg = sitk.RescaleIntensity(itkImg, 0, 1)
    moving_image = al.create_tensor_image_from_itk_image(itkImg, dtype=dtype, device=device)

    # create image pyramide size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(fixed_image, [[4, 4], [2, 2]])
    moving_image_pyramid = al.create_image_pyramid(moving_image, [[4, 4], [2, 2]])

    constant_displacement = None
    regularisation_weight = [1, 5, 50]
    number_of_iterations = [500, 500, 500]
    sigma = [[11, 11], [11, 11], [3, 3]]

    for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):

        registration = al.PairwiseRegistration(dtype=dtype, device=device)

        # define the transformation
        transformation = al.transformation.pairwise.BsplineTransformation(mov_im_level.size,
                                                                          sigma=sigma[level],
                                                                          order=3,
                                                                          dtype=dtype,
                                                                          device=device)

        if level > 0:
            constant_displacement = al.transformation.utils.upsample_displacement(constant_displacement,
                                                                                  mov_im_level.size,
                                                                                  interpolation="linear")

            transformation.set_constant_displacement(constant_displacement)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level)

        registration.set_image_loss([image_loss])

        # define the regulariser for the displacement
        regulariser = al.regulariser.displacement.DiffusionRegulariser(mov_im_level.spacing)
        regulariser.SetWeight(regularisation_weight[level])

        registration.set_regulariser_displacement([regulariser])

        #define the optimizer
        optimizer = th.optim.Adam(transformation.parameters())

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations[level])

        registration.start()

        constant_displacement = transformation.get_displacement()

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(moving_image, displacement)
    displacement = al.create_displacement_image_from_image(displacement, moving_image)

    end = time.time()

    print("=================================================================")

    print("Registration done in: ", end - start)
    print("Result parameters:")

    # plot the results
    plt.subplot(221)
    plt.imshow(fixed_image.numpy(), cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(222)
    plt.imshow(moving_image.numpy(), cmap='gray')
    plt.title('Moving Image')

    plt.subplot(223)
    plt.imshow(warped_image.numpy(), cmap='gray')
    plt.title('Warped Moving Image')

    plt.subplot(224)
    plt.imshow(displacement.magnitude().numpy(), cmap='jet')
    plt.title('Magnitude Displacement')

    plt.show()

    # write result images
    # sitk.WriteImage(warped_image.itk(), '/tmp/rigid_warped_image.vtk')
    # sitk.WriteImage(moving_image.itk(), '/tmp/rigid_moving_image.vtk')
    # sitk.WriteImage(fixed_image.itk(), '/tmp/rigid_fixed_image.vtk')
    # sitk.WriteImage(displacement.itk(), '/tmp/demons_displacement_image.vtk')




if __name__ == '__main__':
    main()