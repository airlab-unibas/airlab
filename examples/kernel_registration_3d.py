
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

import os, sys, time
import multiprocessing as mp
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(mp.cpu_count())

import torch as th

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al


def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    #device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    device = th.device("cuda:0")

    # directory to store results
    tmp_directory = "/tmp/"

    # load the image data and normalize intensities to [0, 1]
    loader = al.ImageLoader(tmp_directory)

    print("loading images")
    fixed_image = loader.load("4DCT_POPI_1", "image_00").to(dtype, device)
    moving_image = loader.load("4DCT_POPI_5", "image_00").to(dtype, device)

    print("preprocessing images")
    (fixed_image, fixed_body_mask) = al.RemoveBedFilter(fixed_image)
    fixed_image.image -= fixed_image.image.min()
    fixed_image.image /= fixed_image.image.max()

    (moving_image, moving_body_mask) = al.RemoveBedFilter(moving_image)
    moving_image.image -= moving_image.image.min()
    moving_image.image /= moving_image.image.max()


    f_image, f_mask, m_image, m_mask = al.get_joint_domain_images(fixed_image, moving_image, cm_alignment=True, compute_masks=True)


    # create image pyramid size/8 size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(f_image, [[8, 8, 8], [4, 4, 4], [2, 2, 2]])
    fixed_mask_pyramid = al.create_image_pyramid(f_mask, [[8, 8, 8], [4, 4, 4], [2, 2, 2]])
    moving_image_pyramid = al.create_image_pyramid(m_image, [[8, 8, 8], [4, 4, 4], [2, 2, 2]])
    moving_mask_pyramid = al.create_image_pyramid(m_mask, [[8, 8, 8], [4, 4, 4], [2, 2, 2]])


    constant_displacement = None
    regularisation_weight = [1, 10, 100, 1000]
    number_of_iterations = [300, 200, 100, 50]
    sigma = [[11, 11, 11], [11, 11, 11], [9, 9, 9], [9, 9, 9]]

    print("perform registration")
    for level, (mov_im_level, mov_msk_level, fix_im_level, fix_msk_level) in enumerate(zip(moving_image_pyramid, moving_mask_pyramid, fixed_image_pyramid, fixed_mask_pyramid)):


        print("---- Level "+str(level)+" ----")
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
        image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level, mov_msk_level, fix_msk_level)

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

        # store current displacement field
        constant_displacement = transformation.get_displacement()


    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(m_image, displacement)
    displacement = al.transformation.utils.unit_displacement_to_dispalcement(displacement) # unit measures to image domain measures
    displacement = al.create_displacement_image_from_image(displacement, m_image)

    end = time.time()


    # write result images
    print("writing results")
    warped_image.write('/tmp/bspline_warped_image.vtk')
    m_image.write('/tmp/bspline_moving_image.vtk')
    m_mask.write('/tmp/bspline_moving_mask.vtk')
    f_image.write('/tmp/bspline_fixed_image.vtk')
    f_mask.write('/tmp/bspline_fixed_mask.vtk')
    displacement.write('/tmp/bspline_displacement_image.vtk')


    print("=================================================================")
    print("Registration done in: ", end - start, " seconds")




if __name__ == '__main__':
    main()