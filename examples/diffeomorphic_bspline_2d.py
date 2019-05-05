
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

from create_test_image_data import create_C_2_O_test_images

def main():
	start = time.time()

	# set the used data type
	dtype = th.float32
	# set the device for the computaion to CPU
	device = th.device("cpu")

	# In order to use a GPU uncomment the following line. The number is the device index of the used GPU
	# Here, the GPU with the index 0 is used.
	# device = th.device("cuda:0")

	# create test image data
	fixed_image, moving_image, shaded_image = create_C_2_O_test_images(256, dtype=dtype, device=device)

	# create image pyramide size/4, size/2, size/1
	fixed_image_pyramid = al.create_image_pyramid(fixed_image, [[4, 4], [2, 2]])
	moving_image_pyramid = al.create_image_pyramid(moving_image, [[4, 4], [2, 2]])

	constant_displacement = None
	regularisation_weight = [1, 5, 50]
	number_of_iterations = [500, 500, 500]
	sigma = [[11, 11], [11, 11], [3, 3]]

	for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):

		registration = al.PairwiseRegistration(verbose=True)

		# define the transformation
		transformation = al.transformation.pairwise.BsplineTransformation(mov_im_level.size,
																		  sigma=sigma[level],
																		  order=3,
																		  dtype=dtype,
																		  device=device,
																		  diffeomorphic=True)

		if level > 0:
			constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
																				  mov_im_level.size,
																				  interpolation="linear")
			transformation.set_constant_flow(constant_flow)

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

		constant_flow = transformation.get_flow()

	# create final result
	displacement = transformation.get_displacement()
	warped_image = al.transformation.utils.warp_image(shaded_image, displacement)
	displacement = al.create_displacement_image_from_image(displacement, moving_image)


	# create inverse displacement field
	inverse_displacement = transformation.get_inverse_displacement()
	inverse_warped_image = al.transformation.utils.warp_image(warped_image, inverse_displacement)
	inverse_displacement = al.create_displacement_image_from_image(inverse_displacement, moving_image)

	end = time.time()

	print("=================================================================")

	print("Registration done in: ", end - start)
	print("Result parameters:")

	# plot the results
	plt.subplot(241)
	plt.imshow(fixed_image.numpy(), cmap='gray')
	plt.title('Fixed Image')

	plt.subplot(242)
	plt.imshow(moving_image.numpy(), cmap='gray')
	plt.title('Moving Image')

	plt.subplot(243)
	plt.imshow(warped_image.numpy(), cmap='gray')
	plt.title('Warped Shaded Moving Image')

	plt.subplot(244)
	plt.imshow(displacement.magnitude().numpy(), cmap='jet')
	plt.title('Magnitude Displacement')

	# plot the results
	plt.subplot(245)
	plt.imshow(warped_image.numpy(), cmap='gray')
	plt.title('Warped Shaded Moving Image')

	plt.subplot(246)
	plt.imshow(shaded_image.numpy(), cmap='gray')
	plt.title('Shaded Moving Image')

	plt.subplot(247)
	plt.imshow(inverse_warped_image.numpy(), cmap='gray')
	plt.title('Inverse Warped Shaded Moving Image')

	plt.subplot(248)
	plt.imshow(inverse_displacement.magnitude().numpy(), cmap='jet')
	plt.title('Magnitude Inverse Displacement')

	plt.show()

if __name__ == '__main__':
	main()
