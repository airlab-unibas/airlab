
import numpy as np


"""
Create a two dimensional coordinate grid
"""
def _compute_coordinate_grid_2d(image):

    x = np.linspace(0, image.size[0] - 1, num=image.size[0])
    y = np.linspace(0, image.size[1] - 1, num=image.size[1])

    x_m, y_m = np.meshgrid(y, x)

    return [x_m, y_m]

"""
Create a three dimensional coordinate grid
"""
def _compute_coordinate_grid_3d(image):

    x = np.linspace(0, image.size[0] - 1, num=image.size[0])
    y = np.linspace(0, image.size[1] - 1, num=image.size[1])
    z = np.linspace(0, image.size[2] - 1, num=image.size[2])

    x_m, y_m, z_m = np.meshgrid(y, x, z)

    return [x_m, y_m, z_m]


def CenterOfMass(image):
    # input is an airlab image

    # average over image coordinates by average their contribution to the average
    # with the image intensity at their respective location

    num_points = np.prod(image.size)
    coordinate_value_array = np.zeros([num_points, len(image.size)+1]) # allocate coordinate value array

    values = image.image.squeeze().numpy().reshape(num_points)  # vectorize image
    coordinate_value_array[:, 0] = values

    if len(image.size)==2:
        X, Y = _compute_coordinate_grid_2d(image)
        coordinate_value_array[:, 1] = X.reshape(num_points)
        coordinate_value_array[:, 2] = Y.reshape(num_points)

    elif len(image.size)==3:
        Y, Z, X = _compute_coordinate_grid_3d(image)
        coordinate_value_array[:, 1] = X.reshape(num_points)
        coordinate_value_array[:, 2] = Y.reshape(num_points)
        coordinate_value_array[:, 3] = Z.reshape(num_points)

    else:
        raise Exception("Only 2 and 3 space dimensions supported")

    # compared to the itk implementation the
    #   center of gravity for the 2d lenna should be [115.626, 91.9961]
    #   center of gravity for 3d image it should be [2.17962, 5.27883, -1.81531]

    cm = np.average(coordinate_value_array[:, 1:], axis=0, weights=coordinate_value_array[:, 0])
    cm = cm * image.spacing + image.origin

    # return center of mass
    return cm

# TODO: center of mass alignment
# TODO: calculate intersecting domain
# TODO: resample both images with same spacing (min of both) using resampleimagefilter of simpleITK
# TODO: default value is equal to padding
