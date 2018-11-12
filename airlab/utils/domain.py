
import numpy as np


"""
Create a two dimensional coordinate grid
"""
def _compute_coordinate_grid_2d(image):

    x = np.linspace(image.origin[0], image.origin[0] + (image.size[0] - 1) * image.spacing[0], num=image.size[0])
    y = np.linspace(image.origin[1], image.origin[1] + (image.size[1] - 1) * image.spacing[1], num=image.size[1])

    x_m, y_m = np.meshgrid(y, x)

    return [x_m, y_m]

"""
Create a three dimensional coordinate grid
"""
def _compute_coordinate_grid_3d(image):

    x = np.linspace(image.origin[0], image.origin[0] + (image.size[0] - 1) * image.spacing[0], num=image.size[0])
    y = np.linspace(image.origin[1], image.origin[1] + (image.size[1] - 1) * image.spacing[1], num=image.size[1])
    z = np.linspace(image.origin[2], image.origin[2] + (image.size[2] - 1) * image.spacing[1], num=image.size[2])

    x_m, y_m, z_m = np.meshgrid(y, x, z)

    return [x_m, y_m, z_m]


def CenterOfMass(image):
    # input is an airlab image

    # average over image coordinates by average their contribution to the average
    # with the image intensity at their respective location

    # nonZeroMasses = masses[numpy.nonzero(masses[:,3])] # not really needed
    # CM = numpy.average(nonZeroMasses[:,:3], axis=0, weights=nonZeroMasses[:,3])

    num_points = np.prod(image.size)
    coordinate_value_array = np.zeros([num_points, len(image.size)+1]) # allocate coordinate value array

    values = image.image.reshape(num_points)  # vectorize image
    coordinate_value_array[:, 0] = values

    if len(image.size)==2:
        X, Y = _compute_coordinate_grid_2d(image)
        coordinate_value_array[:, 1] = X.reshape(num_points)
        coordinate_value_array[:, 2] = Y.reshape(num_points)

    elif len(image.size)==3:
        Y, X, Z = _compute_coordinate_grid_3d(image)
        coordinate_value_array[:, 1] = X.reshape(num_points)
        coordinate_value_array[:, 2] = Y.reshape(num_points)
        coordinate_value_array[:, 3] = Z.reshape(num_points)

    else:
        raise Exception("Only 2 and 3 space dimensions supported")


    # return center of mass
    return np.average(coordinate_value_array[:,1:], axis=0, weights=coordinate_value_array[:,0])


