
import numpy as np
import SimpleITK as sitk
import airlab as al

"""
Create a two dimensional coordinate grid
"""
def _compute_coordinate_grid_2d(image):

    x = np.linspace(0, image.size[0] - 1, num=image.size[0])
    y = np.linspace(0, image.size[1] - 1, num=image.size[1])

    y_m, x_m = np.meshgrid(y, x)

    return [x_m, y_m]

"""
Create a three dimensional coordinate grid
"""
def _compute_coordinate_grid_3d(image):

    x = np.linspace(0, image.size[0] - 1, num=image.size[0])
    y = np.linspace(0, image.size[1] - 1, num=image.size[1])
    z = np.linspace(0, image.size[2] - 1, num=image.size[2])

    y_m, x_m, z_m = np.meshgrid(y, x, z)

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
        X, Y, Z = _compute_coordinate_grid_3d(image)
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


def GetJointDomainImages(fixed_image, moving_image, default_value=0):

    # align images using center of mass
    moving_image.origin = moving_image.origin - CenterOfMass(moving_image) + CenterOfMass(fixed_image)

    # common origin
    origin = np.minimum(fixed_image.origin, moving_image.origin)

    # common extent
    f_extent = np.array(fixed_image.origin) + (np.array(fixed_image.size)-1)*np.array(fixed_image.spacing)
    m_extent = np.array(moving_image.origin) + (np.array(moving_image.size)-1)*np.array(moving_image.spacing)
    extent = np.maximum(f_extent, m_extent)

    # common spacing
    spacing = np.minimum(fixed_image.spacing, moving_image.spacing)

    # common size
    size = np.ceil((extent-origin)/spacing).astype(int)

    # create new images
    f_image = al.Image(np.zeros(size), size, spacing, origin)
    m_image = np.zeros(size)

    # create masks
    f_mask = np.zeros_like(f_image)
    m_mask = np.zeros_like(m_image)

    # start and end indizes for masks
    f_start = (origin - fixed_image.origin)/spacing
    f_end = size - (extent - f_extent)/spacing
    m_start = (origin - moving_image.origin) / spacing
    m_end = size - (extent - m_extent) / spacing

    # fill masks


    # resample fixed image
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetDefaultPixelValue(default_value)

    # resample fixed image
    resampler.SetInput(fixed_image.itk())
    resampler.UpdateLargestPossibleRegion()
    f_image = al.Image(sitk.GetArrayFromImage(resampler.GetOutput()), size, spacing, origin)

    # resample moving image
    resampler.SetInput(moving_image.itk())
    resampler.UpdateLargestPossibleRegion()
    m_image = al.Image(sitk.GetArrayFromImage(resampler.GetOutput()), size, spacing, origin)

    return f_image, f_mask, m_image, m_mask


