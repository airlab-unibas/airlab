
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
    """
    Returns the center of mass of the image (weighted average of coordinates where the intensity values serve as weights)

    image (Image): input is an airlab image
    return (array): coordinates of the center of mass
    """

    num_points = np.prod(image.size)
    coordinate_value_array = np.zeros([num_points, len(image.size)+1])  # allocate coordinate value array

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

    return cm


def GetJointDomainImages(fixed_image, moving_image, default_value=0, interpolator=2):
    """
    The method brings the fixed and moving image in a common image domain in order to be compatible with the
    registration framework of airlab. Different from the ITK convention, the registration in airlab is performed
    on pixels and not on points. This allows an efficient evaluation of the image metrics and the synthesis of
    displacement fields.

    Step 1: The moving image is aligned to the fixed image by matching the center of mass of the two images.
    Step 2: The new image domain is the smallest possible domain where both images are contained completely.
            The minimum spacing is taken as new spacing. This second step can increase the amount of pixels.
    Step 3: Fixed and moving image are resampled on this new domain.
    Step 4: Masks are built which defines in which region the respective image is not defined on this new domain.

    Note: The minimum possible value of the fixed image type is used as placeholder when resampling.
          Hence, this value should not be present in the images

    fixed_image (Image): fixed image provided as airlab image
    moving_image (Image): moving image provided as airlab image
    default_value (float|int): default value which defines the value which is set where the images are not defined in the new domain
    interpolator (int):  nn=1, linear=2, bspline=3
    return (tuple): resampled fixed image, fixed mask, resampled moving image, moving mask
    """

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

    # Resample images
    # fixed and moving image are resampled in new domain
    # the default value for resampling is set to a predefined value
    # (minimum possible value of the fixed image type) to use it
    # to create masks. At the end, default values are replaced with
    # the provided default value
    minimum_value = float(np.finfo(fixed_image.image.numpy().dtype).tiny)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size.tolist())
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetDefaultPixelValue(minimum_value)
    resampler.SetInterpolator(interpolator)

    # resample fixed and moving image
    f_image = al.Image(resampler.Execute(fixed_image.itk()))
    m_image = al.Image(resampler.Execute(moving_image.itk()))

    # create masks
    f_mask = np.ones_like(f_image.image)
    m_mask = np.ones_like(m_image.image)

    f_mask[np.where(f_image.image == minimum_value)] = 0
    m_mask[np.where(m_image.image == minimum_value)] = 0

    f_image.image[np.where(f_image.image == minimum_value)] = default_value
    m_image.image[np.where(m_image.image == minimum_value)] = default_value

    return f_image, f_mask, m_image, m_mask


