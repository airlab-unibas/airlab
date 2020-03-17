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

"""
Create a two dimensional mesh grid
"""
def _compute_mesh_grid_2d(kernel_size):

    nx = int(kernel_size[0])
    ny = int(kernel_size[1])

    x = np.linspace(-(nx - 1)/2, (nx - 1)/2, num=nx)
    y = np.linspace(-(ny - 1)/2, (ny - 1)/2, num=ny)

    x_m, y_m = np.meshgrid(x, y)

    return [x_m, y_m]

"""
Create a three dimensional mesh grid
"""
def _compute_mesh_grid_3d(kernel_size):

    nx = int(kernel_size[0])
    ny = int(kernel_size[1])
    nz = int(kernel_size[2])

    x = np.linspace(-(nx - 1)/2, (nx - 1)/2, num=nx)
    y = np.linspace(-(ny - 1)/2, (ny - 1)/2, num=ny)
    z = np.linspace(-(nz - 1)/2, (nz - 1)/2, num=nz)

    x_m, y_m, z_m = np.meshgrid(x, y, z)

    return [x_m, y_m, z_m]

"""
Create a one dimensional gaussian kernel matrix
"""
def gaussian_kernel_1d(sigma, asTensor=False, dtype=th.float32, device='cpu'):

    kernel_size = int(2*np.ceil(sigma*2) + 1)

    x = np.linspace(-(kernel_size - 1) // 2, (kernel_size - 1) // 2, num=kernel_size)

    kernel = 1.0/(sigma*np.sqrt(2*np.pi))*np.exp(-(x**2)/(2*sigma**2))
    kernel = kernel/np.sum(kernel)

    if asTensor:
        return th.tensor(kernel, dtype=dtype, device=device)
    else:
        return kernel


"""
Create a two dimensional gaussian kernel matrix
"""
def gaussian_kernel_2d(sigma, asTensor=False, dtype=th.float32, device='cpu'):

    y_1 = gaussian_kernel_1d(sigma[0])
    y_2 = gaussian_kernel_1d(sigma[1])

    kernel = np.tensordot(y_1, y_2, 0)
    kernel = kernel / np.sum(kernel)

    if asTensor:
        return th.tensor(kernel, dtype=dtype, device=device)
    else:
        return kernel

"""
Create a three dimensional gaussian kernel matrix
"""
def gaussian_kernel_3d(sigma, asTensor=False, dtype=th.float32, device='cpu'):

    kernel_2d = gaussian_kernel_2d(sigma[:2])
    kernel_1d = gaussian_kernel_1d(sigma[-1])

    kernel = np.tensordot(kernel_2d, kernel_1d, 0)
    kernel = kernel / np.sum(kernel)

    if asTensor:
        return th.tensor(kernel, dtype=dtype, device=device)
    else:
        return kernel


"""
    Create a Gaussian kernel matrix
"""
def gaussian_kernel(sigma, dim=1, asTensor=False, dtype=th.float32, device='cpu'):

    assert dim > 0 and dim <=3

    if dim == 1:
        return gaussian_kernel_1d(sigma, asTensor=asTensor, dtype=dtype, device=device)
    elif dim == 2:
        return gaussian_kernel_2d(sigma, asTensor=asTensor, dtype=dtype, device=device)
    else:
        return gaussian_kernel_3d(sigma, asTensor=asTensor, dtype=dtype, device=device)

"""
Create a 1d Wendland kernel matrix
"""
def wendland_kernel_1d(sigma, type="C4", asTensor=False, dtype=th.float32, device='cpu'):

	kernel_size = sigma*2 + 1

	x = np.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, num=kernel_size)

	r = np.sqrt((x/float(sigma))**2)
	f = np.maximum(1 - r, 0)

	#kernel = ((f**6.0)*(3.0 + 18.0*r + 35.0*(r**2)))*(560.0/1680.0)
	if type=='C2':
		kernel = ((f**3.0)*(1.0 + 3.0*r))*5./4.
	elif type=='C4':
		kernel = ((f**5.0)*(1.0 + 5.0*r + 8.0*(r**2)/3.))*3./2.
	elif type=='C6':
		kernel = ((f**7.0)*(1.0 + 7.0*r + 19.0*(r**2) + 21*(r**3)))*55./32.
	else:
		raise ValueError(type)


	if asTensor:
		return th.tensor(kernel, dtype=dtype, device=device)
	else:
		return kernel

"""
Create a 2d Wendland kernel matrix
"""
def wendland_kernel_2d(sigma, type="C4", asTensor=False, dtype=th.float32, device='cpu'):

	kernel_size = np.array(sigma)*2 + 1

	xv, yv = _compute_mesh_grid_2d(kernel_size)

	r = np.sqrt((xv/sigma[0])**2 + (yv/sigma[1])**2)
	f = np.maximum(1 - r, 0)

	if type=='C2':
		kernel = ((f**4.0)*(1.0 + 4.0*r))*7./np.pi
	elif type=='C4':
		kernel = ((f**6.0)*(1.0 + 6.0*r + 35.0*(r**2)/3.))*9./np.pi
	elif type=='C6':
		kernel = ((f**8.0)*(1.0 + 8.0*r + 25.0*(r**2) + 32*(r**3)))*78./(7.*np.pi)
	else:
		raise ValueError(type)
		

	if asTensor:
		return th.tensor(kernel, dtype=dtype, device=device)
	else:
		return kernel


"""
Create a 3d Wendland kernel matrix
"""
def wendland_kernel_3d(sigma, type="C4", asTensor=False, dtype=th.float32, device='cpu'):

	kernel_size = np.array(sigma)*2 + 1

	x_grid, y_grid, z_grid = _compute_mesh_grid_3d(kernel_size)

	r = np.sqrt((x_grid/sigma[0])**2 + (y_grid/sigma[1])**2 + (z_grid/sigma[2])**2)
	f = np.maximum(1 - r, 0)

	#kernel = ((f**6.0)*(3.0 + 18.0*r + 35.0*(r**2)))*(560.0/1680.0)
	if type=='C2':
		kernel = ((f**4.0)*(1.0 + 4.0*r))*21./(2.*np.pi)
	elif type=='C4':
		kernel = ((f**6.0)*(1.0 + 6.0*r + 35.0*(r**2)/3.))*495./(32.*np.pi)
	elif type=='C6':
		kernel = ((f**8.0)*(1.0 + 8.0*r + 25.0*(r**2) + 32*(r**3)))*1365./(64.*np.pi)
	else:
		raise ValueError(type)

	if asTensor:
		return th.tensor(kernel, dtype=dtype, device=device)
	else:
		return kernel


"""
    Create a Wendland kernel matrix
"""
def wendland_kernel(sigma, dim=1, type="C4", asTensor=False, dtype=th.float32, device='cpu'):

    assert dim > 0 and dim <=3

    if dim == 1:
        return wendland_kernel_1d(sigma, type=type, asTensor=asTensor, dtype=dtype, device=device)
    elif dim == 2:
        return wendland_kernel_2d(sigma, type=type, asTensor=asTensor, dtype=dtype, device=device)
    else:
        return wendland_kernel_3d(sigma, type=type, asTensor=asTensor, dtype=dtype, device=device)


"""
    Create a 1d bspline kernel matrix
"""
def bspline_kernel_1d(sigma, order=2, asTensor=False, dtype=th.float32, device='cpu'):

    kernel_ones = th.ones(1, 1, sigma)
    kernel = kernel_ones
	
    padding = sigma - 1

    for i in range(1, order + 1):
        kernel = F.conv1d(kernel, kernel_ones, padding=padding)/sigma
	


    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()


"""
    Create a 2d bspline kernel matrix
"""
def bspline_kernel_2d(sigma=[1, 1], order=2, asTensor=False, dtype=th.float32, device='cpu'):
    kernel_ones = th.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma) - 1

    for i in range(1, order + 1):
        kernel = F.conv2d(kernel, kernel_ones, padding=(padding).tolist())/(sigma[0]*sigma[1])
    


    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()


"""
    Create a 3d bspline kernel matrix
"""
def bspline_kernel_3d(sigma=[1, 1, 1], order=2, asTensor=False, dtype=th.float32, device='cpu'):
    kernel_ones = th.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma) - 1

    for i in range(1, order + 1):
        kernel = F.conv3d(kernel, kernel_ones, padding=(padding).tolist())/(sigma[0]*sigma[1]*sigma[2])
	


    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()


"""
    Create a bspline kernel matrix for a given dim
"""
def bspline_kernel(sigma, order=2, dim=1, asTensor=False, dtype=th.float32, device='cpu'):

    assert dim > 0 and dim <=3

    if dim == 1:
        return bspline_kernel_1d(sigma, order=order, asTensor=asTensor, dtype=dtype, device=device)
    elif dim == 2:
        return bspline_kernel_2d(sigma, order=order, asTensor=asTensor, dtype=dtype, device=device)
    else:
        return bspline_kernel_3d(sigma, order=order, asTensor=asTensor, dtype=dtype, device=device)




