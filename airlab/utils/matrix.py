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

class MatrixDiagonalElement():
    def __init__(self, edge_index, edge_values, offset, dtype=th.float32, device='cpu'):
        self.edge_index = th.from_numpy(edge_index).to(dtype=th.int64, device=device)
        self.edge_values = th.from_numpy(edge_values).to(dtype=dtype, device=device)
        self.offset = offset.to(dtype=th.int64, device=device)

class LaplaceMatrix():
    def __init__(self, number_of_nodes, diag_elements, dtype=th.float32, device='cpu'):
        self.main_diag = th.zeros(int(number_of_nodes), dtype=dtype, device=device)
        self.diag_elements = diag_elements
        self.size = int(number_of_nodes)


        self.update()

    def update(self):
        self.main_diag.data.fill_(0)

        for diag in self.diag_elements:
            self.main_diag[diag.edge_index[-1, :]] -= diag.edge_values
            self.main_diag[diag.edge_index[-1, :] + diag.offset] -= diag.edge_values

    def full(self):
        mat = th.zeros(self.size, self.size, dtype=self.main_diag.dtype, device=self.main_diag.device)
        mat = mat + th.diag(self.main_diag)

        for diag in self.diag_elements:
            mat[diag.edge_index[-1, :], diag.edge_index[-1, :] + diag.offset] = diag.edge_values
            mat[diag.edge_index[-1, :] + diag.offset, diag.edge_index[-1, :]] = diag.edge_values

        return mat



def band_mv(A, x):

    y = th.zeros(x.size()[0], dtype=x.dtype, device=x.device)

    # add the main diagonal to the vector
    th.mul(A.main_diag, x, out=y)

    for diag in A.diag_elements:
        y[diag.edge_index[-1, :]] += th.mul(x[diag.edge_index[-1, :] + diag.offset], diag.edge_values)
        y[diag.edge_index[-1, :] + diag.offset] += th.mul(x[diag.edge_index[-1, :]], diag.edge_values)

    return y


def expm_eig(A):
    eigen_values, eigen_vector = th.eig(A, eigenvectors=True)

    eigen_values.exp_()

    return th.mm(th.mm(eigen_vector, th.diag(eigen_values[:, 0])), eigen_vector.t_())


def expm_krylov(A, x, phi=1, krylov_dim=30, inplace=True):

    if krylov_dim > x.size()[0]:
        krylov_dim = x.size()[0]

    Q = th.zeros(x.size()[0], krylov_dim, dtype=x.dtype, device=x.device)
    T = th.zeros(krylov_dim + 1, krylov_dim + 1, dtype=x.dtype, device=x.device)

    #compute the norm of the vector
    norm_x = th.norm(x, p=2)

    # normalize vector
    q = x/norm_x
    Q[:, 0] = q.clone()

    r = band_mv(A, q)

    T[0, 0] = th.dot(q, r)
    r = r - q.mul(T[0, 0])


    T[0, 1] = th.norm(r, p=2) + 1e-10
    T[1, 0] = T[0, 1]

    for k in range(1, krylov_dim):
        b = q
        q = r

        q.div_(T[k -1, k])

        Q[:,k] = q.clone()

        r = band_mv(A, q) - b.mul_(T[k -1, k])

        T[k, k] = th.dot(q, r)

        r =  r - q.mul(T[k, k])

        T[k + 1, k] = th.norm(r, p=2) + 1e-10
        T[k, k + 1] = T[k + 1, k]

    T.mul_(phi)
    exp_mat = expm_eig(T[:-1, :-1])

    if inplace:
        th.mv(Q, exp_mat[:,0], out=x)
        x.mul_(norm_x)
    else:
        return  th.mv(Q, exp_mat[:,0]).mul_(x)







