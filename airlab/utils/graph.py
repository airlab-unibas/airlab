# Copyright 2018 University of Basel, Center for medical image Analysis and Navigation
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
import numpy as np
from .matrix import MatrixDiagonalElement, LaplaceMatrix


class Graph():
    def __init__(self, graph_size, dtype=th.float32, device='cpu'):

        self.dtype=dtype
        self.device=device
        self._dim = len(graph_size)
        self._graph_size = np.array(graph_size)
        # compute the number of nodes
        self._number_of_nodes = np.prod(self._graph_size)

        self.laplace_matrix = None

        if self._dim == 2:
            self._create_graph_2d()
        elif self._dim == 3:
            self._create_graph_3d()


    def _create_graph_2d(self):

        # compute the number of edges
        self._number_of_edges = self._number_of_nodes*2.0

        for length in self._graph_size:
            self._number_of_edges -= length

        self._edge_index_D1 = [[], [], []]
        self._edge_index_D2 = [[], [], []]


        for y in range(self._graph_size[1]):
            for x in range(self._graph_size[0]):

                if x + 1 < self._graph_size[0]:
                    self._edge_index_D1[0].append(x + 1)
                    self._edge_index_D1[1].append(y)
                    self._edge_index_D1[2].append(x + y*self._graph_size[0])


                if y + 1 < self._graph_size[1]:
                    self._edge_index_D2[0].append(x)
                    self._edge_index_D2[1].append(y + 1)
                    self._edge_index_D2[2].append(x + y*self._graph_size[0])


        element_1 = MatrixDiagonalElement(np.asarray(self._edge_index_D1), np.ones(len(self._edge_index_D1[0])),
                                                 th.tensor(1, dtype=th.int64), dtype=self.dtype, device=self.device)
        element_2 = MatrixDiagonalElement(np.asarray(self._edge_index_D2), np.ones(len(self._edge_index_D2[0])),
                                                 th.tensor(self._graph_size[0], dtype=th.int64), dtype=self.dtype, device=self.device)

        self.laplace_matrix = LaplaceMatrix(self._number_of_nodes, [element_1, element_2], dtype=self.dtype,
                                                device=self.device)


    def _create_graph_3d(self):

        # compute the number of edges
        self._number_of_edges = self._number_of_nodes * 2.0

        # for length in self._graph_size:
        #     self._number_of_edges -= length
        #
        # self._number_of_edges *= self._graph_size[2];
        # self._number_of_edges += self._graph_size[0]*self._graph_size[1]*(self._graph_size[2] - 1);
        #
        # self._edge_index_D1 = [[], [], [], []]
        # self._edge_index_D2 = [[], [], [], []]
        # self._edge_index_D3 = [[], [], [], []]
        #
        # for z in range(self._graph_size[2]):
        #     for y in range(self._graph_size[1]):
        #         for x in range(self._graph_size[0]):
        #
        #             if x + 1 < self._graph_size[0]:
        #                 self._edge_index_D1[0].append(x-1)
        #                 self._edge_index_D1[1].append(y)
        #                 self._edge_index_D1[2].append(z)
        #                 self._edge_index_D1[3].append(x + y * self._graph_size[0] + z*self._graph_size[0]* self._graph_size[1])
        #
        #             if y + 1 < self._graph_size[1]:
        #                 self._edge_index_D2[0].append(x)
        #                 self._edge_index_D2[1].append(y + 1)
        #                 self._edge_index_D1[2].append(z)
        #                 self._edge_index_D2[3].append(x + y * self._graph_size[0] + z*self._graph_size[0]* self._graph_size[1])
        #
        #             if z + 1 < self._graph_size[2]:
        #                 self._edge_index_D3[0].append(x)
        #                 self._edge_index_D3[1].append(y)
        #                 self._edge_index_D3[2].append(z + 1)
        #                 self._edge_index_D3[3].append(x + y * self._graph_size[0] + z * self._graph_size[0] * self._graph_size[1])
        #
        # element_1 = MatrixDiagonalElement(np.asarray(self._edge_index_D1), np.ones(len(self._edge_index_D1[0])),
        #                                       th.tensor(1, dtype=th.int64), dtype=self.dtype, device=self.device)
        #
        # element_2 = MatrixDiagonalElement(np.asarray(self._edge_index_D2), np.ones(len(self._edge_index_D2[0])),
        #                                       th.tensor(self._graph_size[0], dtype=th.int64), dtype=self.dtype,
        #                                       device=self.device)
        #
        # element_3 = MatrixDiagonalElement(np.asarray(self._edge_index_D3), np.ones(len(self._edge_index_D3[0])),
        #                                       th.tensor(self._graph_size[0]*self._graph_size[1], dtype=th.int64), dtype=self.dtype,
        #                                       device=self.device)
        #
        # self.laplace_matrix = LaplaceMatrix(self._number_of_nodes, [element_1, element_2, element_3])
