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

import numpy as np
import torch as th
import SimpleITK as sitk

from .image import Displacement

class Points:
    """
        Class implementing functionality for dealing with points:

        - read/write: supported formats are pts and vtk (polydata)
        - transform: transform the points given a displacement field
        - TRE: calculates the target registration error between two point sets
    """
    @staticmethod
    def read(filename):
        """
        Read points from file. Following formats are supported:

        - pts: each point is represended in one line where the coordinates are separated with a tab

        - vtk: the vtk polydata is supported as well

        filename (str): filename
        return (array): two dimensional array
        """
        if filename.endswith("pts"):
            points = []
            with open(filename) as f:
                lines = f.readlines()
                for l in lines:
                    points.append([float(p) for p in l.split()])
            return np.array(points)

        elif filename.endswith("vtk"):
            with open(filename) as f:
                lines = f.readlines()
                if not lines[1] == "vtk output\n" and \
                    not lines[2] == "ASCII\n" and \
                    not lines[3] == "DATASET POLYDATA\n":
                    raise Exception("Tried to read corrupted vtk polydata file")
                n = int(lines[4].split()[1])

                one_line = ''.join(''.join(lines[5:]).split('\n'))
                one_line = [float(p) for p in one_line.split()]
                return np.array(one_line).reshape((n, 3))

        else:
            raise Exception("Format not supported: "+str(filename))

    @staticmethod
    def write(filename, points):
        """
        Write point list to hard drive
        filename (str): destination filename
        points (array): two dimensional array
        """
        if filename.endswith("pts"):
            with open(filename, 'w') as f:
                for p in points:
                    f.write('\t'.join([str(v) for v in p])+'\n')

        elif filename.endswith("vtk"):
            n = points.shape[0]
            with open(filename, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("vtk output\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")
                f.write("POINTS "+str(n)+" float\n")
                for p in points:
                    f.write('\t'.join([str(v) for v in p])+'\n')

        else:
            raise Exception("Format not supported: "+str(filename))

    @staticmethod
    def transform(points, displacement):
        """
        Transforms a set of points with a displacement field

        points (array): array of points
        displacement (SimpleITK.Image | Displacement ): displacement field to transform points
        return (array): transformed points
        """
        if type(displacement) == sitk.SimpleITK.Image:
            df_transform = sitk.DisplacementFieldTransform(displacement)
        elif type(displacement) == Displacement:
            df_transform = sitk.DisplacementFieldTransform(displacement.to(dtype=th.float64).itk())
        else:
            raise Exception("Datatype of displacement field not supported.")

        df_transform.SetSmoothingOff()

        transformed_points = np.zeros_like(points)
        for i in range(points.shape[0]):
            transformed_points[i, :] = df_transform.TransformPoint(points[i, :])

        return transformed_points

    @staticmethod
    def TRE(points1, points2):
        """
        Computes the average distance between points in points1 and points2

        Note: if there is a different amount of points in the two sets, only the first points are compared

        points1 (array): point set 1
        points2 (array): point set 2
        return (float): mean difference
        """
        n = min(points1.shape[0], points2.shape[0])
        return np.mean(np.linalg.norm(points1[:n,:]-points2[:n,:], axis=1))
