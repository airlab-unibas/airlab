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
import SimpleITK as sitk

class Points:
    """
        Class read and write points
    """
    @staticmethod
    def read(filename):
        """
        Read points from file. Following formats are supported:

        - pts:

        - vtk:

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
                if not lines[0] == "# vtk DataFile Version 3.0\n" and\
                    not lines[1] == "vtk output\n" and \
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

        points (array): array of points
        displacement (Displacement): displacement field to transform points
        return (array): transformed points
        """

        pass
