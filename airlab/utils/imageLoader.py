# -*- coding: latin-1 -*-

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

import tempfile
import shutil
import os
import urllib.request

import torch as th

from .image import Image
from .points import Points

class ImageLoader(object):
    """
    Image loader is a class to download and cache images. Currently, the 6 4DCT datasets of the POPI model are supported.
    The downloaded images are cached in a temporary folder, such that, if the image is loaded twice, it will be taken
    from that folder. If landmark points are available too, they are downloaded and cached as well. The class is
    implemented as singleton in order to hold a database of cached images only once and consistent.

    Using the show() method, one can print the different images which can be loaded. They are grouped into subjects
    and images. For example: 4DCT_POPI_2 image_30 means, the subject 4DCT_POPI_2 and the third image of the respiratory
    cycle.

    Usage:
    loader = ImageLoader("/tmp/")
    (image, points) = loader.load("4DCT_POPI_2", "image_30")

    If no landmark points are defined, points is set to None.

    In the generate_database() other images can be registered to the database.

    Note: the provided publicly available images have been published for research purposes. If you are using them for
          your research, please cite the authors appropriately.

    """

    # singleton helper
    # (there is only one instance for class variables)
    __instance = None

    def __new__(cls, tmpdir=None):
        if ImageLoader.__instance is None:
            ImageLoader.__instance = object.__new__(cls)
            ImageLoader.__instance._database = {}
            ImageLoader.__instance._links = ImageLoader.generate_database()
            if tmpdir is None:
                ImageLoader.__instance._tmpdir = tempfile.mkdtemp()
            else:
                ImageLoader.__instance._tmpdir = tmpdir
        return ImageLoader.__instance

    def __str__(self):
        return "(ImageLoader) directory: " + ImageLoader.__instance._tmpdir + ", database: " + str(ImageLoader.__instance._database)


    class DataItem:
        def __init__(self, name, filename, copyright="N/A"):
            self.name = name
            self.filename = filename
            self.copyright = copyright
            self.data = None


    def show(self):
        """
        Prints all available images which can be loaded
        """
        for i in self._links:
            print(i)
            for j in self._links[i]:
                if not str(j)=="copyright":
                    print("\t"+str(j))

    def load(self, name, image, dtype=th.float32, device='cpu'):
        """
        Providing the subject name and the image of interest, the image is loaded either from the memory cache, the
        temporary folder or it is downloaded from the internet. Images which are registered in the _database can be loaded.

        Note: if no points are available to the image, None is returned as points

        name (str): subject name
        image (str): image name
        dtype: which pixel type the image should be converted to
        device: on which device the image should be allocated
        return (Image, array): tuple of the loaded image and the corresponding points
        """
        identifier = name +"_"+ image

        # check if image has already been loaded before to possibly return just the reference
        if not identifier in self._database:

            # check if the subjects is registered
            if not name in self._links:
                raise Exception("Image not found in link database: " + name)

            # check if the image is registered under the subject name
            if not image in self._links[name]:
                raise Exception("Image not found in image links: " + name + "/" + image)

            # generate temporary filenames
            image_filename = os.path.join(self._tmpdir, identifier+".mha")
            points_filename = os.path.join(self._tmpdir, identifier + ".pts")
            data = None
            copyright = self._links[name]["copyright"]

            # check if file is already available in the temp directory and possibly read it from there
            if os.path.isfile(image_filename):
                data = Image.read(image_filename, dtype, device)
            else:
                # get the links from the database
                link_mhd = self._links[name][image][0]["link_mhd"]
                link_raw = self._links[name][image][0]["link_raw"]

                # print copyright notice
                print("-------------------------------------------------------")
                print("Downloading: "+link_mhd)
                print("\nCopyright notice for " + identifier)
                print(copyright)

                try:
                    # download the image
                    urllib.request.urlretrieve(link_mhd, os.path.join(self._tmpdir, "download.mhd"))
                    urllib.request.urlretrieve(link_raw, os.path.join(self._tmpdir, "download.raw"))
                except:
                    print("Could not download the image")
                    raise

                # adjust meta file to match the temporary download name
                with open(os.path.join(self._tmpdir, "download.mhd"), 'r') as file:
                    lines = file.readlines()

                lines[-1] = lines[-1].split("=")[0] + "= download.raw"

                with open(os.path.join(self._tmpdir, "download.mhd"), 'w') as file:
                    file.write("".join(lines))

                # save image as mha file
                data = Image.read(os.path.join(self._tmpdir, "download.mhd"))
                data.write(image_filename)


            # load the corresponding points
            points = None # by default, points is None

            # check if the points have already been downloaded and available in the temp directory
            if os.path.isfile(points_filename):
                points = Points.read(points_filename)
            else:
                # get link from the database
                link_pts = self._links[name][image][0]["link_pts"]
                try:
                    # download the points
                    urllib.request.urlretrieve(link_pts, os.path.join(self._tmpdir, "download.pts"))
                    points = Points.read(os.path.join(self._tmpdir, "download.pts"))
                    Points.write(points_filename, points)
                except:
                    print("Warning: for subject "+name+" and image "+image+" no points are defined.")


            # make data item for the database and save the tuple consisting out of the image and the corresponding points
            item = ImageLoader.DataItem(identifier, image_filename, copyright)
            item.data = (data, points)

            self._database[identifier] = item

        return self._database[identifier].data


    @staticmethod
    def clear():
        """
        Delete database of images and the temp directory

        Finally, a new temp directory is created
        """
        # clear dict
        ImageLoader.__instance._database = {}

        # delete temp files
        shutil.rmtree(ImageLoader.__instance._tmpdir)

        # generate new directory
        ImageLoader.__instance._tmpdir = tempfile.mkdtemp()

    @staticmethod
    def get_temp_directory():
        """
        Returns the current temp directory
        """
        return ImageLoader.__instance._tmpdir


    @staticmethod
    def generate_database():
        """
        Generate database of links to the images which can be loaded. Currently, the 6 4DCT images of the POPI model
        are supported.
        return (dictionary): returns the database containing the names of the images and the respective links
        """

        # Adding DIR Validation Data to the database
        tags = ["bl", "ng", "dx", "gt", "mm2", "bh"]
        prefix = "4DCT_POPI_"

        data = {}
        for i in range(len(tags)):
            data[prefix+str(i)] = {}
            for j in range(10):
                data[prefix + str(i)]["image_" + str(j) + "0"] = []
                data[prefix + str(i)]["image_" + str(j) + "0"].append(
                {
                  "link_mhd": "https://www.creatis.insa-lyon.fr/~srit/POPI/MedPhys11/"+tags[i]+"/mhd/"+str(j)+"0.mhd",
                  "link_raw": "https://www.creatis.insa-lyon.fr/~srit/POPI/MedPhys11/"+tags[i]+"/mhd/"+str(j)+"0.raw",
                  "link_pts": "https://www.creatis.insa-lyon.fr/~srit/POPI/MedPhys11/"+tags[i]+"/pts/"+str(j)+"0.pts"
                }
                )
            data[prefix + str(i)]["copyright"] = """
    Data has been provided by the Léon Bérard Cancer Center & CREATIS lab, Lyon, France.
    The data is described in:
    
    J. Vandemeulebroucke, S. Rit, J. Kybic, P. Clarysse, and D. Sarrut. 
    "Spatiotemporal motion estimation for respiratory-correlated imaging of the lungs."
    In Med Phys, 2011, 38(1), 166-178.
    
    This data can be used for research only. If you use this data for your research, 
    please acknowledge the originators appropriately!
    """

        return data

