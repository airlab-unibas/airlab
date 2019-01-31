from setuptools import setup, find_packages

__version__ = '0.2.1'
url = 'https://github.com/airlab-unibas/airlab'

install_requires = ['SimpleITK', 'torch', 'numpy', 'matplotlib']


setup(
    name='airlab',
    description='Autograd Image Registraion Laboratory',
    version=__version__,
    author='Robin Sandkuehler, Christoph Jud',
    author_email='robin.sandkuehler@unibas.ch',
    url=url,
    keywords=['image registration'],
    install_requires=install_requires,
    packages=find_packages(exclude=['example']),
    ext_package='')
