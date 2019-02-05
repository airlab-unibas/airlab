# Change Log

All notable changes to this project will be documented in this file.

## [v0.2.2]
- Support for masking in displacement regularizers
- Bugfix in parametric regularizers with weight semantics

## [v0.2.1]
- Bugfix diffeomorphic transformation for multi-level registration
- Refectoring of the transformation class for the distinction for flow fields and displacement fields
- Added new example for multi-level diffeomorphic B-spline registration

## [v0.2.0]
- Added new diffeomorphic transformation option for all deformable transformations
- Added new image loss: Mutual Information
- Added new image loss: Normalized Gradient Fields
- Added new image loss: Structural Similarity Image Measure
- Reduce number of input parameter for the registration classes
- Added new non-deformable transformation: similarity and affine transformation including the option for adding the 
center of mass to the optimization parameters
- Bugfix verbose flag

## [v0.1.0]
- Extended Image class and adaption of loaded SimpleITK images to internal representation
- Functionality in image metrics to consider a fixed image mask and a moving image mask
- Functionality to derive a joint domain of the fixed and moving image. This includes center of mass alignment, creation of fixed and moving domain mask and resampling to a common pixel spacing
- Image loader class, where the 6 4DCT images of the POPI model can be loaded and cached (checkout: ImageLoader.show())
- New class for handling point sets (read, write, transformation with a displacement field and TRE calculation)
- Added a BedRemoval filter and an auto crop function
- Added a Bspline kernel registration example for 3d where all the new features are used

