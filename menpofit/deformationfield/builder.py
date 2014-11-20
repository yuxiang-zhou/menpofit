from menpofit.aam import AAMBuilder
from menpofit.transform import DifferentiablePiecewiseAffine
from menpo.feature import igo
from menpo.shape import PointCloud

import numpy as np


class DeformationFieldBuilder(AAMBuilder):
    def __init__(self, features=igo, transform=DifferentiablePiecewiseAffine,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=True,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=3):
        super(DeformationFieldBuilder, self).__init__(
            features, transform, trilist, normalization_diagonal, n_levels,
            downscale, scaled_shape_models, max_shape_components,
            max_appearance_components,  boundary)

    def _build_aam(self, shape_models, appearance_models, n_training_images):
        r"""
        Returns a DeformationField object.

        Parameters
        ----------
        shape_models : :map:`PCAModel`
            The trained multilevel shape models.

        appearance_models : :map:`PCAModel`
            The trained multilevel appearance models.

        n_training_images : `int`
            The number of training images.

        Returns
        -------
        aam : :map:`DeformationField`
            The trained DeformationField object.
        """
        from .base import DeformationField
        return DeformationField(shape_models, appearance_models,
                                n_training_images, self.transform,
                                self.features, self.reference_shape,
                                self.downscale, self.scaled_shape_models,
                                self.n_landmarks)

    def _build_shape_model(self, shapes, max_components):
        sparse_shape_model = super(DeformationFieldBuilder, self).\
            _build_shape_model(shapes, max_components)

        # Currently keeping same amount of landmarks
        # TODO: Need to handle inconsistent shapes
        mean_sparse_shape = sparse_shape_model.mean()
        self.n_landmarks = mean_sparse_shape.n_points
        self.reference_frame = self._build_reference_frame(mean_sparse_shape)

        # compute non-linear transforms
        transforms = (
            [self.transform(self.reference_frame.landmarks['source'].lms, s)
             for s in shapes])

        # build dense shapes
        dense_shapes = []
        for (t, s) in zip(transforms, shapes):
            warped_points = t.apply(self.reference_frame.mask.true_indices)
            dense_shape = PointCloud(np.vstack((s.points, warped_points)))
            dense_shapes.append(dense_shape)

        # build dense shape model
        dense_shape_model = super(DeformationFieldBuilder, self).\
            _build_shape_model(dense_shapes, max_components)

        return dense_shape_model

    def _build_reference_frame(self, mean_shape, sparsed=True):
        r"""
        Generates the reference frame given a mean shape.

        Parameters
        ----------
        mean_shape : :map:`PointCloud`
            The mean shape to use.

        Returns
        -------
        reference_frame : :map:`MaskedImage`
            The reference frame.
        """

        reference_shape = mean_shape[:self.n_landmarks] \
            if sparsed else mean_shape

        return super(DeformationFieldBuilder, self)._build_reference_frame(
            reference_shape
        )