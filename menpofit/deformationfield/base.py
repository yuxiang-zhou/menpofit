from menpofit.aam import AAM
from menpo.shape import PointCloud
from menpofit.aam.builder import build_reference_frame

import numpy as np


class DeformationField(AAM):

    def __init__(self, shape_models, appearance_models, n_training_images,
                 transform, features, reference_shape, downscale,
                 scaled_shape_models):
        super(DeformationField, self).__init__(
            shape_models, appearance_models, n_training_images, transform,
            features, reference_shape, downscale, scaled_shape_models)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `string`
        """
        return 'AAM Based Deformation Field'

    def __str__(self):
        out = super(DeformationField, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        return '\n'.join(out_splitted)

    def _instance(self, level, shape_instance, appearance_instance):
        template = self.appearance_models[level].mean()
        landmarks = template.landmarks['source'].lms

        reference_frame = self._build_reference_frame(
            shape_instance)

        sparse_index = np.arange(0, landmarks.n_points, np.power(4, level))

        source = PointCloud(
            reference_frame.landmarks['source'].lms.points[sparse_index]
        )

        target = PointCloud(landmarks.points[sparse_index])

        transform = self.transform(source, target)

        return appearance_instance.warp_to_mask(reference_frame.mask,
                                                transform, warp_landmarks=True)

    def _build_reference_frame(self, reference_shape):

        return build_reference_frame(
            reference_shape, trilist=None, boundary=0)
