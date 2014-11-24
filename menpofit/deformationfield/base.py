from menpofit.aam import AAM
from menpo.shape import PointCloud


class DeformationField(AAM):

    def __init__(self, shape_models, appearance_models, n_training_images,
                 transform, features, reference_shape, downscale,
                 scaled_shape_models, n_landmarks):
        super(DeformationField, self).__init__(
            shape_models, appearance_models, n_training_images, transform,
            features, reference_shape, downscale, scaled_shape_models)
        self.n_landmarks = n_landmarks

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
        spares_shape = PointCloud(shape_instance.points[:self.n_landmarks])

        reference_frame = self._build_reference_frame(spares_shape)

        transform = self.transform(
            reference_frame.landmarks['source'].lms, landmarks)

        return appearance_instance.warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True)

    def _build_reference_frame(self, reference_shape, landmarks=None):
        from .builder import build_reference_frame as brf
        return brf(reference_shape, self.n_landmarks)
