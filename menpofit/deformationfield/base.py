from menpofit.aam import AAM


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
        out = super(AAM, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        return '\n'.join(out_splitted)


# TODO: overwrite _instance

# TODO: build reference frame
