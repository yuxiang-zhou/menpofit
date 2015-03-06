from __future__ import division
import numpy as np
from menpo.image import Image

from menpofit.transform import DifferentiableAlignmentSimilarity
from menpofit.modelinstance import PDM, OrthoPDM
from menpofit.fitter import MultilevelFitter
from menpofit.gradientdescent import RLMS


class CLMFitter(MultilevelFitter):
    r"""
    Abstract Interface for defining Constrained Local Models Fitters.

    Parameters
    -----------
    clm : :map:`CLM`
        The Constrained Local Model to be used.
    """
    def __init__(self, clm):
        self.clm = clm

    @property
    def reference_shape(self):
        r"""
        The reference shape of the CLM.

        :type: :map:`PointCloud`
        """
        return self.clm.reference_shape

    @property
    def features(self):
        r"""
        The feature extracted at each pyramidal level during CLM building.
        Stored in ascending pyramidal order.

        :type: `list`
        """
        return self.clm.features

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during CLM building.

        :type: `int`
        """
        return self.clm.n_levels

    @property
    def downscale(self):
        r"""
        The downscale used to generate the final scale factor applied at
        each pyramidal level during CLM building.
        The scale factor is computed as:

            ``(downscale ** k) for k in range(n_levels)``

        :type: `float`
        """
        return self.clm.downscale


class GradientDescentCLMFitter(CLMFitter):
    r"""
    Gradient Descent based :map:`Fitter` for Constrained Local Models.

    Parameters
    -----------
    clm : :map:`CLM`
        The Constrained Local Model to be used.

    algorithm : subclass :map:`GradientDescent`, optional
        The :map:`GradientDescent` class to be used.

    pdm_transform : :map:`GlobalPDM` or subclass, optional
        The point distribution class to be used.

        .. note::

            Only :map:`GlobalPDM` and its subclasses are supported.
            :map:`PDM` is not supported at the moment.

    n_shape : `int` ``> 1``, ``0. <=`` `float` ``<= 1.``, `list` of the
        previous or ``None``, optional
        The number of shape components or amount of shape variance to be
        used per pyramidal level.

        If `None`, all available shape components ``(n_active_components)``
        will be used.
        If `int` ``> 1``, the specified number of shape components will be
        used.
        If ``0. <=`` `float` ``<= 1.``, the number of shape components
        capturing the specified variance ratio will be computed and used.

        If `list` of length ``n_levels``, then the number of components is
        defined per level. The first element of the list corresponds to the
        lowest pyramidal level and so on.
        If not a `list` or a `list` of length 1, then the specified number of
        components will be used for all levels.
    """
    def __init__(self, clm, algorithm=RLMS,
                 pdm_transform=OrthoPDM, n_shape=None, **kwargs):
        super(GradientDescentCLMFitter, self).__init__(clm)
        self._set_up(algorithm=algorithm, pdm_transform=pdm_transform,
                     n_shape=n_shape, **kwargs)

    @property
    def algorithm(self):
        r"""
        Returns a string containing the name of fitting algorithm.

        :type: `string`
        """
        return 'GD-CLM-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=RLMS,
                pdm_transform=OrthoPDM,
                global_transform=DifferentiableAlignmentSimilarity,
                n_shape=None, **kwargs):
        r"""
        Sets up the Gradient Descent Fitter object.

        Parameters
        -----------
        algorithm : :map:`GradientDescent`, optional
            The Gradient Descent class to be used.

        pdm_transform : :map:`GlobalPDM` or subclass, optional
            The point distribution class to be used.

        n_shape : `int` ``> 1``, ``0. <=`` `float` ``<= 1.``, `list` of the
            previous or ``None``, optional
            The number of shape components or amount of shape variance to be
            used per fitting level.

            If `None`, all available shape components ``(n_active_components)``
            will be used.
            If `int` ``> 1``, the specified number of shape components will be
            used.
            If ``0. <=`` `float` ``<= 1.``, the number of components capturing the
            specified variance ratio will be computed and used.

            If `list` of length ``n_levels``, then the number of components is
            defined per level. The first element of the list corresponds to the
            lowest pyramidal level and so on.
            If not a `list` or a `list` of length 1, then the specified number of
            components will be used for all levels.
        """
        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.clm.n_levels > 1:
                for sm in self.clm.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.clm.n_levels:
                for sm, n in zip(self.clm.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float or None'
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.clm.n_levels))

        self._fitters = []
        for j, (sm, clf) in enumerate(zip(self.clm.shape_models,
                                          self.clm.classifiers)):

            if pdm_transform is not PDM:
                pdm_trans = pdm_transform(sm, global_transform)
            else:
                pdm_trans = pdm_transform(sm)
            self._fitters.append(algorithm(clf, self.clm.patch_shape,
                                           pdm_trans, **kwargs))

    def __str__(self):
        from menpofit.base import name_of_callable
        out = "{0} Fitter\n" \
              " - Gradient-Descent {1}\n" \
              " - Transform is {2}.\n" \
              " - {3} training images.\n".format(
            self.clm._str_title, self._fitters[0].algorithm,
            self._fitters[0].transform.__class__.__name__,
            self.clm.n_training_images)
        # small strings about number of channels, channels string and downscale
        down_str = []
        for j in range(self.n_levels):
            if j == self.n_levels - 1:
                down_str.append('(no downscale)')
            else:
                down_str.append('(downscale by {})'.format(
                    self.downscale**(self.n_levels - j - 1)))
        temp_img = Image(image_data=np.random.rand(50, 50))
        if self.pyramid_on_features:
            temp = self.features(temp_img)
            n_channels = [temp.n_channels] * self.n_levels
        else:
            n_channels = []
            for j in range(self.n_levels):
                temp = self.features[j](temp_img)
                n_channels.append(temp.n_channels)
        # string about features and channels
        if self.pyramid_on_features:
            feat_str = "- Feature is {} with ".format(
                name_of_callable(self.features))
            if n_channels[0] == 1:
                ch_str = ["channel"]
            else:
                ch_str = ["channels"]
        else:
            feat_str = []
            ch_str = []
            for j in range(self.n_levels):
                if isinstance(self.features[j], str):
                    feat_str.append("- Feature is {} with ".format(
                        self.features[j]))
                elif self.features[j] is None:
                    feat_str.append("- No features extracted. ")
                else:
                    feat_str.append("- Feature is {} with ".format(
                        name_of_callable(self.features[j])))
                if n_channels[j] == 1:
                    ch_str.append("channel")
                else:
                    ch_str.append("channels")
        if self.n_levels > 1:
            if self.clm.scaled_shape_models:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}.\n   - Each level has a scaled shape " \
                      "model (reference frame).\n   - Patch size is {}W x " \
                      "{}H.\n".format(out, self.n_levels, self.downscale,
                                      self.clm.patch_shape[1],
                                      self.clm.patch_shape[0])

            else:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}:\n   - Shape models (reference frames) " \
                      "are not scaled.\n   - Patch size is {}W x " \
                      "{}H.\n".format(out, self.n_levels, self.downscale,
                                      self.clm.patch_shape[1],
                                      self.clm.patch_shape[0])
            if self.pyramid_on_features:
                out = "{}   - Pyramid was applied on feature space.\n   " \
                      "{}{} {} per image.\n".format(out, feat_str,
                                                    n_channels[0], ch_str[0])
            else:
                out = "{}   - Features were extracted at each pyramid " \
                      "level.\n".format(out)
            for i in range(self.n_levels - 1, -1, -1):
                out = "{}   - Level {} {}: \n".format(out, self.n_levels - i,
                                                      down_str[i])
                if not self.pyramid_on_features:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                out = "{0}     - {1} motion components\n     - {2} {3} " \
                      "classifiers.\n".format(
                    out, self._fitters[i].transform.n_parameters,
                    len(self._fitters[i].classifiers),
                    name_of_callable(self._fitters[i].classifiers[0]))
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - {4} motion components\n   - {5} {6} " \
                  "classifiers.".format(
                out, feat_str[0], n_channels[0], ch_str[0],
                out, self._fitters[0].transform.n_parameters,
                len(self._fitters[0].classifiers),
                name_of_callable(self._fitters[0].classifiers[0]))
        return out
