from __future__ import division

from menpofit.fitter import MultilevelFitter
from menpofit.fittingresult import AMMultilevelFittingResult
from menpofit.transform import (ModelDrivenTransform, OrthoMDTransform,
                                DifferentiableAlignmentSimilarity)
from menpofit.lucaskanade.residual import SSD, GaborFourier
from menpofit.lucaskanade.image import IC
from menpofit.base import name_of_callable


class ATMFitter(MultilevelFitter):
    r"""
    Abstract Interface for defining Active Template Models Fitters.

    Parameters
    -----------
    atm : :map:`ATM`
        The Active Template Model to be used.
    """
    def __init__(self, atm):
        self.atm = atm

    @property
    def reference_shape(self):
        r"""
        The reference shape of the ATM.

        :type: :map:`PointCloud`
        """
        return self.atm.reference_shape

    @property
    def features(self):
        r"""
        The feature extracted at each pyramidal level during ATM building.
        Stored in ascending pyramidal order.

        :type: `list`
        """
        return self.atm.features

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during ATM building.

        :type: `int`
        """
        return self.atm.n_levels

    @property
    def downscale(self):
        r"""
        The downscale used to generate the final scale factor applied at
        each pyramidal level during ATM building.
        The scale factor is computed as:

            ``(downscale ** k) for k in range(n_levels)``

        :type: `float`
        """
        return self.atm.downscale

    def _create_fitting_result(self, image, fitting_results, affine_correction,
                               gt_shape=None):
        r"""
        Creates a :map:`ATMMultilevelFittingResult` associated to a
        particular fitting of the ATM fitter.

        Parameters
        -----------
        image : :map:`Image` or subclass
            The image to be fitted.

        fitting_results : `list` of :map:`FittingResult`
            A list of fitting result objects containing the state of the
            the fitting for each pyramidal level.

        affine_correction : :map:`Affine`
            An affine transform that maps the result of the top resolution
            level to the scale space of the original image.

        gt_shape : :map:`PointCloud`, optional
            The ground truth shape associated to the image.

        error_type : 'me_norm', 'me' or 'rmse', optional
            Specifies how the error between the fitted and ground truth
            shapes must be computed.

        Returns
        -------
        fitting : :map:`ATMMultilevelFittingResult`
            A fitting result object that will hold the state of the ATM
            fitter for a particular fitting.
        """
        return ATMMultilevelFittingResult(
            image, self, fitting_results, affine_correction, gt_shape=gt_shape)


class LucasKanadeATMFitter(ATMFitter):
    r"""
    Lucas-Kanade based :map:`Fitter` for Active Template Models.

    Parameters
    -----------
    atm : :map:`ATM`
        The Active Template Model to be used.

    algorithm : subclass of :map:`ImageLucasKanade`, optional
        The Image Lucas-Kanade class to be used.

    md_transform : :map:`ModelDrivenTransform` or subclass, optional
        The model driven transform class to be used.

    n_shape : `int` ``> 1``, ``0. <=`` `float` ``<= 1.``, `list` of the
        previous or ``None``, optional
        The number of shape components or amount of shape variance to be
        used per pyramidal level.

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
    def __init__(self, atm, algorithm=IC, residual=SSD,
                 md_transform=OrthoMDTransform, n_shape=None, **kwargs):
        super(LucasKanadeATMFitter, self).__init__(atm)
        self._set_up(algorithm=algorithm, residual=residual,
                     md_transform=md_transform, n_shape=n_shape, **kwargs)

    @property
    def algorithm(self):
        r"""
        Returns a string containing the name of fitting algorithm.

        :type: `str`
        """
        return 'LK-ATM-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=IC,
                residual=SSD, md_transform=OrthoMDTransform,
                global_transform=DifferentiableAlignmentSimilarity,
                n_shape=None, **kwargs):
        r"""
        Sets up the Lucas-Kanade fitter object.

        Parameters
        -----------
        algorithm : subclass of :map:`ImageLucasKanade`, optional
            The Image Lucas-Kanade class to be used.

        md_transform : :map:`ModelDrivenTransform` or subclass, optional
            The model driven transform class to be used.

        n_shape : `int` ``> 1``, ``0. <=`` `float` ``<= 1.``, `list` of the
            previous or ``None``, optional
            The number of shape components or amount of shape variance to be
            used per pyramidal level.

            If `None`, all available shape components ``(n_active_components)``
            will be used.
            If `int` ``> 1``, the specified number of shape components will be
            used.
            If ``0. <=`` `float` ``<= 1.``, the number of components capturing
            the specified variance ratio will be computed and used.

            If `list` of length ``n_levels``, then the number of components is
            defined per level. The first element of the list corresponds to the
            lowest pyramidal level and so on.
            If not a `list` or a `list` of length 1, then the specified number
            of components will be used for all levels.

        Raises
        -------
        ValueError
            ``n_shape`` can be an `int`, `float`, ``None`` or a `list`
            containing ``1`` or ``n_levels`` of those.
        """
        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.atm.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.atm.n_levels > 1:
                for sm in self.atm.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.atm.n_levels:
                for sm, n in zip(self.atm.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an integer or a float or None '
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.atm.n_levels))

        self._fitters = []
        for j, (t, sm) in enumerate(zip(self.atm.warped_templates,
                                        self.atm.shape_models)):

            if md_transform is not ModelDrivenTransform:
                md_trans = md_transform(
                    sm, self.atm.transform, global_transform,
                    source=t.landmarks['source'].lms)
            else:
                md_trans = md_transform(
                    sm, self.atm.transform,
                    source=t.landmarks['source'].lms)

            if residual is not GaborFourier:
                self._fitters.append(
                    algorithm(t, residual(), md_trans, **kwargs))
            else:
                self._fitters.append(
                    algorithm(t, residual(t.shape), md_trans,
                              **kwargs))

    def __str__(self):
        out = "{0} Fitter\n" \
              " - Lucas-Kanade {1}\n" \
              " - Transform is {2} and residual is {3}.\n" \
              " - {4} training images.\n".format(
              self.atm._str_title, self._fitters[0].algorithm,
              self._fitters[0].transform.__class__.__name__,
              self._fitters[0].residual.type, self.atm.n_training_shapes)
        # small strings about number of channels, channels string and downscale
        n_channels = []
        down_str = []
        for j in range(self.n_levels):
            n_channels.append(
                self._fitters[j].template.n_channels)
            if j == self.n_levels - 1:
                down_str.append('(no downscale)')
            else:
                down_str.append('(downscale by {})'.format(
                    self.downscale**(self.n_levels - j - 1)))
        # string about features and channels
        if self.pyramid_on_features:
            feat_str = "- Feature is {} with ".format(name_of_callable(
                self.features))
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
                        self.features[j].__name__))
                if n_channels[j] == 1:
                    ch_str.append("channel")
                else:
                    ch_str.append("channels")
        if self.n_levels > 1:
            if self.atm.scaled_shape_models:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}.\n   - Each level has a scaled shape " \
                      "model (reference frame).\n".format(out, self.n_levels,
                                                          self.downscale)

            else:
                out = "{} - Gaussian pyramid with {} levels and downscale " \
                      "factor of {}:\n   - Shape models (reference frames) " \
                      "are not scaled.\n".format(out, self.n_levels,
                                                 self.downscale)
            if self.pyramid_on_features:
                out = "{}   - Pyramid was applied on feature space.\n   " \
                      "{}{} {} per image.\n".format(out, feat_str,
                                                    n_channels[0], ch_str[0])
                if not self.atm.scaled_shape_models:
                    out = "{}   - Reference frames of length {} " \
                          "({} x {}C, {} x {}C)\n".format(
                          out,
                          self._fitters[0].template.n_true_pixels() *
                                                                   n_channels[0],
                          self._fitters[0].template.n_true_pixels(),
                          n_channels[0], self._fitters[0].template._str_shape,
                          n_channels[0])
            else:
                out = "{}   - Features were extracted at each pyramid " \
                      "level.\n".format(out)
            for i in range(self.n_levels - 1, -1, -1):
                out = "{}   - Level {} {}: \n".format(out, self.n_levels - i,
                                                      down_str[i])
                if not self.pyramid_on_features:
                    out = "{}     {}{} {} per image.\n".format(
                        out, feat_str[i], n_channels[i], ch_str[i])
                if (self.atm.scaled_shape_models or
                        (not self.pyramid_on_features)):
                    out = "{}     - Reference frame of length {} " \
                          "({} x {}C, {} x {}C)\n".format(
                          out,
                          self._fitters[i].template.n_true_pixels() *
                                                                   n_channels[i],
                          self._fitters[i].template.n_true_pixels(),
                          n_channels[i], self._fitters[i].template._str_shape,
                          n_channels[i])
                out = "{0}     - {1} motion components\n\n".format(
                      out, self._fitters[i].transform.n_parameters)
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n" \
                  "   - Reference frame of length {4} ({5} x {6}C, " \
                  "{7} x {8}C)\n   - {9} motion parameters\n".format(
                  out, feat_str[0], n_channels[0], ch_str[0],
                  self._fitters[0].template.n_true_pixels() * n_channels[0],
                  self._fitters[0].template.n_true_pixels(),
                  n_channels[0], self._fitters[0].template._str_shape,
                  n_channels[0], self._fitters[0].transform.n_parameters)
        return out


class ATMMultilevelFittingResult(AMMultilevelFittingResult):
    r"""
    Class that holds the state of a :map:`ATMFitter` object before,
    during and after it has fitted a particular image.
    """
    @property
    def atm_reconstructions(self):
        r"""
        The list containing the atm reconstruction (i.e. the template warped on
        the shape instance reconstruction) obtained at each fitting iteration.

        Note that this reconstruction is only tested to work for the
        :map:`OrthoMDTransform`

        :type: list` of :map:`Image` or subclass
        """
        atm_reconstructions = []
        for level, f in enumerate(self.fitting_results):
            for shape_w in f.parameters:
                shape_w = shape_w[4:]
                sm_level = self.fitter.aam.shape_models[level]
                swt = shape_w / sm_level.eigenvalues[:len(shape_w)] ** 0.5
                atm_reconstructions.append(self.fitter.aam.instance(
                    shape_weights=swt, level=level))
        return atm_reconstructions
