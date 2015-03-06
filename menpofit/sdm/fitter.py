import numpy as np
from menpo.image import Image

from menpofit.base import name_of_callable
from menpofit.aam.fitter import AAMFitter
from menpofit.clm.fitter import CLMFitter
from menpofit.fitter import MultilevelFitter


class SDFitter(MultilevelFitter):
    r"""
    Abstract Supervised Descent Fitter.
    """
    def _set_up(self):
        r"""
        Sets up the SD fitter object.
        """

    def fit(self, image, initial_shape, max_iters=None, gt_shape=None,
            **kwargs):
        r"""
        Fits a single image.

        Parameters
        -----------
        image : :map:`MaskedImage`
            The image to be fitted.
        initial_shape : :map:`PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        max_iters :  int  or `list`, optional
            The maximum number of iterations.

            If `int`, then this will be the overall maximum number of iterations
            for all the pyramidal levels.

            If `list`, then a maximum number of iterations is specified for each
            pyramidal level.

        gt_shape : :map:`PointCloud`
            The ground truth shape of the image.

        **kwargs : `dict`
            optional arguments to be passed through.

        Returns
        -------
        fitting_list : :map:`FittingResultList`
            A fitting result object.
        """
        if max_iters is None:
            max_iters = self.n_levels
        return MultilevelFitter.fit(self, image, initial_shape,
                                    max_iters=max_iters, gt_shape=gt_shape,
                                    **kwargs)


class SDMFitter(SDFitter):
    r"""
    Supervised Descent Method.

    Parameters
    -----------
    regressors : :map:`RegressorTrainer`
        The trained regressors.

    n_training_images : `int`
        The number of images that were used to train the SDM fitter. It is
        only used for informational reasons.

    features : `callable` or ``[callable]``, optional
        If list of length ``n_levels``, feature extraction is performed at
        each level after downscaling of the image.
        The first element of the list specifies the features to be extracted at
        the lowest pyramidal level and so on.

        If ``callable`` the specified feature will be applied to the original
        image and pyramid generation will be performed on top of the feature
        image. Also see the `pyramid_on_features` property.

    reference_shape : :map:`PointCloud`
        The reference shape that was used to resize all training images to a
        consistent object size.

    downscale : `float`
        The downscale factor that will be used to create the different
        pyramidal levels. The scale factor will be::

            (downscale ** k) for k in range(n_levels)

    References
    ----------
    .. [XiongD13] Supervised Descent Method and its Applications to
       Face Alignment
       Xuehan Xiong and Fernando De la Torre Fernando
       IEEE International Conference on Computer Vision and Pattern Recognition
       May, 2013
    """
    def __init__(self, regressors, n_training_images, features,
                 reference_shape, downscale):
        self._fitters = regressors
        self._features = features
        self._reference_shape = reference_shape
        self._downscale = downscale
        self._n_training_images = n_training_images

    @property
    def algorithm(self):
        r"""
        Returns a string containing the algorithm used from the SDM family.

        : str
        """
        return 'SDM-' + self._fitters[0].algorithm

    @property
    def reference_shape(self):
        r"""
        The reference shape used during training.

        :type: :map:`PointCloud`
        """
        return self._reference_shape

    @property
    def features(self):
        r"""
        The feature type per pyramid level. Note that they are stored from
        lowest to highest level resolution.

        :type: `list`
        """
        return self._features

    @property
    def n_levels(self):
        r"""
        The number of pyramidal levels used during training.

        : int
        """
        return len(self._fitters)

    @property
    def downscale(self):
        r"""
        The downscale per pyramidal level used during building the AAM.
        The scale factor is: (downscale ** k) for k in range(n_levels)

        :type: `float`
        """
        return self._downscale

    def __str__(self):
        out = "Supervised Descent Method\n" \
              " - Non-Parametric '{}' Regressor\n" \
              " - {} training images.\n".format(
            name_of_callable(self._fitters[0].regressor),
            self._n_training_images)
        # small strings about number of channels, channels string and downscale
        down_str = []
        for j in range(self.n_levels):
            if j == self.n_levels - 1:
                down_str.append('(no downscale)')
            else:
                down_str.append('(downscale by {})'.format(
                    self.downscale**(self.n_levels - j - 1)))
        temp_img = Image(image_data=np.random.rand(40, 40))
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
                        self.features[j].__name__))
                if n_channels[j] == 1:
                    ch_str.append("channel")
                else:
                    ch_str.append("channels")
        if self.n_levels > 1:
            out = "{} - Gaussian pyramid with {} levels and downscale " \
                  "factor of {}.\n".format(out, self.n_levels,
                                           self.downscale)
            if self.pyramid_on_features:
                out = "{}   - Pyramid was applied on feature space.\n   " \
                      "{}{} {} per image.\n".format(out, feat_str,
                                                    n_channels[0], ch_str[0])
            else:
                out = "{}   - Features were extracted at each pyramid " \
                      "level.\n".format(out)
                for i in range(self.n_levels - 1, -1, -1):
                    out = "{}   - Level {} {}: \n     {}{} {} per " \
                          "image.\n".format(
                        out, self.n_levels - i, down_str[i], feat_str[i],
                        n_channels[i], ch_str[i])
        else:
            if self.pyramid_on_features:
                feat_str = [feat_str]
            out = "{0} - No pyramid used:\n   {1}{2} {3} per image.\n".format(
                out, feat_str[0], n_channels[0], ch_str[0])
        return out


class SDAAMFitter(AAMFitter, SDFitter):
    r"""
    Supervised Descent Fitter for AAMs.

    Parameters
    -----------
    aam : :map:`AAM`
        The Active Appearance Model to be used.

    regressors : :map:``RegressorTrainer`
        The trained regressors.

    n_training_images : `int`
        The number of training images used to train the SDM fitter.
    """
    def __init__(self, aam, regressors, n_training_images):
        super(SDAAMFitter, self).__init__(aam)
        self._fitters = regressors
        self._n_training_images = n_training_images

    @property
    def algorithm(self):
        r"""
        Returns a string containing the algorithm used from the SDM family.

        :type: `string`
        """
        return 'SD-AAM-' + self._fitters[0].algorithm

    def __str__(self):
        return "{}Supervised Descent Method for AAMs:\n" \
               " - Parametric '{}' Regressor\n" \
               " - {} training images.\n".format(
            self.aam.__str__(), name_of_callable(self._fitters[0].regressor),
            self._n_training_images)


class SDCLMFitter(CLMFitter, SDFitter):
    r"""
    Supervised Descent Fitter for CLMs.

    Parameters
    -----------
    clm : :map:`CLM`
        The Constrained Local Model to be used.

    regressors : :map:`RegressorTrainer`
        The trained regressors.

    n_training_images : `int`
        The number of training images used to train the SDM fitter.

    References
    ----------
    .. [Asthana13] Robust Discriminative Response Map Fitting with Constrained
       Local Models
       A. Asthana, S. Zafeiriou, S. Cheng, M. Pantic.
       IEEE Conference onComputer Vision and Pattern Recognition.
       Portland, Oregon, USA, June 2013.
    """
    def __init__(self, clm, regressors, n_training_images):
        super(SDCLMFitter, self).__init__(clm)
        self._fitters = regressors
        self._n_training_images = n_training_images

    @property
    def algorithm(self):
        r"""
        Returns a string containing the algorithm used from the SDM family.

        :type: `string`
        """
        return 'SD-CLM-' + self._fitters[0].algorithm

    def __str__(self):
        return "{}Supervised Descent Method for CLMs:\n" \
               " - Parametric '{}' Regressor\n" \
               " - {} training images.\n".format(
            self.clm.__str__(), name_of_callable(self._fitters[0].regressor),
            self._n_training_images)
