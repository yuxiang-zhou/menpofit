from menpofit.aam.fitter import LucasKanadeAAMFitter
from menpofit.lucaskanade.appearance import AlternatingInverseCompositional
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import DifferentiableAlignmentSimilarity
from menpofit.fittingresult import ParametricFittingResult
from menpo.transform.base import Transform, VInvertible, VComposable
from menpo.transform import Scale
from menpo.shape import PointCloud

import numpy as np
import scipy


class LinearWarp(OrthoPDM, Transform, VInvertible, VComposable):

    def __init__(self, model, n_landmarks):
        super(LinearWarp, self).__init__(model,
                                         DifferentiableAlignmentSimilarity)
        self.n_landmarks = n_landmarks
        self.W = np.vstack((self.similarity_model.components,
                            self.model.components))
        v = self.W[:, :self.n_dims*self.n_landmarks]
        self.pinv_v = scipy.linalg.pinv(v)

    @property
    def dense_target(self):
        return PointCloud(self.target.points[self.n_landmarks:])

    @property
    def sparse_target(self):
        return PointCloud(self.target.points[:self.n_landmarks])

    def set_target(self, target):
        if target.n_points == self.n_landmarks:
            # densify target
            target = np.dot(np.dot(target.as_vector(), self.pinv_v), self.W)
            target = PointCloud(np.reshape(target, (-1, self.n_dims)))
        OrthoPDM.set_target(self, target)

    def _apply(self, _, **kwargs):
        return self.target.points[self.n_landmarks:]

    def d_dp(self, _):
        return OrthoPDM.d_dp(self, _)[self.n_landmarks:, ...]

    def has_true_inverse(self):
        return False

    def pseudoinverse_vector(self, vector):
        return -vector

    def compose_after_from_vector_inplace(self, delta):
        self.from_vector_inplace(self.as_vector() + delta)


class DFFittingResult(ParametricFittingResult):
    @property
    def final_shape(self):
        return self.final_transform.target

    @property
    def initial_shape(self):
        return self.initial_transform.target


class DeformationFieldAICompositional(AlternatingInverseCompositional):

    def _create_fitting_result(self, image, parameters, gt_shape=None):
        return ParametricFittingResult(image, self, parameters=[parameters],
                                       gt_shape=gt_shape)


class LucasKanadeDeformationFieldAAMFitter(LucasKanadeAAMFitter):

    def __init__(self, aam, algorithm=DeformationFieldAICompositional,
                 md_transform=LinearWarp, n_shape=None,
                 n_appearance=None, **kwargs):
        super(LucasKanadeDeformationFieldAAMFitter, self).__init__(
            aam, algorithm, md_transform, n_shape, n_appearance, **kwargs)

    @property
    def algorithm(self):
        r"""
        Returns a string containing the name of fitting algorithm.

        :type: `str`
        """
        return 'DF-AAM-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=DeformationFieldAICompositional,
                md_transform=LinearWarp,
                global_transform=DifferentiableAlignmentSimilarity,
                n_shape=None, n_appearance=None, **kwargs):

        # check n_shape parameter
        if n_shape is not None:
            if type(n_shape) is int or type(n_shape) is float:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape
            elif len(n_shape) == 1 and self.aam.n_levels > 1:
                for sm in self.aam.shape_models:
                    sm.n_active_components = n_shape[0]
            elif len(n_shape) == self.aam.n_levels:
                for sm, n in zip(self.aam.shape_models, n_shape):
                    sm.n_active_components = n
            else:
                raise ValueError('n_shape can be an int or a float or None '
                                 'or a list containing 1 or {} of '
                                 'those'.format(self.aam.n_levels))

        # check n_appearance parameter
        if n_appearance is not None:
            if type(n_appearance) is int or type(n_appearance) is float:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance
            elif len(n_appearance) == 1 and self.aam.n_levels > 1:
                for am in self.aam.appearance_models:
                    am.n_active_components = n_appearance[0]
            elif len(n_appearance) == self.aam.n_levels:
                for am, n in zip(self.aam.appearance_models, n_appearance):
                    am.n_active_components = n
            else:
                raise ValueError('n_appearance can be an integer or a float '
                                 'or None or a list containing 1 or {} of '
                                 'those'.format(self.aam.n_levels))

        self._fitters = []
        for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
                                         self.aam.shape_models)):
            transform = md_transform(sm, self.aam.n_landmarks)
            self._fitters.append(algorithm(am, transform, **kwargs))

    def _fit(self, images, initial_shape, gt_shapes=None, max_iters=50,
             **kwargs):
        r"""
        Fits the fitter to the multilevel pyramidal images.

        Parameters
        -----------
        images: :class:`menpo.image.masked.MaskedImage` list
            The images to be fitted.
        initial_shape: :class:`menpo.shape.PointCloud`
            The initial shape from which the fitting will start.
        gt_shapes: :class:`menpo.shape.PointCloud` list, optional
            The original ground truth shapes associated to the multilevel
            images.

            Default: None
        max_iters: int or list, optional
            The maximum number of iterations.
            If int, then this will be the overall maximum number of iterations
            for all the pyramidal levels.
            If list, then a maximum number of iterations is specified for each
            pyramidal level.

            Default: 50

        Returns
        -------
        fitting_results: :class:`menpo.fit.fittingresult.FittingResult` list
            The fitting object containing the state of the whole fitting
            procedure.
        """
        shape = initial_shape
        gt_shape = None
        n_levels = self.n_levels

        # check max_iters parameter
        if type(max_iters) is int:
            max_iters = [np.round(max_iters/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) == 1 and n_levels > 1:
            max_iters = [np.round(max_iters[0]/n_levels)
                         for _ in range(n_levels)]
        elif len(max_iters) != n_levels:
            raise ValueError('max_iters can be integer, integer list '
                             'containing 1 or {} elements or '
                             'None'.format(self.n_levels))

        # fit images
        fitting_results = []
        for j, (i, f, it) in enumerate(zip(images, self._fitters, max_iters)):
            if gt_shapes is not None:
                gt_shape = gt_shapes[j]

            parameters = f.get_parameters(shape)
            fitting_result = f.fit(i, parameters, gt_shape=gt_shape,
                                   max_iters=it, **kwargs)
            fitting_results.append(fitting_result)

            shape = PointCloud(
                fitting_result.final_shape.points[:initial_shape.n_points])
            Scale(self.downscale, n_dims=shape.n_dims).apply_inplace(shape)

        return fitting_results

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `string`
        """
        return 'AAM Based Deformation Field Fitter'

    def __str__(self):
        out = super(LucasKanadeDeformationFieldAAMFitter, self).__str__()
        out_splitted = out.splitlines()
        out_splitted[0] = self._str_title
        return '\n'.join(out_splitted)