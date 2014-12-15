from menpofit.aam.fitter import LucasKanadeAAMFitter
from menpofit.lucaskanade.appearance import AlternatingInverseCompositional
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import DifferentiableAlignmentSimilarity
from menpofit.fittingresult import ParametricFittingResult, compute_error, \
    MultilevelFittingResult
from menpo.transform.base import Transform, VInvertible, VComposable
from menpo.transform import UniformScale, Translation
from menpo.shape import PointCloud

from .builder import ICP

import numpy as np
import scipy


class LinearWarp(OrthoPDM, Transform, VInvertible, VComposable):

    def __init__(self, model, n_landmarks=0):
        super(LinearWarp, self).__init__(model,
                                         DifferentiableAlignmentSimilarity)
        self.n_landmarks = n_landmarks
        self.W = np.vstack((self.similarity_model.components,
                            self.model.components))
        # v = self.W[:, :self.n_dims*self.n_landmarks]
        # self.pinv_v = scipy.linalg.pinv(v)

        # sm_mean_l = self.models[self.model_index-1].mean()
        # sm_mean_h = self.model.mean()
        # icp = ICP([sm_mean_l], sm_mean_h)
        # spare_index = spare_index_base = icp.point_correspondence[0]*2
        #
        # for i in range(self.n_dims-1):
        #     spare_index = np.vstack((spare_index, spare_index_base+i+1))
        #
        # spare_index = spare_index.T.reshape(
        #     spare_index_base.shape[0]*self.n_dims
        # )
        #
        # v = self.W[:, spare_index]
        # self.pinv_v = scipy.linalg.pinv(v)

    @property
    def dense_target(self):
        return PointCloud(self.target.points[self.n_landmarks:])

    @property
    def sparse_target(self):
        return PointCloud(self.target.points[:self.n_landmarks])

    def set_target(self, target):
        if target.n_points < self.target.n_points:
            tmin, tmax = target.bounds()
            smin, smax = self.model.mean().bounds()
            ss = PointCloud(np.array(
                [[smin[0], smin[1]],
                [smin[0], smax[1]],
                [smax[0], smin[1]],
                [smax[0], smax[1]]]
            ))

            tt = PointCloud(np.array(
                [[tmin[0], tmin[1]],
                [tmin[0], tmax[1]],
                [tmax[0], tmin[1]],
                [tmax[0], tmax[1]]])
            )
            t = DifferentiableAlignmentSimilarity(ss, tt)

            target = t.apply(self.model.mean())

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
    def n_landmarks(self):
        return self.fitter.transform.n_landmarks

    # @property
    # def final_shape(self):
    #     return PointCloud(self.final_transform.target.points[
    #                       :self.n_landmarks])
    #
    # @property
    # def initial_shape(self):
    #     return PointCloud(self.initial_transform.target.points[
    #                       :self.n_landmarks])


class DeformationFieldAICompositional(AlternatingInverseCompositional):

    def _create_fitting_result(self, image, parameters, gt_shape=None):
        return DFFittingResult(image, self, parameters=[parameters],
                                       gt_shape=gt_shape)


class DFMultilevelFittingResult(MultilevelFittingResult):
    def final_error(self, error_type='me_norm'):
        return 100

    def initial_error(self, error_type='me_norm'):
        return 100

    pass
    # def errors(self, error_type='me_norm'):
    #     r"""
    #     Returns a list containing the error at each fitting iteration.
    #
    #     Parameters
    #     -----------
    #     error_type : `str` ``{'me_norm', 'me', 'rmse'}``, optional
    #         Specifies the way in which the error between the fitted and
    #         ground truth shapes is to be computed.
    #
    #     Returns
    #     -------
    #     errors : `list` of `float`
    #         The errors at each iteration of the fitting process.
    #     """
    #     if self.gt_shape is not None:
    #         return [compute_error(
    #             PointCloud(t.points[:self.fitting_results[-1].n_landmarks]),
    #             self.gt_shape, error_type)
    #             for t in self.shapes]
    #     else:
    #         raise ValueError('Ground truth has not been set, errors cannot '
    #                          'be computed')


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
            transform = md_transform(sm)
            self._fitters.append(algorithm(am, transform, **kwargs))

    def _create_fitting_result(self, image, fitting_results, affine_correction,
                               gt_shape=None):
        r"""
        Creates the :class: `menpo.aam.fitting.MultipleFitting` object
        associated with a particular Fitter object.

        Parameters
        -----------
        image: :class:`menpo.image.masked.MaskedImage`
            The original image to be fitted.
        fitting_results: :class:`menpo.fit.fittingresult.FittingResultList`
            A list of basic fitting objects containing the state of the
            different fitting levels.
        affine_correction: :class: `menpo.transforms.affine.Affine`
            An affine transform that maps the result of the top resolution
            fitting level to the space scale of the original image.
        gt_shape: class:`menpo.shape.PointCloud`, optional
            The ground truth shape associated to the image.

            Default: None
        error_type: 'me_norm', 'me' or 'rmse', optional
            Specifies the way in which the error between the fitted and
            ground truth shapes is to be computed.

            Default: 'me_norm'

        Returns
        -------
        fitting: :class:`menpo.fitmultilevel.fittingresult.MultilevelFittingResult`
            The fitting object that will hold the state of the fitter.
        """
        return DFMultilevelFittingResult(image, self, fitting_results,
                            affine_correction, gt_shape=gt_shape)

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