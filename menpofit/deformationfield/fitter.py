from menpofit.aam.fitter import AAMFitter
from menpofit.lucaskanade.appearance import AlternatingInverseCompositional
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import (ModelDrivenTransform,
                                DifferentiableAlignmentSimilarity)


class LucasKanadeDeformationFieldAAMFitter(AAMFitter):

    # def __init__(self, linear_aam, algorithm=AIC_GN,
    #              n_shape=None, n_appearance=None, **kwargs):

        # super(LucasKanadeDeformationFieldAAMFitter, self).__init__(
        #     linear_aam, n_shape, n_appearance)
        #
        # for j, (am, sm) in enumerate(zip(self.aam.appearance_models,
        #                                  self.aam.shape_models)):
        #
        #     transform = LinearWarp(sm, self.aam.n_landmarks,
        #                            sigma2=am.noise_variance())
        #
        #     self._algorithms.append(
        #         algorithm(LinearAAMInterface, am, transform, **kwargs))
        # pass

    def __init__(self, aam, algorithm=AlternatingInverseCompositional,
                 md_transform=LinearWarp, n_shape=None,
                 n_appearance=None, **kwargs):
        super(LucasKanadeDeformationFieldAAMFitter, self).__init__(aam)
        self._set_up(algorithm=algorithm, md_transform=md_transform,
                     n_shape=n_shape, n_appearance=n_appearance, **kwargs)

    @property
    def algorithm(self):
        r"""
        Returns a string containing the name of fitting algorithm.

        :type: `str`
        """
        return 'DF-AAM-' + self._fitters[0].algorithm

    def _set_up(self, algorithm=AlternatingInverseCompositional,
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
                raise ValueError('n_shape can be an integer or a float or None '
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

            # TODO: Linear Warp Transform
            if md_transform is not ModelDrivenTransform:
                md_trans = md_transform(
                    sm, self.aam.transform, global_transform,
                    source=am.mean().landmarks['source'].lms)
            else:
                md_trans = md_transform(
                    sm, self.aam.transform,
                    source=am.mean().landmarks['source'].lms)
            self._fitters.append(algorithm(am, md_trans, **kwargs))


class LinearWarp(OrthoPDM):
    def __init__(self):
        pass