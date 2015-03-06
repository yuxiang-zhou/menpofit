from scipy.linalg import norm
import numpy as np

from .base import LucasKanade


class ImageLucasKanade(LucasKanade):

    def __init__(self, template, residual, transform, eps=10 ** -6):
        super(ImageLucasKanade, self).__init__(residual, transform, eps=eps)
        # in image alignment, we align a template image to the target image
        self.template = template
        # pre-compute
        self._set_up()


class FA(ImageLucasKanade):
    r"""
    Forward Additive algorithm
    """
    @property
    def algorithm(self):
        return 'Image-FA'

    def _fit(self, fitting_result, max_iters=20):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        n_iters = 0

        # Forward Additive Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # Compute the Jacobian of the warp
            dW_dp = np.rollaxis(
                self.transform.d_dp(self.template.indices()), -1)

            # TODO: rename kwarg "forward" to "forward_additive"
            # Compute steepest descent images, VI_dW_dp
            self._J = self.residual.steepest_descent_images(
                image, dW_dp, forward=(self.template, self.transform))

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp weights
            parameters = self.transform.as_vector() + delta_p
            self.transform.from_vector_inplace(parameters)
            fitting_result.parameters.append(parameters)

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        fitting_result.fitted = True
        return fitting_result


class FC(ImageLucasKanade):
    r"""
    Forward Compositional algorithm
    """
    @property
    def algorithm(self):
        return 'Image-FC'

    def _set_up(self):
        r"""
        The forward compositional algorithm pre-computes the Jacobian of the
        warp. This is set as an attribute on the class.
        """
        # Compute the Jacobian of the warp
        self._dW_dp = np.rollaxis(
            self.transform.d_dp(self.template.indices()), -1)

    def _fit(self, fitting_result, max_iters=20):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        n_iters = 0

        # Forward Compositional Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # TODO: add "forward_compositional" kwarg with options
            # In the forward compositional algorithm there are two different
            # ways of computing the steepest descent images:
            #   1. V[I(x)](W(x,p)) * dW/dx * dW/dp
            #   2. V[I(W(x,p))] * dW/dp -> this is what is currently used
            # Compute steepest descent images, VI_dW_dp
            self._J = self.residual.steepest_descent_images(IWxp, self._dW_dp)

            # Compute Hessian and inverse
            self._H = self.residual.calculate_hessian(self._J)

            # Compute steepest descent parameter updates
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, self.template, IWxp)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(delta_p)
            fitting_result.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        fitting_result.fitted = True
        return fitting_result


class IC(ImageLucasKanade):
    r"""
    Inverse Compositional algorithm
    """
    @property
    def algorithm(self):
        return 'Image-IC'

    def _set_up(self):
        r"""
        The Inverse Compositional algorithm pre-computes the Jacobian of the
        warp, the steepest descent images and the Hessian. These are all
        stored as attributes on the class.
        """
        # Compute the Jacobian of the warp
        dW_dp = np.rollaxis(self.transform.d_dp(self.template.indices()), -1)

        # Compute steepest descent images, VT_dW_dp
        self._J = self.residual.steepest_descent_images(
            self.template, dW_dp)

        # TODO: Pre-compute the inverse
        # Compute Hessian and inverse
        self._H = self.residual.calculate_hessian(self._J)

    def _fit(self, fitting_result, max_iters=20):
        # Initial error > eps
        error = self.eps + 1
        image = fitting_result.image
        n_iters = 0

        # Baker-Matthews, Inverse Compositional Algorithm
        while n_iters < max_iters and error > self.eps:
            # Compute warped image with current weights
            IWxp = image.warp_to_mask(self.template.mask, self.transform,
                                      warp_landmarks=False)

            # Compute steepest descent parameter updates.
            sd_delta_p = self.residual.steepest_descent_update(
                self._J, IWxp, self.template)

            # Compute gradient descent parameter updates
            delta_p = np.real(self._calculate_delta_p(sd_delta_p))

            # Request the pesudoinverse vector from the transform
            inv_delta_p = self.transform.pseudoinverse_vector(delta_p)

            # Update warp weights
            self.transform.compose_after_from_vector_inplace(inv_delta_p)
            fitting_result.parameters.append(self.transform.as_vector())

            # Test convergence
            error = np.abs(norm(delta_p))
            n_iters += 1

        fitting_result.fitted = True
        return fitting_result
