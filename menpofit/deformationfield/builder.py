from menpofit.aam import AAMBuilder
from menpofit.transform import DifferentiablePiecewiseAffine
from menpo.feature import igo
from menpo.shape import PointCloud
from menpo.transform.groupalign.base import MultipleAlignment

import numpy as np


class ICP(MultipleAlignment):
    def __init__(self, sources, target):
        self.target = target
        self._test_iteration = []
        self.aligned_shapes = [self._align_source(s) for s in sources]

        super(ICP, self).__init__(self.aligned_shapes, target)

    def _align_source(self, source, eps=1e-3, max_iter=100):
        pf = p0 = source.points
        n_p = p0.shape[0]
        tolerance = eps + 1
        tolerance_old = np.sum(np.power(pf - self.target.points, 2)) / n_p
        iter = 0
        iters = [pf]
        while tolerance > eps and iter < max_iter:
            pk = pf

            # Compute Closest Points
            yk = self._cloest_points(pk)

            # Compute Registration
            qr, qt = self._compute_registration(pk, yk)

            # Update source
            pf = self._update_source(pk, np.hstack((qr, qt)))

            # Calculate Mean Square Matching Error
            tolerance_new = np.sum(np.power(pf - self.target.points, 2)) / n_p
            tolerance = abs(tolerance_old - tolerance_new)
            tolerance_old = tolerance_new

            iter += 1
            iters.append(pf)

        self._test_iteration.append(iters)

        return PointCloud(pf)

    def _update_source(self, p, q):
        return _apply_q(p, q)[:, :p.shape[1]]

    def _compute_registration(self, p, x):
        # The algorithm handles points in 3D
        if p.shape[1] == 2:
            p = np.hstack((p, np.zeros((p.shape[0], 1))))
            x = np.hstack((x, np.zeros((x.shape[0], 1))))

        # Calculate Covariance
        up = np.mean(p, axis=0)
        ux = np.mean(x, axis=0)
        u = up[:, None].dot(ux[None, :])
        n_p = p.shape[0]
        cov = sum([pi[:, None].dot(xi[None, :])
                   for (pi, xi) in zip(p, x)]) / n_p - u

        # Calculate Symmetric Matrix
        anti_sym = cov - cov.T
        dt = np.array([anti_sym[1, 2],
                       anti_sym[2, 0],
                       anti_sym[0, 1]])
        cov_tr = np.trace(cov)
        temp_q = cov + cov.T - cov_tr * np.identity(3)
        temp_q = np.vstack((dt[None, :], temp_q))
        q = np.hstack((np.hstack((cov_tr, dt))[:, None], temp_q))

        # Calculate Rotation Matrix
        eig_values, eig_vectors = np.linalg.eig(q)
        qr = eig_vectors[np.argmax(eig_values)]
        r = _compose_r(qr)

        # Calculate Translation Vector
        qt = ux[:, None] - r.dot(up[:, None])

        return qr, qt.reshape(3)

    def _cloest_points(self, source):
        return np.array([self._closest_node(s) for s in source])

    def _closest_node(self, node):
        nodes = np.asarray(self.target.points)
        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        return self.target.points[np.argmin(dist_2)]


def _compose_r(qr):
    q0, q1, q2, q3 = qr
    r = np.zeros((3, 3))
    r[0, 0] = np.sum(np.power(qr * [1, 1, -1, -1], 2))
    r[1, 1] = np.sum(np.power(qr[[0, 2, 1, 3]] * [1, 1, -1, -1], 2))
    r[2, 2] = np.sum(np.power(qr[[0, 3, 1, 2]] * [1, 1, -1, -1], 2))
    r[0, 1] = 2 * (q1 * q2 - q0 * q3)
    r[1, 0] = 2 * (q1 * q2 + q0 * q3)
    r[0, 2] = 2 * (q1 * q3 + q0 * q2)
    r[2, 0] = 2 * (q1 * q3 - q0 * q2)
    r[1, 2] = 2 * (q2 * q3 - q0 * q1)
    r[2, 1] = 2 * (q2 * q3 + q0 * q1)

    return r


def _apply_q(source, q):
    if source.shape[1] == 2:
        source = np.hstack((source, np.zeros((source.shape[0], 1))))

    r = _compose_r(q[:4])
    t = q[4:]
    s1 = [r.dot(s) + t for s in source]
    return np.array(s1)


class DeformationFieldBuilder(AAMBuilder):
    def __init__(self, features=igo, transform=DifferentiablePiecewiseAffine,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=True,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=3):
        super(DeformationFieldBuilder, self).__init__(
            features, transform, trilist, normalization_diagonal, n_levels,
            downscale, scaled_shape_models, max_shape_components,
            max_appearance_components, boundary)

    def _build_aam(self, shape_models, appearance_models, n_training_images):
        r"""
        Returns a DeformationField object.

        Parameters
        ----------
        shape_models : :map:`PCAModel`
            The trained multilevel shape models.

        appearance_models : :map:`PCAModel`
            The trained multilevel appearance models.

        n_training_images : `int`
            The number of training images.

        Returns
        -------
        aam : :map:`DeformationField`
            The trained DeformationField object.
        """
        from .base import DeformationField

        return DeformationField(shape_models, appearance_models,
                                n_training_images, self.transform,
                                self.features, self.reference_shape,
                                self.downscale, self.scaled_shape_models,
                                self.n_landmarks)

    def _build_shape_model(self, shapes, max_components):
        # Currently keeping same amount of landmarks
        # TODO: Need to handle inconsistent shapes

        sparse_shape_model = super(DeformationFieldBuilder, self). \
            _build_shape_model(shapes, max_components)

        mean_sparse_shape = sparse_shape_model.mean()
        self.n_landmarks = mean_sparse_shape.n_points
        self.reference_frame = AAMBuilder._build_reference_frame(
            self, mean_sparse_shape)

        # compute non-linear transforms
        transforms = (
            [self.transform(self.reference_frame.landmarks['source'].lms, s)
             for s in shapes])

        # build dense shapes
        dense_shapes = []
        for (t, s) in zip(transforms, shapes):
            warped_points = t.apply(self.reference_frame.mask.true_indices())
            dense_shape = PointCloud(np.vstack((s.points, warped_points)))
            dense_shapes.append(dense_shape)

        # build dense shape model
        dense_shape_model = super(DeformationFieldBuilder, self). \
            _build_shape_model(dense_shapes, max_components)

        return dense_shape_model

    def _build_reference_frame(self, mean_shape, sparsed=True):
        r"""
        Generates the reference frame given a mean shape.

        Parameters
        ----------
        mean_shape : :map:`PointCloud`
            The mean shape to use.

        Returns
        -------
        reference_frame : :map:`MaskedImage`
            The reference frame.
        """

        return self.reference_frame


def build_reference_frame(mean_shape, n_landmarks, sparsed=True):
    reference_shape = PointCloud(mean_shape.points[:n_landmarks]) \
        if sparsed else mean_shape

    from menpofit.aam.base import build_reference_frame as brf

    return brf(reference_shape)