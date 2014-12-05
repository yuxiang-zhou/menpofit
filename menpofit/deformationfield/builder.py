from menpofit.aam import AAMBuilder
from menpofit.transform import DifferentiablePiecewiseAffine
from menpofit.builder import normalization_wrt_reference_shape
from menpo.feature import igo
from menpo.shape import PointCloud
from menpo.transform.groupalign.base import MultipleAlignment
from menpo.math import principal_component_decomposition as pca

import numpy as np


class ICP(MultipleAlignment):
    def __init__(self, sources, target=None):
        self._test_iteration = []
        self.transformations = []
        self.point_correspondence = []

        sources = np.array(sources)
        sources = sources[
            np.argsort(np.array([s.n_points for s in sources]))[-1::-1]
        ]

        if target is None:
            target = sources[0]

        super(ICP, self).__init__(sources, target)

        self.aligned_shapes = [self._align_source(s) for s in sources]

    def _align_source(self, source, eps=1e-3, max_iter=100):
        # align helper function
        def _align(i_s):
            # Align Shapes
            it = 0
            pf = i_s
            n_p = i_s.shape[0]
            tolerance_old = tolerance = eps + 1
            while tolerance > eps and it < max_iter:
                pk = pf

                # Compute Closest Points
                yk, _ = self._cloest_points(pk)

                # Compute Registration
                pf, rot, smean, tmean = self._compute_registration(pk, yk)
                transforms.append([rot, smean, tmean])

                # Update source
                # pf = self._update_source(pk, np.hstack((qr, qt)))

                # Calculate Mean Square Matching Error
                tolerance_new = np.sum(np.power(pf - yk, 2)) / n_p
                tolerance = abs(tolerance_old - tolerance_new)
                tolerance_old = tolerance_new

                it += 1
                iters.append(pf)

            return pf

        transforms = []

        # Initial Alignment using PCA
        p0, r, sm, tm = self._pca_align(source)
        transforms.append([r, sm, tm])
        iters = [source.points, p0]

        a_p = _align(p0)

        _, point_corr = self._cloest_points(self.target.points, a_p)

        self._test_iteration.append(iters)
        self.transformations.append(transforms)
        self.point_correspondence.append(point_corr)

        return PointCloud(a_p)

    def _pca_align(self, source):
        # Apply PCA on both source and target
        svecs, svals, smean = pca(source.points)
        tvecs, tvals, tmean = pca(self.target.points)

        # Compute Rotation
        svec = svecs[np.argmax(svals)]
        tvec = tvecs[np.argmax(tvals)]

        sang = np.arctan2(svec[1], svec[0])
        tang = np.arctan2(tvec[1], tvec[0])

        da = sang - tang

        tr = np.array([[np.cos(da), np.sin(da)],
                       [-1*np.sin(da), np.cos(da)]])

        # Compute Aligned Point
        pt = np.array([tr.dot(s - smean) + tmean for s in source.points])

        return pt, tr, smean, tmean

    def _update_source(self, p, q):
        return _apply_q(p, q)[:, :p.shape[1]]

    def _compute_registration(self, p, x):
        # Calculate Covariance
        up = np.mean(p, axis=0)
        ux = np.mean(x, axis=0)
        u = up[:, None].dot(ux[None, :])
        n_p = p.shape[0]
        cov = sum([pi[:, None].dot(xi[None, :])
                   for (pi, xi) in zip(p, x)]) / n_p - u

        # Apply SVD
        U, W, T = np.linalg.svd(cov)

        # Calculate Rotation Matrix
        qr = T.T.dot(U.T)
        # Calculate Translation Point
        pk = np.array([qr.dot(s - up) + ux for s in p])

        return pk, qr, up, ux

    def _cloest_points(self, source, target=None):
        points = np.array([self._closest_node(s, target) for s in source])

        return np.vstack(points[:, 0]), np.hstack(points[:, 1])

    def _closest_node(self, node, target=None):
        if target is None:
            target = self.target

        nodes = target
        if isinstance(target, PointCloud):
            nodes = np.array(target.points)

        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        index = np.argmin(dist_2)
        return nodes[index], index


def _compose_r(qr):
    q0, q1, q2, q3 = qr
    r = np.zeros((3, 3))
    r[0, 0] = np.sum(np.power(qr, 2)) * [1, 1, -1, -1]
    r[1, 1] = np.sum(np.power(qr[[0, 2, 1, 3]], 2)) * [1, 1, -1, -1]
    r[2, 2] = np.sum(np.power(qr[[0, 3, 1, 2]], 2)) * [1, 1, -1, -1]
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

    def build(self, images, group=None, label=None, verbose=False, icp=True):
        _, normalized_image = \
            self._icp_alignment(
                images, group, label, verbose
            )

        if icp:
            return super(DeformationFieldBuilder, self).build(
                normalized_image, 'ICP', label, verbose
            )
        else:
            return super(DeformationFieldBuilder, self).build(
                images, group, label, verbose
            )

    def _icp_alignment(
            self, images, group, label, verbose=False):

        reference_shape, normalized_image = \
            normalization_wrt_reference_shape(
                images, group, label, self.normalization_diagonal, verbose
            )

        shapes = [i.landmarks[group][label] for i in normalized_image]
        icp = ICP(shapes)

        icp_shapes = [
            PointCloud(
                shapes[i].points[icp.point_correspondence[i]]
            ) for i in range(
                icp.n_sources
            )
        ]

        shapes = icp_shapes

        for i, s in enumerate(shapes):
            normalized_image[i].landmarks['ICP'] = s

        return reference_shape, normalized_image

    def _build_shape_model(self, shapes, max_components):
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