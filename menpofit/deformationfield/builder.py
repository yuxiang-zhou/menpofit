from menpofit.aam import AAMBuilder
from menpofit.base import create_pyramid
from menpofit.transform import DifferentiableThinPlateSplines
from menpofit.transform import DifferentiablePiecewiseAffine
from menpofit.builder import normalization_wrt_reference_shape
from menpo.feature import igo
from menpo.shape import PointCloud
from menpo.transform.groupalign.base import MultipleAlignment
from menpo.math import principal_component_decomposition as pca
from menpo.model import PCAModel
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.transform import Translation

import numpy as np


class ICP(MultipleAlignment):
    def __init__(self, sources, target=None):
        self._test_iteration = []
        self.transformations = []
        self.point_correspondence = []

        # sort sources in number of points
        sources = np.array(sources)
        sortindex = np.argsort(np.array([s.n_points for s in sources]))[-1::-1]
        sources = sources[sortindex]

        if target is None:
            target = sources[0]
        sources = sources[sortindex]

        super(ICP, self).__init__(sources, target)

        self.aligned_shapes = np.array(
            [self._align_source(s) for s in sources]
        )

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
        # p0, r, sm, tm = self._pca_align(source)
        # transforms.append([r, sm, tm])
        p0 = source.points
        iters = [source.points, p0]

        a_p = _align(p0)

        _, point_corr = self._cloest_points(a_p)

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
    def __init__(self, features=igo, transform=DifferentiableThinPlateSplines,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=False,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=0):
        super(DeformationFieldBuilder, self).__init__(
            features, transform, trilist, normalization_diagonal, n_levels,
            downscale, scaled_shape_models, max_shape_components,
            max_appearance_components, boundary)

    def build(self, images, group=None, label=None, verbose=False):
        r"""
        Builds a Multilevel Active Appearance Model from a list of
        landmarked images.

        Parameters
        ----------
        images : list of :map:`MaskedImage`
            The set of landmarked images from which to build the AAM.

        group : `string`, optional
            The key of the landmark set that should be used. If ``None``,
            and if there is only one set of landmarks, this set will be used.

        label : `string`, optional
            The label of of the landmark manager that you wish to use. If no
            label is passed, the convex hull of all landmarks is used.

        verbose : `boolean`, optional
            Flag that controls information and progress printing.

        Returns
        -------
        aam : :map:`AAM`
            The AAM object. Shape and appearance models are stored from lowest
            to highest level
        """
        # compute reference_shape and normalize images size
        self.reference_shape, normalized_images = \
            normalization_wrt_reference_shape(
                images, group, label, self.normalization_diagonal, verbose
            )

        # create pyramid
        generators = create_pyramid(normalized_images, self.n_levels,
                                    self.downscale, self.features,
                                    verbose=verbose)

        # build the model at each pyramid level
        if verbose:
            if self.n_levels > 1:
                print_dynamic('- Building model for each of the {} pyramid '
                              'levels\n'.format(self.n_levels))
            else:
                print_dynamic('- Building model\n')

        shape_models = []
        appearance_models = []

        # for each pyramid level (high --> low)
        for j in range(self.n_levels):
            # since models are built from highest to lowest level, the
            # parameters in form of list need to use a reversed index
            rj = self.n_levels - j - 1

            if verbose:
                level_str = '  - '
                if self.n_levels > 1:
                    level_str = '  - Level {}: '.format(j + 1)

            # get feature images of current level
            feature_images = []
            for c, g in enumerate(generators):
                if verbose:
                    print_dynamic(
                        '{}Computing feature space/rescaling - {}'.format(
                            level_str,
                            progress_bar_str((c + 1.) / len(generators),
                                             show_bar=False)))
                feature_images.append(next(g))

            # extract potentially rescaled shapes
            shapes = [i.landmarks[group][label] for i in feature_images]

            # define shapes that will be used for training
            if j == 0:
                original_shapes = shapes
                train_shapes = shapes
            else:
                if self.scaled_shape_models:
                    train_shapes = shapes
                else:
                    train_shapes = original_shapes

            # train shape model and find reference frame
            if verbose:
                print_dynamic('{}Building shape model'.format(level_str))
            shape_model = self._build_shape_model(
                train_shapes, self.max_shape_components[rj])
            reference_frame = self._build_reference_frame(shape_model.mean())

            # add shape model to the list
            shape_models.append(shape_model)

            # compute transforms
            transforms = self._compute_transforms(reference_frame,
                                                  feature_images, group,
                                                  label, verbose, level_str)

            # warp images to reference frame
            warped_images = []
            for c, (i, t) in enumerate(zip(feature_images, transforms)):
                if verbose:
                    print_dynamic('{}Warping images - {}'.format(
                        level_str,
                        progress_bar_str(float(c + 1) / len(feature_images),
                                         show_bar=False)))
                si = i.rescale(np.power(self.downscale, j))
                warped_images.append(si.warp_to_mask(reference_frame.mask, t))

            # attach reference_frame to images' source shape
            for i in warped_images:
                i.landmarks['source'] = reference_frame.landmarks['source']

            # build appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))
            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components[rj] is not None:
                appearance_model.trim_components(
                    self.max_appearance_components[rj])

            # add appearance model to the list
            appearance_models.append(appearance_model)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        shape_models.reverse()
        appearance_models.reverse()
        n_training_images = len(images)

        return self._build_aam(shape_models, appearance_models,
                               n_training_images)

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
                                n_training_images,
                                DifferentiableThinPlateSplines,
                                self.features, self.reference_shape,
                                self.downscale, self.scaled_shape_models)

    def _build_shape_model(self, shapes, max_components):
        # Align Shapes Using ICP
        self._icp = icp = ICP(shapes)
        aligned_shapes = icp.aligned_shapes

        # Build Reference Frame from Aligned Shapes
        bound_list = []
        for s in aligned_shapes:
            bmin, bmax = s.bounds()
            bound_list.append(bmin)
            bound_list.append(bmax)
            bound_list.append(np.array([bmin[0], bmax[1]]))
            bound_list.append(np.array([bmax[0], bmin[1]]))
        bound_list = PointCloud(np.array(bound_list))

        self.reference_frame = super(
            DeformationFieldBuilder, self
        )._build_reference_frame(bound_list)

        # Set All True Pixels for Mask
        self.reference_frame.mask.pixels = np.ones(
            self.reference_frame.mask.pixels.shape, dtype=np.bool)

        # Get Dense Shape from Masked Image
        dense_reference_shape = PointCloud(
            self.reference_frame.mask.true_indices()
        )

        # Set Dense Shape as Reference Landmarks
        self.reference_frame.landmarks['source'] = dense_reference_shape

        # compute non-linear transforms (tps)
        self._aligned_shapes = []
        self._shapes = shapes
        transforms = []

        align_centre = icp.target.centre_of_bounds()
        align_t = Translation(
            dense_reference_shape.centre_of_bounds()-align_centre
        )
        align_corr = icp.point_correspondence

        for a_s, a_corr in zip(aligned_shapes, align_corr):
            # Align shapes with reference frame
            temp_as = align_t.apply(a_s)
            temp_s = align_t.apply(PointCloud(icp.target.points[a_corr]))

            self._aligned_shapes.append(temp_as)
            transforms.append(self.transform(temp_s, temp_as))

        self.transforms = transforms

        # build dense shapes
        dense_shapes = []
        for i, (t, s) in enumerate(zip(transforms, shapes)):
            warped_points = t.apply(dense_reference_shape)
            dense_shape = warped_points
            dense_shapes.append(dense_shape)

        self._dense_shapes = dense_shapes

        # build dense shape model
        dense_shape_model = super(DeformationFieldBuilder, self). \
            _build_shape_model(dense_shapes, max_components)

        return dense_shape_model

    def _compute_transforms(self, reference_frame, feature_images, group,
                            label, verbose, level_str):
        if verbose:
            print_dynamic('{}Computing transforms'.format(level_str))

        transforms = []
        for a_s, s, i in zip(self._aligned_shapes, self._shapes,
                             feature_images):
            image_center = i.landmarks[group][label].centre_of_bounds()
            # Align shapes with images
            temp_as = Translation(
                image_center-a_s.centre_of_bounds()
            ).apply(a_s)

            temp_s = Translation(
                image_center-s.centre_of_bounds()
            ).apply(s)

            transforms.append(self.transform(temp_as, temp_s))

        return self.transforms

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


def build_reference_frame(mean_shape):
    reference_shape = mean_shape

    from menpofit.aam.base import build_reference_frame as brf

    return brf(reference_shape)