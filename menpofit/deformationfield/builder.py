from menpofit.aam import AAMBuilder
from menpofit.aam.lineerror import interpolate
from menpofit.base import create_pyramid
from menpofit.transform import DifferentiableThinPlateSplines
from menpo.transform.base import Transform
from menpofit.deformationfield import SVS
from menpofit.builder import normalization_wrt_reference_shape
from menpofit.deformationfield.MatlabExecuter import MatlabExecuter
from menpo.feature import igo
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform.groupalign.base import MultipleAlignment
from menpo.math import pca
from menpo.model import PCAModel
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.transform import Translation, AlignmentSimilarity
from menpo.transform.icp import nicp
from menpo.shape import TriMesh
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean as dist
from skimage import filters
from os.path import  isfile

import os
import sys
import uuid
import subprocess
import numpy as np
import menpo.io as mio
import scipy.io as sio


# Optical Flow Transform
class OpticalFlowTransform(Transform):
    def __init__(self, u, v):
        super(OpticalFlowTransform, self).__init__()
        self._u = v
        self._v = u

    def _apply(self, x, **kwargs):
        ret = x.copy()
        for p in ret:
            i, j = p[0].astype(int), p[1].astype(int)
            p += np.array([self._u[i, j], self._v[i, j]])
        return ret


# ICP --------------------------------------------
class ICP(MultipleAlignment):
    def __init__(self, sources, target=None):
        self._test_iteration = []
        self.transformations = []
        self.point_correspondence = []

        # sort sources in number of points
        sources = np.array(sources)
        sortindex = np.argsort(np.array([s.n_points for s in sources]))[-1::-1]
        sort_sources = sources[sortindex]

        # Set first source as target (e.g. having most number of points)
        if target is None:
            target = sort_sources[0]

        super(ICP, self).__init__(sources, target)

        # Align Source with Target
        self.aligned_shapes = np.array(
            [self._align_source(s) for s in sources]
        )

    def _align_source(self, source, eps=1e-3, max_iter=100):

        # Initial Alignment using PCA
        # p0, r, sm, tm = self._pca_align(source)
        # transforms.append([r, sm, tm])
        p0 = source.points

        a_p, transforms, iters, point_corr = self._align(p0, eps, max_iter)
        iters = [source.points, p0] + iters

        self._test_iteration.append(iters)
        self.transformations.append(transforms)
        self.point_correspondence.append(point_corr)

        return PointCloud(a_p)

    def _align(self, i_s, eps, max_iter):
        # Align Shapes
        transforms = []
        iters = []
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

        _, point_corr = self._cloest_points(pf)

        return pf, transforms, iters, point_corr

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


class NICP(ICP):
    def __init__(self, sources, target=None):
        self.n_dims = sources[0].n_dims
        super(NICP, self).__init__(sources, target)

    def _align(self, tplt, eps, max_iter):

        # Configuration
        higher = 2001
        lower = 1
        step = 100
        transforms = []
        iters = []

        # Build TriMesh Source
        tplt_tri = TriMesh(tplt).trilist

        # Generate Edge List
        tplt_edge = tplt_tri[:, [0, 1]]
        tplt_edge = np.vstack((tplt_edge, tplt_tri[:, [0, 2]]))
        tplt_edge = np.vstack((tplt_edge, tplt_tri[:, [1, 2]]))
        tplt_edge = np.sort(tplt_edge)

        # Get Unique Edge List
        b = np.ascontiguousarray(tplt_edge).view(
            np.dtype((np.void, tplt_edge.dtype.itemsize * tplt_edge.shape[1]))
        )
        _, idx = np.unique(b, return_index=True)
        tplt_edge = tplt_edge[idx]

        # init
        m = tplt_edge.shape[0]
        n = tplt.shape[0]

        # get node-arc incidence matrix
        M = np.zeros((m, n))
        M[range(m), tplt_edge[:, 0]] = -1
        M[range(m), tplt_edge[:, 1]] = 1

        # weight matrix
        G = np.identity(self.n_dims+1)

        # build the kD-tree
        target_2d = self.target.points
        kdOBJ = KDTree(target_2d)

        # init tranformation
        prev_X = np.zeros((self.n_dims, self.n_dims+1))
        prev_X = np.tile(prev_X, n).T
        tplt_i = tplt

        # start nicp
        # for each stiffness
        sf = np.logspace(lower, higher, num=step, base=1.005)[-1::-1]
        sf_kron = np.kron(M, G)
        errs = []

        for alpha in sf:
            # get the term for stiffness
            sf_term = alpha*sf_kron
            # iterate until X converge
            while True:
                # find nearest neighbour
                _, match = kdOBJ.query(tplt_i)

                # formulate target and template data, and distance term
                U = target_2d[match, :]

                point_size = self.n_dims+1
                D = np.zeros((n, n*point_size))
                for k in range(n):
                    D[k, k*point_size:k*point_size+2] = tplt_i[k, :]
                    D[k, k*point_size+2] = 1

                # % correspondence detection for setting weight
                # add distance term
                sA = np.vstack((sf_term, D))
                sB = np.vstack((np.zeros((sf_term.shape[0], self.n_dims)), U))
                sX = np.linalg.pinv(sA).dot(sB)

                # deform template
                tplt_i = D.dot(sX)
                err = np.linalg.norm(prev_X-sX, ord='fro')
                errs.append([alpha, err])
                prev_X = sX

                transforms.append(sX)
                iters.append(tplt_i)

                if err/np.sqrt(np.size(prev_X)) < eps:
                    break

        # final result
        fit_2d = tplt_i
        _, point_corr = kdOBJ.query(fit_2d)
        return fit_2d, transforms, iters, point_corr


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

# END ICP ----------------------------------------


# Deformation Field using ICP, NICP --------------
class DeformationFieldBuilder(AAMBuilder):
    def __init__(self, features=igo, transform=DifferentiableThinPlateSplines,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=False,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=0, template=0):
        super(DeformationFieldBuilder, self).__init__(
            features, transform, trilist, normalization_diagonal, n_levels,
            downscale, scaled_shape_models, max_shape_components,
            max_appearance_components, boundary)
        self.template = template

    def build(self, images, group=None, label=None, verbose=False,
              target_shape=None):
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
                images, group, label, self.normalization_diagonal, target_shape, verbose
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
        self._feature_images = []
        self._warpped_images = []
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


            self._feature_images.append(feature_images)

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
            if j == 0:
                shape_model = self._build_shape_model(
                    train_shapes, self.max_shape_components[rj],
                    target_shape
                )
            else:
                if self.scaled_shape_models:
                    shape_model = self._build_shape_model(
                        train_shapes, self.max_shape_components[rj],
                        target_shape
                    )
                else:
                    shape_model = shape_models[-1].copy()

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
                si = self._image_pre_process(i, j, c)
                warped_images.append(si.warp_to_mask(reference_frame.mask, t))
            self._warpped_images.append(warped_images)

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
                                self.downscale, self.scaled_shape_models,
                                self.reference_frame, self._icp,
                                self.normalization_diagonal,
                                self.n_landmarks, self.group_corr)

    def _build_shape_model(self, shapes, max_components, target_shape):
        # Simulate inconsist annotation
        sample_groups = []
        g_i = self._feature_images[0][self.template].landmarks[
            'groups'].items()
        lindex = 0
        for i in g_i:
            g_size = i[1].n_points
            sample_groups.append(range(lindex, lindex + g_size))
            lindex += g_size

        # Align Shapes Using ICP
        if target_shape is None:
            target_shape = shapes[self.template]

        sample_shapes = shapes
        self._icp = icp = ICP(sample_shapes, target_shape)
        aligned_shapes = icp.aligned_shapes

        # Store Removed Transform
        self._removed_transform = []
        for a_s, s in zip(aligned_shapes, sample_shapes):
            ast = AlignmentSimilarity(a_s, s)
            self._removed_transform.append(ast)

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

        # Transforms to align reference frame
        align_centre = icp.target.centre_of_bounds()
        align_t = Translation(
            self.reference_frame.centre - align_centre
        )

        self._rf_align = Translation(
            align_centre - self.reference_frame.centre
        )
        # Mask Reference Frame
        self.reference_frame.landmarks['sparse'] = align_t.apply(icp.target)
        self.reference_frame.constrain_mask_to_landmarks(group='sparse')

        # Get Dense Shape from Masked Image
        dense_reference_shape = PointCloud(
            np.vstack((
                align_t.apply(icp.target).points,
                self.reference_frame.mask.true_indices()
            ))
        )
        # dense_reference_shape = PointCloud(
        #     self.reference_frame.mask.true_indices()
        # )
        # Set Dense Shape as Reference Landmarks
        self.reference_frame.landmarks['source'] = dense_reference_shape
        self._shapes = shapes
        self._aligned_shapes = []
        transforms = []

        # group correspondence
        align_gcorr = None
        self.group_corr = groups = np.array(sample_groups)
        for g in groups:
            g_align_s = []
            for aligned_s in icp.aligned_shapes:
                g_align_s.append(PointCloud(aligned_s.points[g]))
            _, point_correspondence = FastNICP(g_align_s, PointCloud(icp.target.points[g]))
            g_align = np.array(point_correspondence) + g[0]
            if align_gcorr is None:
                align_gcorr = g_align
            else:
                align_gcorr = np.hstack((align_gcorr, g_align))

        # compute non-linear transforms (tps)
        for a_s, a_corr in zip(aligned_shapes, align_gcorr):
            # Align shapes with reference frame
            temp_as = align_t.apply(a_s)
            temp_s = align_t.apply(PointCloud(icp.target.points[a_corr]))

            self._aligned_shapes.append(temp_as)
            transforms.append(self.transform(temp_s, temp_as))
            # transforms.append(pwa(temp_s, temp_as))

        self.transforms = transforms
        self._corr = align_gcorr

        # build dense shapes
        dense_shapes = []
        for i, t in enumerate(transforms):
            warped_points = t.apply(dense_reference_shape)
            dense_shape = warped_points
            dense_shapes.append(dense_shape)

        self._dense_shapes = dense_shapes

        # build dense shape model
        dense_shape_model = super(DeformationFieldBuilder, self). \
            _build_shape_model(dense_shapes, max_components)

        self.n_landmarks = icp.target.points.shape[0]

        return dense_shape_model

    def _image_pre_process(self, img, scale, index):
        si = img.rescale(np.power(self.downscale, scale))
        return si

    def _compute_transforms(self, reference_frame, feature_images, group,
                            label, verbose, level_str):
        if verbose:
            print_dynamic('{}Computing transforms'.format(level_str))

        transforms = []
        for t, rt in zip(self.transforms, self._removed_transform):
            ct = t.compose_before(self._rf_align).compose_before(rt)
            transforms.append(ct)
        return transforms

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


# Deformation Field using SVS, Optical Flow
class OpticalFieldBuilder(DeformationFieldBuilder):

    def __init__(self, features=igo, transform=DifferentiableThinPlateSplines,
                 trilist=None, normalization_diagonal=None, n_levels=3,
                 downscale=2, scaled_shape_models=False,
                 max_shape_components=None, max_appearance_components=None,
                 boundary=10, template=0):
        self._svs_path = None
        self._flow_path = None
        self._is_mc = True
        self._alpha = 15
        self._shape_desc = 'SVS'
        super(OpticalFieldBuilder, self).__init__(
            features, transform,
            trilist, normalization_diagonal, n_levels,
            downscale, scaled_shape_models,
            max_shape_components, max_appearance_components,
            boundary, template
        )

    def build(self, images, group=None, label=None, verbose=False,
              target_shape=None, svs_path=None, flow_path=None, multi_channel=True, alpha=15, shape_desc='SVS'):

        self._svs_path = svs_path
        self._flow_path = flow_path
        self._is_mc = multi_channel
        self._alpha = alpha
        self._shape_desc = shape_desc
        if target_shape is None:
            target_shape = images[self.template].landmarks['PTS'].lms
        else:
            self.template = -1

        return super(OpticalFieldBuilder, self).build(
            images, group, label, verbose, target_shape
        )

    def _build_shape_model(self, shapes, max_components, target_shape):

        # Parameters
        alpha = self._alpha
        pdm = 0
        n_shapes = len(shapes)
        # Simulate inconsist annotation
        sample_groups = []
        g_i = self._feature_images[0][self.template].landmarks[
            'groups'].items()
        lindex = 0
        for i in g_i:
            g_size = i[1].n_points
            sample_groups.append(range(lindex, lindex + g_size))
            lindex += g_size

        # Align Shapes Using ICP

        self._icp = icp = ICP(shapes, target_shape)
        aligned_shapes = icp.aligned_shapes
        # Store Removed Transform
        self._removed_transform = []
        self._icp_transform = []
        for a_s, s in zip(aligned_shapes, shapes):
            ast = AlignmentSimilarity(a_s, s)
            self._removed_transform.append(ast)
            icpt = AlignmentSimilarity(s, a_s)
            self._icp_transform.append(icpt)

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
        # Translation between reference shape and aliened shapes
        align_centre = icp.target.centre_of_bounds()
        align_t = Translation(
            self.reference_frame.centre - align_centre
        )

        self._rf_align = Translation(
            align_centre - self.reference_frame.centre
        )

        # Set All True Pixels for Mask
        self.reference_frame.mask.pixels = np.ones(
            self.reference_frame.mask.pixels.shape, dtype=np.bool)

        # Mask Reference Frame
        n_landmarks = icp.target.points.shape[0]
        self.reference_frame.landmarks['sparse'] = align_t.apply(icp.target)
        # self.reference_frame.constrain_mask_to_landmarks(group='sparse')

        # Get Dense Shape from Masked Image
        dense_reference_shape = PointCloud(
            self.reference_frame.mask.true_indices()
        )

        # Set Dense Shape as Reference Landmarks
        # self.reference_frame.landmarks['source'] = dense_reference_shape
        self._shapes = shapes
        self._aligned_shapes = []

        # Create Cache Directory
        home_dir = os.getcwd()
        dir_hex = uuid.uuid1()
        svs_path_in = '{}/.cache/{}/svs_training'.format(home_dir, dir_hex)
        svs_path_out = '{}/.cache/{}/svs_result'.format(home_dir, dir_hex)
        matE = MatlabExecuter()
        mat_code_path = '/vol/atlas/homes/yz4009/gitdev/mfsfdev'
        
        # Skip building svs is path specified
        if self._svs_path is None:
            if not os.path.exists(svs_path_in):
                os.makedirs(svs_path_in)
            # Build Transform Using SVS
            xr, yr = self.reference_frame.shape
            for j, a_s in enumerate([target_shape] + aligned_shapes.tolist()):
                print_dynamic("  - SVS Training {} out of {}".format(
                    j, len(aligned_shapes) + 1)
                )
                # Align shapes with reference frame
                temp_as = align_t.apply(a_s)
                points = temp_as.points
                # Construct tplt_edge
                tplt_edge = None
                lindex = 0
                # Get Grouped Landmark Indexes
                if j > 0:
                    g_i = self._feature_images[0][j-1].landmarks['groups'].items()
                else:
                    g_i = self._feature_images[0][j].landmarks['groups'].items()
                    if not g_i[0][1].n_points == a_s.n_points:
                        g_i = [['Reference', a_s]]

                edge_g = []
                edge_ig = []
                for g in g_i:
                    g_size = g[1].n_points
                    rindex = g_size+lindex
                    edges_range = np.array(range(lindex, rindex))
                    edge_ig.append(edges_range)
                    edges = np.hstack((
                        edges_range[:g_size-1, None], edges_range[1:, None]
                    ))
                    edge_g.append(edges)
                    tplt_edge = edges if tplt_edge is None else np.vstack((
                        tplt_edge, edges
                    ))
                    lindex = rindex

                tplt_edge = np.concatenate(edge_g)

                # Store SVS Image
                if self._shape_desc == 'SVS':
                    svs = SVS(
                        points, tplt_edge=tplt_edge, tolerance=3, nu=0.8,
                        gamma=0.8, max_f=20
                    )
                    store_image = svs.svs_image(range(xr), range(yr))
                elif self._shape_desc == 'draw':
                    store_image = sample_points(points, xr, yr, edge_ig)
                elif self._shape_desc == 'draw_gaussian':
                    ni = sample_points(points, xr, yr, edge_ig)
                    store_image = Image.init_blank(ni.shape)
                    store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                elif self._shape_desc == 'sample_gaussian':
                    ni = Image.init_blank((xr, yr))
                    for pts in points:
                        ni.pixels[0, pts[0], pts[1]] = 1
                    store_image = Image.init_blank(ni.shape)
                    store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                elif self._shape_desc == 'sample':
                    store_image = Image.init_blank((xr, yr))
                    for pts in points:
                        store_image.pixels[0, pts[0], pts[1]] = 1
                else:
                    raise Exception('Undefined Shape Descriptor: {}'.format(self._shape_desc))

                mio.export_image(
                    store_image,
                    '{}/svs_{:04d}.png'.format(svs_path_in, j),
                    overwrite=True
                )

                # Train Group SVS
                for ii, g in enumerate(edge_ig):
                    g_size = points[g].shape[0]
                    edges_range = np.array(range(g_size))
                    edges = np.hstack((
                        edges_range[:g_size-1, None], edges_range[1:, None]
                    ))

                    # Store SVS Image
                    if self._shape_desc == 'SVS':
                        svs = SVS(
                            points[g], tplt_edge=edges, tolerance=3, nu=0.8,
                            gamma=0.8, max_f=20
                        )
                        store_image = svs.svs_image(range(xr), range(yr))
                    elif self._shape_desc == 'draw':
                        store_image = sample_points(points[g], xr, yr)
                    elif self._shape_desc == 'draw_gaussian':
                        ni = sample_points(points[g], xr, yr, edge_ig)
                        store_image = Image.init_blank(ni.shape)
                        store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                    elif self._shape_desc == 'sample_gaussian':
                        ni = Image.init_blank((xr, yr))
                        for pts in points[g]:
                            ni.pixels[0, pts[0], pts[1]] = 1
                        store_image = Image.init_blank(ni.shape)
                        store_image.pixels[0,:,:] = filters.gaussian_filter(np.squeeze(ni.pixels), 1)
                    elif self._shape_desc == 'sample':
                        store_image = Image.init_blank((xr, yr))
                        for pts in points[g]:
                            store_image.pixels[0, pts[0], pts[1]] = 1
                    else:
                        raise Exception('Undefined Shape Descriptor: {}'.format(self._shape_desc))

                    mio.export_image(
                        store_image,
                        '{}/svs_{:04d}.png'.format(svs_path_in, j),
                        overwrite=True
                    )

                # Create gif from svs group
                #     convert -delay 10 -loop 0 svs_0001_g*.png test.gif
                subprocess.Popen([
                    'convert',
                    '-delay', '10', '-loop', '0',
                    '{0}/svs_{1:04d}_g*.png'.format(svs_path_in, j),
                    '{0}/svs_{1:04d}.gif'.format(svs_path_in, j)])
        else:
            svs_path_in = self._svs_path
            svs_path_out = '{}'.format(svs_path_in)
            seg_mask_points = mio.import_pickle(
                svs_path_in+'/seg_mask_points_group.pkl'
            )
            # compute and normalize mask images size
            # single channel ---------------------------------------------
            mask_images = []
            for mii in range(n_shapes):
                mi = mio.import_image(
                    '{}/{:04d}_mask.png'.format(svs_path_in, mii + 1)
                )
                mi.landmarks['PTS'] = mio.import_landmark_file(
                    svs_path_in + '/{:04d}.pts'.format(mii+1)
                )
                if mi.n_channels == 3:
                        mi = mi.as_greyscale()
                mask_images.append(mi)

            nshape = mask_images[0].shape
            nlms = mask_images[0].landmarks['PTS'].lms.n_points

            _, normalized_mask_images = \
                normalization_wrt_reference_shape(
                    mask_images, 'PTS', None,
                    self.normalization_diagonal, target_shape, False
                )

            for index, mi in enumerate(normalized_mask_images[:len(shapes)]):

                nmi = mi.warp_to_shape(
                    mi.shape, self._removed_transform[index],
                    warp_landmarks=True
                ).warp_to_shape(
                    self.reference_frame.shape, self._rf_align,
                    warp_landmarks=True
                )
                if nmi.n_channels == 3:
                    nmi = nmi.as_greyscale()

                mio.export_image(
                    nmi, '{}/svs_{:04d}.png'.format(svs_path_in, index+1),
                    overwrite=True
                )
                mio.export_landmark_file(
                    nmi.landmarks['PTS'],
                    svs_path_in + '/svs_{:04d}.pts'.format(index+1),
                    overwrite=True
                )
            # end single channel ---------------------------------------------

            # normalise multichannel images
            # multichannel -----------------------------------------------------

            for iindex in range(n_shapes): #{[mask, Z]}:
                msi = seg_mask_points[iindex]
                # for every image
                mask_seg_images = []
                for iseg in range(nlms):
                    seg_img = Image.init_blank(nshape)
                    seg_img.landmarks['PTS'] = mio.import_landmark_file(
                        svs_path_in + '/{:04d}.pts'.format(iindex+1)
                    )
                    for pt in msi[0][np.where(msi[1] == iseg)]:
                        try:
                            seg_img.pixels[0, pt[0], pt[1]] = 1
                        except IndexError:
                            print 'Index Error'
                    mask_seg_images.append(seg_img)

                _, normalized_seg_images = \
                    normalization_wrt_reference_shape(
                        mask_seg_images, 'PTS', None,
                        self.normalization_diagonal, target_shape, False
                    )

                for index, mi in enumerate(normalized_seg_images[:len(shapes)]):

                    nmi = mi.warp_to_shape(
                        mi.shape, self._removed_transform[iindex],
                        warp_landmarks=True
                    ).warp_to_shape(
                        self.reference_frame.shape, self._rf_align,
                        warp_landmarks=True
                    )

                    if nmi.n_channels == 3:
                        nmi = nmi.as_greyscale()

                    # print(iindex,index, nmi.shape, self.reference_frame.shape)

                    mio.export_image(
                        nmi, svs_path_in + '/svs_{:04d}_g{:02d}.png'.format(
                            iindex+1, index
                        ), overwrite=True
                    )

                subprocess.Popen([
                    'convert',
                    '-delay', '10', '-loop', '0',
                    '{}/svs_{:04d}_g*.png'.format(svs_path_in, iindex+1),
                    '{}/svs_{:04d}.gif'.format(svs_path_in, iindex+1)])

                print_dynamic(' - Generating Multi-Channel Images: {}/{}'.format(
                    iindex+1, n_shapes
                ))
            # end multi channel ------------------------------------------------

        print_dynamic('  - Building Trajectory Basis')
        nFrame = len(icp.aligned_shapes)

        if self._flow_path is None and False:
            # Build basis
            # group correspondence
            align_gcorr = None
            groups = np.array(sample_groups)
            tps_t = []

            if self._is_mc:
                for g in groups:
                    g_align_s = []
                    for aligned_s in icp.aligned_shapes:
                        g_align_s.append(PointCloud(aligned_s.points[g]))
                    # _, point_correspondence = FastNICP(
                    #   g_align_s, PointCloud(icp.target.points[g])
                    # )
                    gnicp = NICP(g_align_s, PointCloud(icp.target.points[g]))
                    g_align = np.array(gnicp.point_correspondence) + g[0]
                    if align_gcorr is None:
                        align_gcorr = g_align
                    else:
                        align_gcorr = np.hstack((align_gcorr, g_align))
            else:
                print 'single channel basis'
                _, point_correspondence = FastNICP(icp.aligned_shapes, icp.target)
                # gnicp = NICP(icp.aligned_shapes, icp.target)
                align_gcorr = point_correspondence

            # compute non-linear transforms (tps)
            for a_s, a_corr in zip(aligned_shapes, align_gcorr):
                # Align shapes with reference frame
                temp_as = align_t.apply(a_s)
                temp_s = align_t.apply(PointCloud(icp.target.points[a_corr]))

                self._aligned_shapes.append(temp_as)
                tps_t.append(self.transform(temp_s, temp_as))
                # transforms.append(pwa(temp_s, temp_as))

            # build dense shapes
            dense_shapes = []
            for i, t in enumerate(tps_t):
                warped_points = t.apply(dense_reference_shape)
                dense_shape = warped_points
                dense_shapes.append(dense_shape)

            # build dense shape model
            uvs = np.array([ds.points.flatten() - dense_reference_shape.points.flatten()
                            for ds in dense_shapes])
            nPoints = dense_shapes[0].n_points
            h, w = self.reference_frame.shape
            W = np.zeros((2 * nFrame, nPoints))
            v = uvs[:, 0:2*nPoints:2]
            u = uvs[:, 1:2*nPoints:2]

            u = np.transpose(np.reshape(u.T, (w, h, nFrame)), [1, 0, 2])
            v = np.transpose(np.reshape(v.T, (w, h, nFrame)), [1, 0, 2])

            W[0:2*nFrame:2, :] = np.reshape(u, (w*h, nFrame)).T
            W[1:2*nFrame:2, :] = np.reshape(v, (w*h, nFrame)).T

            S = W.dot(W.T)
            U, var, _ = np.linalg.svd(S)
            # csum = np.cumsum(var)
            # csum = 100 * csum / csum[-1]
            # accept_rate = 99.9
            # # rank = np.argmin(np.abs(csum - accept_rate))
            Q = U[:, :]
            basis = np.vstack((Q[1::2, :], Q[0::2, :]))

            # construct basis
            sio.savemat('{}/{}'.format(svs_path_in, 'bas.mat'), {
                'bas': basis,
                'tps_u': W[0:2*nFrame:2, :],
                'tps_v': W[1:2*nFrame:2, :],
                'ou': uvs[:, 1:2*nPoints:2],
                'ov': uvs[:, 0:2*nPoints:2],
                'tu': u,
                'tv': v
            })

        # Call Matlab to Build Flows
        if self._flow_path is None:
            print_dynamic('  - Building Shape Flow')
            matE.cd(mat_code_path)
            ext = 'gif' if self._is_mc else 'png'
            fstr = 'addpath(\'{0}/{1}\');' \
                   'addpath(\'{0}/{2}\');' \
                   'build_flow(\'{3}\', \'{4}\', \'{5}\', {6}, {7}, ' \
                   '{8}, \'{3}/{9}\', {10}, {11}, 0, {12}, \'{13}\')'.format(
                        mat_code_path, 'cudafiles', 'tools',
                        svs_path_in, svs_path_out, 'svs_%04d.{}'.format(ext),
                        self.template+1,
                        1, nFrame, 'no',
                        alpha, pdm, 200, 'svs_%04d.pts'
                   )
            sys.stderr.write(fstr)
            p = matE.run_function(fstr)
            p.wait()
        else:
            svs_path_out = self._flow_path

        # Retrieve Results
        mat = sio.loadmat(
            '{}/result.mat'.format(svs_path_out)
        )

        _u, _v = mat['u'], mat['v']

        # Build Transforms
        print_dynamic("  - Build Transform")
        transforms = []
        for i in range(nFrame):
            transforms.append(
                OpticalFlowTransform(_u[:, :, i], _v[:, :, i])
            )
        self.transforms = transforms

        # build dense shapes
        print_dynamic("  - Build Dense Shapes")
        # self.reference_frame.constrain_mask_to_landmarks(group='sparse')

        # Get Dense Shape from Masked Image
        dense_reference_shape = PointCloud(
            np.vstack((
                align_t.apply(icp.target).points,
                self.reference_frame.mask.true_indices()
            ))
        )

        # Set Dense Shape as Reference Landmarks
        self.reference_frame.landmarks['source'] = dense_reference_shape
        dense_shapes = []
        for i, t in enumerate(transforms):
            warped_points = t.apply(dense_reference_shape)
            dense_shape = warped_points
            dense_shapes.append(dense_shape)

        self._dense_shapes = dense_shapes

        # build dense shape model
        dense_shape_model = super(DeformationFieldBuilder, self). \
            _build_shape_model(dense_shapes, max_components)

        self.n_landmarks = n_landmarks

        # group correlation
        if self._is_mc:
            self.group_corr = sample_groups
        else:
            self.group_corr = [range(self.n_landmarks)]

        return dense_shape_model


# Helper Functions -------------------------------
def build_reference_frame(mean_shape):
    reference_shape = mean_shape

    from menpofit.aam.base import build_reference_frame as brf

    return brf(reference_shape)


def minimum_distance(v, w, p, tolerance=1.0):
#     Return minimum distance between line segment (v,w) and point p
    l2 = dist(v, w)  # i.e. |w-v|^2 -  avoid a sqrt
    if l2 == 0.0:
        return dist(p, v)

#     Consider the line extending the segment, parameterized as v + t (w - v).
#     We find projection of point p onto the line.
#     It falls where t = [(p-v) . (w-v)] / |w-v|^2
    t = np.dot((p - v) / l2, (w - v) / l2)
    if t < 0.0:
        return dist(p, v) + tolerance      # // Beyond the 'v' end of the segment
    elif t > 1.0:
        return dist(p, w) + tolerance  # // Beyond the 'w' end of the segment

    projection = v + t * (w - v)  # // Projection falls on the segment
    return dist(p, projection)


def sample_points(target, range_x, range_y, edge=None):
    ret_img = Image.init_blank((range_x, range_y))

    if edge is None:
        edge = [range(len(target))]

    for eg in edge:
        for pts in interpolate(target[eg], 0.1):
            ret_img.pixels[0, pts[0], pts[1]] = 1

    return ret_img


def FastNICP(sources, target):
    aligned = []
    corrs = []
    for source in sources:
        mesh = TriMesh(source.points)
        a_s, corr = nicp(mesh, target, us=2001, ls=1, step=100)
        aligned.append(a_s)
        corrs.append(corr)
    return aligned, corrs