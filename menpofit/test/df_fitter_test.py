import menpo.io as mio
from menpofit.deformationfield import DFBuilder
from menpo.feature import hog

path_to_ear_db = '/vol/atlas/homes/yz4009/ear/200EW/'
ear_training_images_r = []
ear_training_images_l = []
# load landmarked images
for i in mio.import_images(path_to_ear_db + '*_r.*',
                           verbose=True, max_images=20):
#     print i.path
    # crop image
    i.crop_to_landmarks_proportion_inplace(0.1)
    # convert it to greyscale if needed
    if i.n_channels == 3:
        i = i.as_greyscale(mode='luminosity')
    ear_training_images_r.append(i)

# load landmarked images
for i in mio.import_images(path_to_ear_db + '*_l.*',
                           verbose=True, max_images=20):
#     print i.path
    # crop image
    i.crop_to_landmarks_proportion_inplace(0.1)
    # convert it to greyscale if needed
    if i.n_channels == 3:
        i = i.as_greyscale(mode='luminosity')
    ear_training_images_l.append(i)

# build AAM
aam_r = DFBuilder(features=hog, normalization_diagonal=100)\
    .build(ear_training_images_r, verbose=True)

# build AAM
aam_l = DFBuilder(features=hog, normalization_diagonal=100)\
    .build(ear_training_images_l, verbose=True)

print aam_r

print aam_l

from menpofit.deformationfield import DFFitter

# define Lucas-Kanade based AAM fitter
fitter_r = DFFitter(aam_r, n_shape=[5, 10, 15], n_appearance=0.7)

# define Lucas-Kanade based AAM fitter
fitter_l = DFFitter(aam_l, n_shape=[5, 10, 15], n_appearance=0.7)

print fitter_r

print fitter_l

# load test images
path_to_test_set = path_to_ear_db
test_images_r = ear_training_images_r[10:]
test_images_l = ear_training_images_l[10:]

fitting_results_r = []
fitting_results_l = []
# fit images
for j, (i_r, i_l) in enumerate(zip(test_images_r, test_images_l)):
#     obtain ground truth (Xoriginal) landmarks
    gt_s_r = i_r.landmarks['PTS'].lms
    gt_s_l = i_l.landmarks['PTS'].lms

    # generate initialization landmarks
    initial_sr = fitter_r.perturb_shape(gt_s_r)
    initial_sl = fitter_l.perturb_shape(gt_s_l)

    # fit image
    frr = fitter_r.fit(i_r, initial_sr, max_iters=50, gt_shape=gt_s_r)
    frl = fitter_l.fit(i_l, initial_sl, max_iters=50, gt_shape=gt_s_l)

    # append fitting result to list
    fitting_results_r.append(frr)
    fitting_results_l.append(frl)

    print frr.errors()
    print frl.errors()
