
# coding: utf-8

## AAM vs DF

#### Defining Useful helper functions

# In[1]:

import menpo.io as mio
import matplotlib.pyplot as plt
from IPython.html.widgets import interact
from menpo.visualize import visualize_images
import numpy as np


#### Load Training Ear Images

# In[2]:

path_to_ear_db = '/vol/atlas/homes/yz4009/ear/200EW/'
ear_training_images_r = []
max_img = 100
training_part = max_img - max_img / 5
# load landmarked images
for i in mio.import_images(path_to_ear_db + '*_r.*', verbose=True, max_images=max_img):
#     print i.path
    # crop image
    i.crop_to_landmarks_proportion_inplace(0.1)
    # convert it to greyscale if needed
    if i.n_channels == 3:
        i = i.as_greyscale(mode='luminosity')
    ear_training_images_r.append(i)




#### Build LinearGlobelAAM

# In[4]:

from menpofit.deformationfield import DFBuilder
from menpofit.aam import AAMBuilder
from menpo.feature import igo, hog


# In[5]:

# build AAM
aam_r = AAMBuilder(normalization_diagonal=100, n_levels=3, features=hog).build(ear_training_images_r[:training_part], verbose=True)

# build DF
df_r = DFBuilder(normalization_diagonal=100, n_levels=3, features=hog).build(ear_training_images_r[:training_part], verbose=True)


# In[6]:

print aam_r


# In[7]:

print df_r


##### Visualise AAM



### Fit LinearGlobelAAM

# In[11]:

from menpofit.aam import LucasKanadeAAMFitter
from menpofit.deformationfield import DFFitter

# define Lucas-Kanade based AAM fitter
aam_fitter_r = LucasKanadeAAMFitter(aam_r, n_shape=[5,10,25], n_appearance=0.7)

# define Lucas-Kanade based AAM fitter
df_fitter_r = DFFitter(df_r, n_shape=[5,10,25], n_appearance=0.7)


# In[12]:

print aam_fitter_r


# In[13]:

print df_fitter_r


# In[ ]:

# load test images
path_to_test_set='/vol/atlas/homes/yz4009/ear/200EW/'
path_to_test_set=path_to_ear_db
test_images_r = ear_training_images_r[training_part:]


# In[ ]:

aam_fitting_results_r = []
df_fitting_results_r = []
# fit images
for j, i_r in enumerate(test_images_r):
    # obtain groubnd truth (original) landmarks
    gt_s_r = i_r.landmarks['PTS'].lms
    
    # generate initialization landmarks
    initial_sr = aam_fitter_r.perturb_shape(gt_s_r)
    initial_sr.view()

    # fit image
    aam_frr = aam_fitter_r.fit(i_r, initial_sr, max_iters=50, gt_shape=gt_s_r)
    df_frr = df_fitter_r.fit(i_r, initial_sr, max_iters=50, gt_shape=gt_s_r)
    
    # append fitting result to list
    aam_fitting_results_r.append(aam_frr)
    df_fitting_results_r.append(df_frr)
    
    print j
    print aam_frr
    print df_frr


#### Visualise Fitting Results



#### DF vs AAM Results

# In[ ]:

from menpofit.visualize import plot_ced


# In[ ]:

plot_ced([
    [fr.initial_error() for fr in aam_fitting_results_r], 
    [fr.initial_error() for fr in df_fitting_results_r],
    [fr.errors()[-1] for fr in aam_fitting_results_r], 
    [fr.errors()[-1] for fr in df_fitting_results_r]
])


# In[ ]:



PointCloud(df_fitter_r._fitters[0].transform.from_vector(df_frr.fitting_results[0].parameters[0]).target.points[:55]).view()