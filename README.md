The Turbofan data from the 


1. run data_prep.ipynb; outputs turbofandata.db
2. run make_model.ipynb; outputs scaler_xtrain.pkl and model.h5, and adds dfval to turbofandata.db


3. run make_modelnotime.ipynb ; does not use time as one of the input features (more realistic case): outputs scaler_xtrain_notime.pkl and model_notime.h5 and adds dfval_notime to turbofandata.db



4. run survival_curves_animation.ipynb; creates/overwrites the folder ANIMATEDSURVIVALCURVES and populates it with interactive .html plotly charts of the validation set's predicted survival curves; .csv files of the survival curve data, and .mp4 files with animations. The naming convention is such that the unique identifier of each validation engine is the file name, and then the appropriate extension. Also produces parallel_rul.html




**writeup.ipynb** and writeup.html contain a high level overview of the project.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/doolingdavid/hyper/HEAD)