#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pandas.plotting import parallel_coordinates
import pytz, nltk
from pytz import common_timezones, all_timezones

import scipy as sp
import scipy.stats as stats
import pylab



import matplotlib
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime
import scipy as sp
matplotlib.style.use('seaborn-talk')
matplotlib.style.use('fivethirtyeight');


import scipy
import os
import plotly
plotly.offline.init_notebook_mode(connected=True)


# # Survival Analysis for Predictive Maintenance of Turbofan Engines

# Data-driven prognostics faces the perennial challenge of the lack of run-to-failure data sets. In most cases real world data contain fault signatures for a growing fault but no or little data capture fault evolution until failure. Procuring actual system fault progression data is typically time consuming and expensive. Fielded systems are, most of the time, not properly instrumented for collection of relevant data. Those fortunate enough to be able to collect long-term data for fleets of systems tend to - understandably - hold the data from public release for proprietary or competitive reasons. Few public data repositories exist that make run-to-failure data available. The lack of common data sets, which researchers can use to compare their approaches, is impeding progress in the field of prognostics. While several forecasting competitions have been held in the past, none have been conducted with a PHM-centric focus. All this provided the motivation to conduct the first PHM data challenge. The task was to estimate remaining life of an unspecified system using historical data only, irrespective of hte underlying physical process.
# 
# For most complex sysetms like aircraft engines, finding a suitable model that allows the injection of health related changes certaintly is a challenge in itself. In addition, the question of how the damage propagation should be modeled within a model needed to be addressed. Secondary issues revolved around how this propataion would be manifested in sensor signatures such that users could build meaningful prognostic solutions.
# 
# 
# n this paper we first define the prognostics problem to set the context. Then the following sections introduce the simulation model chosen, along with a brief review of health parameter modeling. This is followed by a description of the damage propagation modeling, a description of the competition data, and a discussion on performance evaluation.
# 
# * Abhinav Saxena is with Research Institute for Advanced Computer Science at NASA, Ames Research Center, Moffett Field, CA 94035 USA, phone: 650-604-3208; fax.: 650-604-4036; email: asaxena@mail.arc.nasa.gov
# * Kai Goebel is with NASA Ames Research Center
# * Don Dimon is with NASA Glenn Research Cetner
# * Neil Eklund is with GE global REsearch, Niskayuna, NY

# # Loading Data

# ```
# Data Set: FD001
# Train trjectories: 100
# Test trajectories: 100
# Conditions: ONE (Sea Level)
# Fault Modes: ONE (HPC Degradation)
# 
# Data Set: FD002
# Train trjectories: 260
# Test trajectories: 259
# Conditions: SIX 
# Fault Modes: ONE (HPC Degradation)
# 
# Data Set: FD003
# Train trjectories: 100
# Test trajectories: 100
# Conditions: ONE (Sea Level)
# Fault Modes: TWO (HPC Degradation, Fan Degradation)
# 
# Data Set: FD004
# Train trjectories: 248
# Test trajectories: 249
# Conditions: SIX 
# Fault Modes: TWO (HPC Degradation, Fan Degradation)
# 
# ```
# 
# ```
# Experimental Scenario
# 
# Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine – i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.
# 
# The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.
# 
# The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
# 1)  unit number
# 2)  time, in cycles
# 3)  operational setting 1
# 4)  operational setting 2
# 5)  operational setting 3
# 6)  sensor measurement  1
# 7)  sensor measurement  2
# ...
# 26) sensor measurement  26
# ```
# 
# ```
# Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
# ```

# In[12]:


newdir = 'CMaps/'

# define column names for easy indexing

# define column names for easy indexing
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
col_names = index_names + setting_names + sensor_names

train_fd001 = pd.read_csv(newdir + 'train_FD001.txt', sep='\s+',
                   header=None,
                names=col_names)
train_fd002 = pd.read_csv(newdir + 'train_FD002.txt', sep='\s+',
                   header=None,
                names=col_names)
train_fd003 = pd.read_csv(newdir + 'train_FD003.txt', sep='\s+',
                   header=None,
                names=col_names)
train_fd004 = pd.read_csv(newdir + 'train_FD004.txt', sep='\s+',
                   header=None,
                names=col_names)


rul_fd001 = pd.read_csv(newdir + 'RUL_FD001.txt', sep='\s+',
            header=None, names=['RUL'])
rul_fd002 = pd.read_csv(newdir + 'RUL_FD002.txt', sep='\s+',
            header=None, names=['RUL'])
rul_fd003 = pd.read_csv(newdir + 'RUL_FD003.txt', sep='\s+',
            header=None, names=['RUL'])
rul_fd004 = pd.read_csv(newdir + 'RUL_FD004.txt', sep='\s+',
            header=None, names=['RUL'])



test_fd001 = pd.read_csv(newdir + 'test_FD001.txt', sep='\s+',
                   header=None,
                names=col_names)
test_fd002 = pd.read_csv(newdir + 'test_FD002.txt', sep='\s+',
                   header=None,
                names=col_names)
test_fd003 = pd.read_csv(newdir + 'test_FD003.txt', sep='\s+',
                   header=None,
                names=col_names)
test_fd004 = pd.read_csv(newdir + 'test_FD004.txt', sep='\s+',
                   header=None,
                names=col_names)


# In[13]:


def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by='unit_nr')
    max_cycle = grouped_by_unit['time_cycles'].max()
    # merge the max cycle back into the orginal dataframe
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'),
            left_on='unit_nr',right_index=True)
    # calculate the remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame['time_cycles']
    result_frame['RUL'] = remaining_useful_life
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


# In[14]:


train_fd001 = add_remaining_useful_life(train_fd001)
train_fd002 = add_remaining_useful_life(train_fd002)
train_fd003 = add_remaining_useful_life(train_fd003)
train_fd004 = add_remaining_useful_life(train_fd004)


# # Data Preparation
# 
# Add a **breakdown** column indicating whether the engine broke down (1) or is still functioning (0):

# In[15]:


train_fd001['breakdown'] = 0
idx_last_record = train_fd001.reset_index().groupby(by='unit_nr')['index'].last()
# engines breakdown at the last cycle
train_fd001.loc[idx_last_record, 'breakdown'] = 1

train_fd002['breakdown'] = 0
idx_last_record = train_fd002.reset_index().groupby(by='unit_nr')['index'].last()
# engines breakdown at the last cycle
train_fd002.loc[idx_last_record, 'breakdown'] = 1

train_fd003['breakdown'] = 0
idx_last_record = train_fd003.reset_index().groupby(by='unit_nr')['index'].last()
# engines breakdown at the last cycle
train_fd003.loc[idx_last_record, 'breakdown'] = 1


train_fd004['breakdown'] = 0
idx_last_record = train_fd004.reset_index().groupby(by='unit_nr')['index'].last()
# engines breakdown at the last cycle
train_fd004.loc[idx_last_record, 'breakdown'] = 1


# Make sure all "birth -> death" record sets have unique identifiers

# In[16]:


train_fd001['id'] = train_fd001['unit_nr'].apply(str) + 'fd_001'
train_fd002['id'] = train_fd002['unit_nr'].apply(str) + 'fd_002'
train_fd003['id'] = train_fd003['unit_nr'].apply(str) + 'fd_003'
train_fd004['id'] = train_fd004['unit_nr'].apply(str) + 'fd_004'

test_fd001['id'] = test_fd001['unit_nr'].apply(str) + 'fd_001'
test_fd002['id'] = test_fd002['unit_nr'].apply(str) + 'fd_002'
test_fd003['id'] = test_fd003['unit_nr'].apply(str) + 'fd_003'
test_fd004['id'] = test_fd004['unit_nr'].apply(str) + 'fd_004'


# In the training set each engine is run to failure; there are no censored observations

# In[17]:


dftrain = pd.concat([train_fd001,train_fd002,
                    train_fd003,train_fd004],ignore_index=True)


# In[18]:


rul_fd001['unit_nr'] = rul_fd001.index + 1
test_fd001['RUL'] = test_fd001['unit_nr'].map(rul_fd001.set_index('unit_nr')['RUL'].to_dict())

rul_fd002['unit_nr'] = rul_fd002.index + 1
test_fd002['RUL'] = test_fd002['unit_nr'].map(rul_fd002.set_index('unit_nr')['RUL'].to_dict())

rul_fd003['unit_nr'] = rul_fd003.index + 1
test_fd003['RUL'] = test_fd003['unit_nr'].map(rul_fd003.set_index('unit_nr')['RUL'].to_dict())

rul_fd004['unit_nr'] = rul_fd004.index + 1
test_fd004['RUL'] = test_fd004['unit_nr'].map(rul_fd004.set_index('unit_nr')['RUL'].to_dict())


# In[27]:


ruls_big_list_fd001 = []
rul_max_fd001_dict = test_fd001.set_index('id')['RUL'].to_dict()
size_fd001_dict = test_fd001.groupby('id',as_index=False,sort=False).size().to_dict()
size_fd001_dict = test_fd001.groupby('id').size().to_dict()


# In[28]:


rul_max_fd001_dict['100fd_001']


# In[29]:


test_fd001.groupby('id').size().to_dict()['100fd_001']


# In[30]:


test_fd001[test_fd001['id'] == '100fd_001'][['id','RUL','time_cycles']]


# In[32]:


ruls_big_list_fd001 = []
rul_max_fd001_dict = test_fd001.set_index('id')['RUL'].to_dict()
#size_fd001_dict = test_fd001.groupby('id',as_index=False,sort=False).size().to_dict()
size_fd001_dict = test_fd001.groupby('id').size().to_dict()
for idd in test_fd001['id'].unique():
    m, s = rul_max_fd001_dict[idd], size_fd001_dict[idd]
    ruls_ = list(range(m+s-1, m-1, -1))
    assert len(ruls_) == s
    ruls_big_list_fd001.extend(ruls_)
    
    
ruls_big_list_fd002 = []
rul_max_fd002_dict = test_fd002.set_index('id')['RUL'].to_dict()
#size_fd002_dict = test_fd002.groupby('id',as_index=False,sort=False).size().to_dict()
size_fd002_dict = test_fd002.groupby('id').size().to_dict()
for idd in test_fd002['id'].unique():
    m, s = rul_max_fd002_dict[idd], size_fd002_dict[idd]
    ruls_ = list(range(m+s-1, m-1, -1))
    assert len(ruls_) == s
    ruls_big_list_fd002.extend(ruls_)
    
    
ruls_big_list_fd003 = []
rul_max_fd003_dict = test_fd003.set_index('id')['RUL'].to_dict()
#size_fd003_dict = test_fd003.groupby('id',as_index=False,sort=False).size().to_dict()
size_fd003_dict = test_fd003.groupby('id').size().to_dict()
for idd in test_fd003['id'].unique():
    m, s = rul_max_fd003_dict[idd], size_fd003_dict[idd]
    ruls_ = list(range(m+s-1, m-1, -1))
    assert len(ruls_) == s
    ruls_big_list_fd003.extend(ruls_)
    
    
    
ruls_big_list_fd004 = []
rul_max_fd004_dict = test_fd004.set_index('id')['RUL'].to_dict()
#size_fd004_dict = test_fd004.groupby('id',as_index=False,sort=False).size().to_dict()
size_fd004_dict = test_fd004.groupby('id').size().to_dict()
for idd in test_fd004['id'].unique():
    m, s = rul_max_fd004_dict[idd], size_fd004_dict[idd]
    ruls_ = list(range(m+s-1, m-1, -1))
    assert len(ruls_) == s
    ruls_big_list_fd004.extend(ruls_)


# In[33]:


test_fd001['REAL_RUL'] = ruls_big_list_fd001
test_fd002['REAL_RUL'] = ruls_big_list_fd002
test_fd003['REAL_RUL'] = ruls_big_list_fd003
test_fd004['REAL_RUL'] = ruls_big_list_fd004


# In[34]:


dftest = pd.concat([test_fd001,test_fd002,
                    test_fd003,test_fd004],ignore_index=True)


# # Save **dftrain** and **dftest** to an SQL DB

# In[35]:


dftrain.head()


# In[36]:


dftest.head()


# In[37]:


import sqlite3

with sqlite3.connect('turbofandata.db') as connection:
    dftrain.to_sql(
        'dftrain', connection, index=False,
        if_exists='replace')
    dftest.to_sql(
        'dftest', connection, index=False,
        if_exists='replace')
    


# In[38]:


with sqlite3.connect('turbofandata.db') as connection:
    dftrain_new = pd.read_sql('SELECT * from dftrain', connection)
    dftest_new = pd.read_sql('SELECT * from dftest', connection)


# In[39]:


assert dftrain_new.shape == dftrain.shape
assert dftest_new.shape == dftest.shape


# In[40]:


dftrain_new.head()


# In[41]:


assert (dftrain_new.values == dftrain.values).all()


# In[42]:


assert (dftest_new.values == dftest.values).all()


# In[ ]:




