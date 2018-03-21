# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:13:27 2018

@author: user
"""

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head(20)
test.head()
