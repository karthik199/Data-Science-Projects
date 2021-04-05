# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:54:11 2021

@author: Karthik
"""

import pandas as pd
data = [['Alex',10],['Bob',12],['Clarke',13]]
df = pd.DataFrame(data,columns=['Name','Age'],dtype=float)
print (df)


