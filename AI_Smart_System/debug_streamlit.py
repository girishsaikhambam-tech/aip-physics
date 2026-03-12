import pandas as pd
from io import BytesIO
import environment_monitor as em

import os
base = os.path.dirname(__file__)
path = os.path.join(base, 'datasets', 'environment_data.csv')
with open(path,'rb') as f:
    data = f.read()

buf = BytesIO(data)
df = pd.read_csv(buf)
print('columns', df.columns)
print(df.head())

# test loader function
buf2 = BytesIO(data)
df2 = em.load_environment_data(buf2)
print('loaded by function')
print(df2.head())
