import sys, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


CLIPPING = 1E-6

input_file = sys.argv[1]
output_file = sys.argv[2]
target_line = sys.argv[3]

df = pd.read_csv(input_file).query('line==@target_line').drop(columns='line')

effect_sizes = df.transpose().iloc[0::2].values
fdrs = df.transpose().iloc[1::2].values.reshape(-1)#.clip(CLIPPING)

figure = plt.figure(figsize=(15,10))
plt.scatter(effect_sizes,-np.log(fdrs)/np.log(10),alpha=0.5,color='gray')
plt.xlabel('Feature median z-score')
plt.ylabel('-log(FDR)')
plt.title('Feature volcano for '+target_line)
plt.savefig(output_file)


