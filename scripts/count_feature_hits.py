import sys, os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = sys.argv[1]
output_file = sys.argv[2]
alpha = np.float64(sys.argv[3])

df = pd.read_csv(input_file)

features = df.columns[1::2]

(df[df.columns[2::2]]<alpha).sum().sort_values().to_csv(output_file)