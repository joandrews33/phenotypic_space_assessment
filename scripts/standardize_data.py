import sys
import pandas as pd
import numpy as np
from cell_profiling.utils import infer_compartment_from_filename, get_metadata_cols, get_ph_cols, standardize_by_kolf

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)

ph_cols = get_ph_cols(df)
df_std = standardize_by_kolf(df,target_features=ph_cols)

df_std.dropna(axis=1,how='any').to_csv(output_file,index=False)