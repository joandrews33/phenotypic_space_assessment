import sys, os
import argparse
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(sys.path[0]))
#from psa.utils import infer_compartment_from_filename, get_metadata_cols, get_ph_cols, standardize_by_kolf, standardize_by_population
from cell_profiling.utils import infer_compartment_from_filename, get_metadata_cols, get_ph_cols, standardize_by_kolf, standardize_by_population

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')
parser.add_argument('--bywell')
args = parser.parse_args()

#input_file = sys.argv[1]
#output_file = sys.argv[2]

df = pd.read_csv(args.input_file)

ph_cols = get_ph_cols(df)
if args.bywell:
    print(args.bywell.split(','))
    df_std = standardize_by_population(df,target_features=ph_cols,population_feature='Metadata_Well',control_value=args.bywell.split(','))#standardize_by_kolf(df,target_features=ph_cols)
else:
    df_std = standardize_by_kolf(df,target_features=ph_cols)

df_std.dropna(axis=1,how='any').to_csv(args.output_file,index=False)