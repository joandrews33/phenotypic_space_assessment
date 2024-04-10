import pandas as pd
import sys
from cell_profiling.utils import infer_compartment_from_filename, get_metadata_cols

#TODO: Refactor this so that I can concatenate an arbitrary number of files together without appending the compartment name to features multiple times...

def df_concat(df_left: pd.DataFrame, df_right: pd.DataFrame, 
              left_compartment: str, right_compartment: str,
              index_columns = ['Metadata_Plate','Metadata_Well','Metadata_Tile','Metadata_Line','ObjectNumber']):
    
    return df_left.set_index(index_columns).rename(columns=lambda x: left_compartment+'_'+x).merge(
        df_right.set_index(index_columns).rename(columns=lambda x: right_compartment+'_'+x),
        left_index=True, right_index=True
    )
    
if __name__ == "__main__":

    input_file_1 = sys.argv[1]
    input_file_2 = sys.argv[2]
    output_file = sys.argv[3]

    left_compartment = infer_compartment_from_filename(input_file_1)
    right_compartment = infer_compartment_from_filename(input_file_2)

    metadata_columns = get_metadata_cols(pd.read_csv(input_file_1,nrows=1))

    df_concat(pd.read_csv(input_file_1),
              pd.read_csv(input_file_2),
              left_compartment=left_compartment,
              right_compartment=right_compartment,
              index_columns=metadata_columns
              ).to_csv(output_file)