import sys
import pandas as pd
import numpy as np
from cell_profiling.utils import infer_compartment_from_filename, get_metadata_cols, get_ph_cols, standardize_by_kolf
from cell_profiling.models import evaluate_knn_purity, evaluate_knn_classification_accuracy,evaluate_GMM_classification_accuracy

input_file = sys.argv[1]
output_file = sys.argv[2]

if len(sys.argv)>3:
    pop_id_column = sys.argv[3]
else:
    pop_id_column = 'Metadata_Line'

if len(sys.argv)>4:
    k=sys.argv[4]
else:
    k=5

df = pd.read_csv(input_file).replace([np.inf, -np.inf], np.nan).dropna(axis=1,how='any')

ph_cols = get_ph_cols(df)
ph_cols = df[ph_cols].select_dtypes(exclude=['object']).dropna(axis=1,how='any').columns

raw_knn_purity, _, _ = evaluate_knn_purity(df[ph_cols],df[pop_id_column],k=k)

raw_knn_accuracy = evaluate_knn_classification_accuracy(df[ph_cols].values,df[pop_id_column].to_list(),k=k)

raw_GMM_accuracy, _, _ = evaluate_GMM_classification_accuracy(df[ph_cols].values,df[pop_id_column])

df_results = pd.DataFrame(data = np.array([[raw_knn_purity,raw_knn_accuracy,raw_GMM_accuracy]]),
              columns=['KNN_purity','KNN_classification_accuracy','GMM_classification_accuracy']
              ).to_csv(output_file,index=False)