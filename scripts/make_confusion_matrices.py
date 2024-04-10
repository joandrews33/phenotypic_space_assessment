import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cell_profiling.utils import get_ph_cols
from cell_profiling.models import evaluate_GMM_classification_accuracy

input_file =sys.argv[1]

output_cells = '.'.join(input_file.split('.')[:-1])+'_class_probabilities'
output_class = '.'.join(input_file.split('.')[:-1])+'_class_confusions'

pop_id_column = 'Metadata_Line'

df = pd.read_csv(input_file)
ph_cols = get_ph_cols(df)
ph_cols = df[ph_cols].dropna(axis=1,how='any').columns

_, posterior_matrix, class_ids = evaluate_GMM_classification_accuracy(df[ph_cols].values,df[pop_id_column])

df_predictions = pd.DataFrame(posterior_matrix,columns=class_ids).rename(columns = lambda x: 'p_'+x)
df_predictions['True_Class'] = df[pop_id_column]
df_predictions['Predicted_Class'] = [class_ids[n] for n in np.argmax(posterior_matrix,axis=1)]
df_predictions.to_csv(output_cells+'.csv')

figure =plt.figure(figsize=(20,40))
sns.heatmap(df_predictions.drop(columns=['Predicted_Class','True_Class']))
plt.savefig(output_cells+'.png')


df_class_confusions = df_predictions.drop(columns='Predicted_Class').groupby('True_Class').mean().sort_index(axis=1)
df_class_confusions.to_csv(output_class+'.csv')

figure = plt.figure(figsize=(20,20))
sns.heatmap(df_class_confusions)
plt.savefig(output_class+'.png')
plt.savefig(output_class+'.svg')
