import sys, os
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(sys.path[0]))

#from psa.utils import get_ph_cols 
from cell_profiling.utils import get_ph_cols
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

input_file = sys.argv[1]
output_file = sys.argv[2]

POPULATION = 'Metadata_Line'

df_std = pd.read_csv(input_file)
ph_cols = get_ph_cols(df_std)

def get_pvals(df_pop,df_null,target_features):
    p_vec = mannwhitneyu(df_pop[target_features].values,df_null[target_features].values,axis=0).pvalue
    return p_vec

df = []
for population in df_std[POPULATION].unique():
    print(population)
    target_plate = df_std.query(POPULATION+'==@population').Metadata_Plate.unique()[0]
    p_vec = get_pvals(df_std.query(POPULATION+'==@population'),df_std.query(POPULATION+'=="KOLF"').query('Metadata_Plate in @target_plate'),target_features=ph_cols)
    _,fdr=fdrcorrection(p_vec)
    effect_vec = df_std.query(POPULATION+'==@population')[ph_cols].median().values
    df.append(pd.concat([pd.DataFrame(effect_vec.reshape(1,-1),columns=ph_cols),
        pd.DataFrame(fdr.reshape(1,-1),columns=ph_cols).rename(columns=lambda x: 'fdr_'+x).assign(line=population)],axis=1))
        #pd.DataFrame(p_vec.reshape(1,-1),columns=ph_cols).rename(columns=lambda x: 'p_'+x).assign(line=population)],axis=1))
df = pd.concat(df,axis=0)

column_order = ['line']
for feature in ph_cols:
    column_order.append(feature)
    column_order.append('fdr_'+feature)

df[column_order].to_csv(output_file,index=False)



#MAKE VOLCANOEESESEES
