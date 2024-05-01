import sys, os
import argparse
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(sys.path[0]))

#from psa.utils import get_ph_cols 
from cell_profiling.utils import get_ph_cols
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')
parser.add_argument('--population_feature',help='feature used to group data into populations. default=Metadata_Line')
parser.add_argument('--target_population',help='default population used for null model. default=KOLF')
parser.add_argument('--use_revertants',action='store_true',help='use revertants as the null model for p-values instead of default.')
args = parser.parse_args()

#input_file = sys.argv[1]
#output_file = sys.argv[2]
#use_revertants=True

if args.population_feature is None:
    POPULATION = 'Metadata_Line'
else:
    POPULATION = args.population_feature

if args.target_population is None:
    default_background = 'KOLF'
else:
    if args.use_revertants:
        import warnings
        warnings.warn('Using revertants only makes sense if you are standardizing by the cell line.')
    default_background = args.target_population

df_std = pd.read_csv(args.input_file)
ph_cols = get_ph_cols(df_std)

def set_background(populations,use_revertants=True,default_background='KOLF'):
    
    if not use_revertants: #Return default background
        return [default_background]*len(populations)
    else: #Check if the Revertant is in the set of populations. 
        backgrounds = []
        for population in populations:
            if population[:-3]+'Rev' in populations:
                if 'Rev' in population[-3:]: #Use KOLF as the background for the Revertant lines. 
                    backgrounds.append(default_background)
                else:
                    backgrounds.append(population[:-3]+'Rev')
            else:
                backgrounds.append(default_background)
        return backgrounds

def get_pvals(df_pop,df_null,target_features):
    p_vec = mannwhitneyu(df_pop[target_features].values,df_null[target_features].values,axis=0).pvalue
    return p_vec

df = []
for population, background in zip(df_std[POPULATION].unique(),set_background(df_std[POPULATION].unique(),use_revertants=args.use_revertants,default_background=default_background)):
    print(population)
    target_plate = df_std.query(POPULATION+'==@population').Metadata_Plate.unique()#[0]
    print(target_plate)
    #p_vec = get_pvals(df_std.query(POPULATION+'==@population'),df_std.query(POPULATION+'=="KOLF"').query('Metadata_Plate in @target_plate'),target_features=ph_cols)
    p_vec = get_pvals(df_std.query(POPULATION+'==@population'),df_std.query(POPULATION+'==@background').query('Metadata_Plate in @target_plate'),target_features=ph_cols)
    _,fdr=fdrcorrection(p_vec)
    effect_vec = df_std.query(POPULATION+'==@population')[ph_cols].median().values
    df.append(pd.concat([pd.DataFrame(effect_vec.reshape(1,-1),columns=ph_cols),
        pd.DataFrame(fdr.reshape(1,-1),columns=ph_cols).rename(columns=lambda x: 'fdr_'+x).assign(line=population).assign(background=background)],axis=1))
        #pd.DataFrame(fdr.reshape(1,-1),columns=ph_cols).rename(columns=lambda x: 'fdr_'+x).assign(line=population)],axis=1))
        #pd.DataFrame(p_vec.reshape(1,-1),columns=ph_cols).rename(columns=lambda x: 'p_'+x).assign(line=population)],axis=1))
df = pd.concat(df,axis=0)

column_order = ['line','background']
for feature in ph_cols:
    column_order.append(feature)
    column_order.append('fdr_'+feature)

df[column_order].to_csv(args.output_file,index=False)



