import sys, os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

### Parse Inputs
parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='Input file is a volcano csv.')
parser.add_argument('output_file', help='Output file is a image file. png, svg, etc.')
parser.add_argument('target_line', help='Key to identify target line. If "average" is passed, will average the effect sizes and p-vals across lines.')
parser.add_argument('--alpha',type=np.float64, help='Target FDR for highlighting hits.')
parser.add_argument('--clipping',type=np.float64, help='Sets a minimum p-value. Prevents zero p-vals')
parser.add_argument('--angle',type=np.float64, help='Angle to display point annotations.')
parser.add_argument('--num_annotate', help='Number of top hits to annotate.', type=int)

args = parser.parse_args()

if args.angle is None:
    args.angle=15

### Load and reshape volcano dataframe.
if 'average' in args.target_line: #Passing the target line as average will average the effect sizes and p-values over the different lines. Median averaging of p-values may not be a reasonable mathematical operation.
    df = pd.read_csv(args.input_file).drop(columns='line').median(axis=0)
else:
    df = pd.read_csv(args.input_file).query('line==@args.target_line').drop(columns='line')

feature_names = df.transpose().iloc[0::2].index
effect_sizes = df.transpose().iloc[0::2].values
fdrs = df.transpose().iloc[1::2].values.reshape(-1)
if args.clipping is not None:
    fdrs = fdrs.clip(args.clipping)

df_volcano = pd.DataFrame(effect_sizes,columns=['effect_size'])
df_volcano['fdr']=fdrs
df_volcano['feature']=feature_names

### Make Figure
figure = plt.figure(figsize=(15,10))
sns.scatterplot(x=df_volcano['effect_size'],y=-np.log(df_volcano['fdr'])/np.log(10),alpha=0.5,color='gray')

### If alpha is passed, make annotated figure
if args.alpha is not None:
    q_high = 'fdr < @args.alpha & effect_size > 0'
    q_low = 'fdr < @args.alpha & effect_size < 0'

    sns.scatterplot(x=df_volcano.query(q_high)['effect_size'],y=-np.log(df_volcano.query(q_high)['fdr'])/np.log(10),alpha=0.5,color='green')
    sns.scatterplot(x=df_volcano.query(q_low)['effect_size'],y=-np.log(df_volcano.query(q_low)['fdr'])/np.log(10),alpha=0.5,color='magenta')

if args.num_annotate is not None:
    def annotate_feature_name(df,feature_name,angle=15):
        x = df.loc[feature_name]['effect_size']
        y = -np.log(df.loc[feature_name]['fdr'])/np.log(10)
        plt.text(x,y,feature_name,size='large',color='black',weight='bold',rotation=angle)

    labeled_features = df_volcano.sort_values(by='fdr',ascending=True).head(args.num_annotate)['feature'].to_list()

    for feature_name in labeled_features:
        annotate_feature_name(df_volcano.set_index('feature'),feature_name=feature_name,angle=args.angle)

plt.xlabel('Feature median z-score')
plt.ylabel('-log(FDR)')
plt.title('Feature volcano for '+args.target_line)
plt.savefig(args.output_file)


