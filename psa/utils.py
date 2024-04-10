import sys
import pandas as pd
import numpy as np
from itertools import product
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from scipy.stats import multivariate_normal
import umap

def get_metadata_cols(df):
    '''Extracting columns from df that are metadata. Includes ImageNumber, ObjectNumber and every column that starts with Metadata_.'''
    metadata_columns = ['ImageNumber','ObjectNumber']+[col for col in df.dropna(axis=1,how='all').columns if 'Metadata' in col.split('_'[0])]
    return metadata_columns

def get_ph_cols(df):
    '''Extracts columns that are measured phenotypes. 
    Assumes that every column that isn't metadata is for phenotyping.
    Deletes empty columns.'''

    metadata_columns = get_metadata_cols(df)
    ph_columns =  [col for col in df.dropna(axis=1,how='all').columns if col not in metadata_columns]
    return ph_columns

def infer_compartment_from_filename(filename):
    return filename.split('_')[-1].split('.')[0]

def evaluate_mad(df,df_medians):
    '''df_medians is assumed to have one value per column.'''
    return df.subtract(df_medians.values).abs().median()

def standardize_by_kolf(df,population_feature='Metadata_Line',control_value='KOLF',group_columns=['Metadata_Plate','Metadata_Well','Metadata_Tile','Metadata_Line'], index_columns = ['Metadata_Tile','ObjectNumber'],target_features=None):
    '''Standardizes the numerical columns of df by evaluating the robust z-score. The null model for each
    measurement is estimated as its empirical distribution for the null_gene. If group_column is specified, the 
    null model is evaluated separately for each category in group_column. (E.g., standardizing by well.)'''

    #Warning, this will fail is dataframe contains repeated values for cells
    
    df_out = df.copy()

    if target_features is None:
        target_features = [col for col in df.columns if col not in group_columns+index_columns]
    
    group_medians = (df.query(population_feature+'==@control_value')
                     .groupby(group_columns)[target_features].median()
                     .groupby(population_feature).median()) #Since these are not in well controls, I am taking the median over all fields of view to get one value per feature from KOLF.
    group_mads = evaluate_mad(df.query(population_feature+'==@control_value')[target_features],group_medians)

    df_out[target_features] = df[target_features].subtract(group_medians.values).divide(group_mads).multiply(0.6745)

    return df_out#.reset_index()

def average_per_tile(df,target_columns):
    return df.groupby(['Metadata_Line','Metadata_Plate','Metadata_Well','Metadata_Tile'])[target_columns].median().reset_index()

def average_per_well(df,target_columns):
    '''I average first by FOVs, then by well.'''
    return df.groupby(['Metadata_Line','Metadata_Plate','Metadata_Well','Metadata_Tile'])[target_columns].median().groupby(['Metadata_Line','Metadata_Plate','Metadata_Well']).median().reset_index()

def average_per_line(df,target_columns):
    '''I average first by FOVs, then by Line.'''
    return df.groupby(['Metadata_Plate','Metadata_Well','Metadata_Tile','Metadata_Line'])[target_columns].median().groupby('Metadata_Line').median().reset_index()

def prioritize_features(df,cutoff=0.9):
    sorted_features = df.std().sort_values(ascending=False).dropna().index.to_list()
    excluded_features = set()
    for ii in range(len(sorted_features)):
        if sorted_features[ii] in excluded_features:
            pass
        new_exclusions = set((df[sorted_features[ii+1:]].columns[df[sorted_features[ii+1:]].corrwith(df[sorted_features[ii]]).abs()>cutoff]).tolist())
        excluded_features=excluded_features.union(new_exclusions)
    retained_features = [f for f in sorted_features if f not in list(excluded_features)]
    return retained_features

