# voter data processing stuff
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

thresh = 0.52
clark_boost = 1.15
high_income = 75000

def load_data(path):
    data = pd.read_csv(path)
    print(f"got {len(data)} voters")
    return data

def make_target(df):
    df['target_vote'] = df['vh12g'].fillna(0)
    return df['target_vote']

def process_voting_stuff(data):
    vote_cols = ['vh12p', 'vh10g', 'vh10p', 'vh08g', 'vh08p', 'vh06g', 'vh06p', 'vh04g', 'vh04p', 'vh02g', 'vh02p', 'vh00g', 'vh00p']
    
    for c in vote_cols:
        data[c] = data[c].fillna(0)
    
    data['consistency'] = data[vote_cols].sum(axis=1) / len(vote_cols)
    
    # recent voting matters way more than old stuff, like 2000 elections dont really predict 2014
    data['recent_score'] = data['vh12p'] * 0.4 + data['vh10g'] * 0.3 + data['vh10p'] * 0.2 + data['vh08g'] * 0.1
    
    generals = ['vh10g', 'vh08g', 'vh06g', 'vh04g', 'vh02g', 'vh00g']
    primaries = ['vh12p', 'vh10p', 'vh08p', 'vh06p', 'vh04p', 'vh02p', 'vh00p']
    
    data['gen_rate'] = data[generals].sum(axis=1) / len(generals)
    data['prim_rate'] = data[primaries].sum(axis=1) / len(primaries)
    data['likes_general'] = (data['gen_rate'] > data['prim_rate']).astype(int)
    
    data['streak'] = 0
    for i, row in data.iterrows():
        s = 0
        for col in ['vh12p', 'vh10g', 'vh10p', 'vh08g']:
            if row[col] == 1:
                s += 1
            else:
                break
        data.at[i, 'streak'] = s
    
    return data

def fix_demographics(df):
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['age_grp'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=['young', 'mid1', 'mid2', 'old', 'vold'])
    
    # party strength is like super important for turnout prediction
    party_str = {'Democratic': 1, 'Republican': 1, 'Non-Partisan': 0, 'American Independent': 0.5}
    df['party_str'] = df['party'].map(party_str).fillna(0)
    
    edu_scores = {'nan': 0, 'Unknown': 0, 'HS Diploma - Likely': 1, 'HS Diploma - Extremely Likely': 1,
                 'Some College - Likely': 2, 'Some College - Extremely Likely': 2,
                 'Bach Degree - Likely': 3, 'Bach Degree - Extremely Likely': 3,
                 'Grad Degree - Likely': 4, 'Grad Degree - Extremely Likely': 4}
    df['edu_score'] = df['education'].astype(str).map(edu_scores).fillna(0)
    
    return df

def add_money_stuff(data):
    inc_map = {'Unknown': 0, '0-35k': 1, '35k-75k': 2, '75k-125k': 3, '125k-200k': 4, '200k+': 5}
    data['inc_score'] = data['income'].astype(str).map(inc_map).fillna(0)
    
    worth_map = {'nan': 0, '$100000-249999': 2, '$250000-499999': 3, '$500000+': 4}
    data['worth_score'] = data['net_worth'].astype(str).map(worth_map).fillna(0)
    
    data['homeowner'] = (data['home_owner_or_renter'] == 'Likely Homeowner').astype(int)
    
    data['socio_index'] = data['edu_score'] * 0.4 + data['inc_score'] * 0.3 + data['worth_score'] * 0.2 + data['homeowner'] * 0.1
    
    return data

def location_features(df):
    precinct_turnouts = ['g08_precinct_turnout', 'g10_precinct_turnout', 'g12_precinct_turnout']
    df['avg_precinct'] = df[precinct_turnouts].mean(axis=1)
    df['precinct_trend'] = df['g12_precinct_turnout'] - df['g08_precinct_turnout']
    
    vegas_reno = ['LAS VEGAS DMA (EST.)', 'RENO DMA (EST.)']
    df['urban'] = df['dma'].isin(vegas_reno).astype(int)
    
    return df

def final_features(data):
    cols = ['age', 'party_str', 'edu_score', 'inc_score', 'socio_index', 'homeowner', 'urban',
           'consistency', 'recent_score', 'gen_rate', 'streak', 'likes_general', 'avg_precinct', 
           'precinct_trend', 'vh10g', 'vh08g', 'vh12p']
    
    feats = data[cols].copy()
    feats = feats.fillna(feats.median())
    
    return feats
