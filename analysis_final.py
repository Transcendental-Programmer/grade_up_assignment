# main script for predictions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import *
from model_stuff import VoterPredictor

data_file = "data/voterfile .csv"

def run_analysis():
    print("nevada turnout prediction")
    
    df = load_data(data_file)
    target = make_target(df)
    
    df = process_voting_stuff(df)
    df = fix_demographics(df) 
    df = add_money_stuff(df)
    df = location_features(df)
    features = final_features(df)
    
    print(f"got {features.shape[1]} features")
    
    model = VoterPredictor()
    model.train_ensemble(features, target)
    
    preds, probs = model.predict_turnout(features)
    
    results = pd.DataFrame({
        'optimus_id': df['optimus_id'],
        'age': df['age'],
        'vh14p': df['vh14p'],
        'vh12g': df['vh12g'],
        'vote': preds,
        'vote_prob': probs.round(6)
    })
    
    # add the main features we care about
    key_feats = ['consistency', 'recent_score', 'socio_index', 'avg_precinct', 'party_str']
    for f in key_feats:
        results[f] = features[f]
    
    results.to_csv('voter_predictions_2014.csv', index=False)
    
    total = len(results)
    predicted = results['vote'].sum()
    rate = predicted / total
    
    print(f"total: {total:,}")
    print(f"predicted turnout: {predicted:,} ({rate:.1%})")
    
    make_plots(results, model)
    
    return results, model

def make_plots(results, model):
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(results['vote_prob'], bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('prob distribution')
    
    age_grps = pd.cut(results['age'], bins=[0, 30, 45, 60, 100], labels=['young', 'mid1', 'mid2', 'old'])
    turnout_age = results.groupby(age_grps)['vote_prob'].mean()
    axes[0, 1].bar(range(len(turnout_age)), turnout_age.values, color='green', alpha=0.7)
    axes[0, 1].set_title('turnout by age')
    axes[0, 1].set_xticks(range(len(turnout_age)))
    axes[0, 1].set_xticklabels(turnout_age.index)
    
    top_feats = model.get_top_features(8)
    feat_names = [f[0] for f in top_feats]
    feat_vals = [f[1] for f in top_feats]
    
    axes[1, 0].barh(range(len(feat_names)), feat_vals, color='orange', alpha=0.7)
    axes[1, 0].set_title('important features')
    axes[1, 0].set_yticks(range(len(feat_names)))
    axes[1, 0].set_yticklabels(feat_names)
    
    axes[1, 1].scatter(results['consistency'], results['vote_prob'], alpha=0.5, color='red')
    axes[1, 1].set_title('consistency vs prediction')
    
    plt.tight_layout()
    plt.savefig('voter_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results, model = run_analysis()
    print("done!")
