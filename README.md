# Nevada 2014 Voter Turnout Prediction
Campaign data science project for predicting November 2014 general election turnout.Deadline driven analysis for Nevada campaign operations.
## Quick Start
```bash
python analysis_final.py
```
Outputs: `voter_predictions_2014.csv` + `voter_analysis_plots.png`
## Files Structure
-`analysis_final.py` - main prediction pipeline (run this)
-`data_prep.py` - voter file preprocessing and feature engineering
-`model_stuff.py` - ensemble model training (LR + RF)
- `data/voterfile .csv` - Nevada voter registration data (50k records)
## Methodology
Ensemble approach combining calibrated logistic regression (40%) and random forest (60%).
Training on 2012 general election patterns to predict 2014 turnout.
We actually stumbled on a 2013 paper that showed logistic regression was surprisingly solid for turnout prediction, so we started there. Then, like, midway we thought, "why not ensemble?" because the forest model caught some odd precinct effects. Also, we found that boosting the Clark County weights by historical turnout (per consultant notes) helped more than we expected.
I read an old campaign analysis that recommended mixing simple and complex models seemed legit so we weighted RF a bit higher.
**Key Features:**
- Voting history consistency and recent patterns
- Demographics (age, party affiliation, education)
- Socioeconomic indicators (income, homeownership)
- Geographic effects (precinct turnout rates)
## Results Summary
-**Predicted Turnout:** 26.7% (13,366 out of 50,000 voters)
-**Model Performance:** 87.8% accuracy, 0.943 AUC
- **High Confidence Predictions:** 12,431 voters (>70% probability)

## Data Processing Notes
- Missing voting history filled with 0 (non-participation)
- Ages >100 kept as-is (potential data entry errors)
- Party strength scored: Dem/Rep=1, Non-partisan=0, Am.Independent=0.5
- Precinct effects averaged across 2008-2012 elections

Built for campaign resource allocation and voter targeting.