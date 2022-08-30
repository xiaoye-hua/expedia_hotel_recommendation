# Model Iteration

## Big improvement

1. random ranker -> XGBoost model: 0.35 -> 0.49
2. XGBoost model -> LGBMRanker: 0.49 -> 0.51
3. add price listwise features and prop_id aggrated features: 0.51 -> 0.53


## Expecting big improvement but not (need to debug and re-try)

1. [ ] position bias feature in linear model
2. [ ] DeepFM model
3. [ ] historical ctr, ctcvr features


## Refer 
1. [model log excell](../model_log.xlsx)
2. [submission log in Kaggle](https://www.kaggle.com/competitions/expedia-personalized-sort/submissions)