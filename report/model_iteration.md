# Model Iteration

## Big improvement

1. random ranker -> XGBoost model: 0.35 -> 0.49
2. XGBoost model -> LGBMRanker: 0.49 -> 0.51
3. add price listwise features and prop_id aggrated features: 0.51 -> 0.53


## Expecting big improvement but not (need to debug and re-try)

1. [ ] position bias feature in linear model -> need to be re-check
2. [ ] DeepFM model
3. [ ] historical ctr, ctcvr features


### Experiment 1218

1. Goal: the influence of position feature ridge regress
2. Setting: models with the least number of feature (refer to logs for details)
    1. 0828_ridge_v1: without `position` feature
        1. all data: 0.416; 0.415; 0.417
        2. data that previous version performed poorly: 0.388; 0.39; 0.391 
    2. 0828_ridge_v2: with `position` feature
        1. all data
            1. origianl position feature: 0.499; 0.502; 0.501
            2. setting position features: 0.414; 0.414; 0.417
        2. data that previous version performed poorly:
            1. origianl position feature: 0.386; 0.39; 0.388
            2. setting position features: 0.389; 0.39; 0.393
3. Conclusion & TODO
    1. With position feature, no matter which position that you set, the NDCG are the same
    2. [ ] With original position feature, most of the uplift comes from search lists where previous version perfomed good???


## Refer 
1. [model log excell](../model_log.xlsx)
2. [submission log in Kaggle](https://www.kaggle.com/competitions/expedia-personalized-sort/submissions)