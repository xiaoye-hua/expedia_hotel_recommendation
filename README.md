# expedia_hotel_recommendation



## Run the code

```shell
export PYTHONPATH=./:PYTHONPATH

python3 scripts/reduce_file_size.py

python3 scripts/creat_offline_features.py

python3 scripts/data_cvt.py

python3 model_train.py

python3 model_predict.py

```
## TODO 
1. [ ] feature_cols should be the same for both training, eval and predict: 
    1. feature_cols = dense_features + sparse_features