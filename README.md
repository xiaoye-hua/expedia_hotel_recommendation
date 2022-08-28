# expedia_hotel_recommendation

## Run the code

```shell
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name expedia_project --display-name "Python3.8(expedia_project)"

export PYTHONPATH=./:PYTHONPATH


# 1. reduce file size; 2. save data to pkl; 3. save big_data & small_data version of training data
python3 scripts/reduce_file_size.py
# offline feature only for big_data
python3 scripts/create_offline_features.py

python3 scripts/data_cvt.py

python3 scripts/model_train.py

python3 model_predict.py

```
## TODO 
1. [ ] feature_cols should be the same for both training, eval and predict: 
    1. feature_cols = dense_features + sparse_features
