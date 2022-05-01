# -*- coding: utf-8 -*-
# @File    : save_submission.py
# @Author  : Hua Guo
# @Disc    :
import os
from scripts.train_config import result_dir
from src.config import submission_cols


def save_submission(rec_df, file_name: str,
                    # test_df

                    ):
    # result_df = test_df[['srch_id', 'prop_id']].rename(columns=dict(zip(['srch_id', 'prop_id'], submission_cols)))
    # result_df = result_df.merge(rec_df, how='left', on=submission_cols)
    result_df = rec_df
    # assert result_df.shape[0] == rec_df.shape[0]
    result_df = result_df.sort_values(['SearchId', "predicted"], ascending=[True, False])
    file_name = os.path.join(result_dir, file_name)
    result_df[submission_cols].to_csv(file_name, index=False)
    print(f"shape: {result_df[submission_cols].shape}")
    print(result_df.head())
    print(result_df[submission_cols].head())
