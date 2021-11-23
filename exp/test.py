import argparse
from dataset import movielens, splitter
from model import funk_svd
from utils.timer import Timer
from utils.gen_prediction import gen_ranking_predictions
from evaluation.metrics import (df_rmse, df_mae, map_at_k, ndcg_at_k, precision_at_k,
                                                     recall_at_k, get_top_k_items)

from model.surprise.surprise_utils import predict, compute_ranking_predictions

import surprise
from sklearn.metrics import mean_absolute_error

import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #parser.add_argument("--model", help="specify the model")

    parser.add_argument("--dataset", help="specify the dataset")

    parser.add_argument("--size", help="specify the dataset size")


    args = parser.parse_args()

    MOVIELENS_DATA_SIZE = args.size


    data = movielens.load_pandas_df(
                size=MOVIELENS_DATA_SIZE,
                header=["userID", "itemID", "rating"]
            )
    train, test = splitter.python_random_split(data, 0.75)

    train_set = surprise.Dataset.load_from_df(train, reader=surprise.Reader('ml-100k')).build_full_trainset()

    svd = surprise.SVD(random_state=0, n_factors=150, n_epochs=30, verbose=True)

    with Timer() as train_time:
        svd.fit(train_set)

    print("Took {:.2f} seconds for training.".format(train_time.interval))

    #predictions = svd.predict(test)
    predictions = predict(svd, test, usercol='userID', itemcol='itemID')

    with Timer() as test_time:
        all_predictions = compute_ranking_predictions(svd, train, usercol='userID', itemcol='itemID', remove_seen=True)
    print("Took {} seconds for prediction.".format(test_time.interval))


    # evaluate the rating metric with predcions and test
    eval_rmse = df_rmse(test, predictions)
    eval_mae = df_mae(test, predictions)

    k = 10
    eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=k)
    eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=k)
    eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=k)
    eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=k)


    print("\n\n")
    print("="*30)
    print("\n")

    print("RMSE:\t\t%f" % eval_rmse,
	  "MAE:\t\t%f" % eval_mae, sep='\n')
    print('----')

    print("MAP:\t%f" % eval_map,
	  "NDCG:\t%f" % eval_ndcg,
	  "Precision@K:\t%f" % eval_precision,
	  "Recall@K:\t%f" % eval_recall, sep='\n')
