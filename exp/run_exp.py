from dataset import movielens, splitter
from model import funk_svd
from utils.timer import Timer
from utils.gen_prediction import gen_ranking_predictions
from evaluation.metrics import (df_rmse, df_mae, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)


from sklearn.metrics import mean_absolute_error

import pandas as pd

if __name__ =='__main__':

    MOVIELENS_DATA_SIZE = '100k'

    data = movielens.load_pandas_df(
                size=MOVIELENS_DATA_SIZE,
                header=["u_id", "i_id", "rating"]
                #header=["userID", "itemID", "rating"]
            )

    train, test = splitter.python_random_split(data, 0.75)
    #val = data.drop(train.index.tolist()).sample(frac=0.5, random_state=8)


    svd = funk_svd.SVD(lr=0.001, reg=0.005, n_epochs=30, n_factors=200,
                early_stopping=True, shuffle=False, min_rating=1, max_rating=5)

    with Timer() as train_time:
        #svd.fit(X=train, X_val=val)
        svd.fit(X=train)
    print("Took {:.2f} seconds for training.".format(train_time.interval))

    predictions = svd.predict(test)


    with Timer() as test_time:
        all_predictions = gen_ranking_predictions(svd, train, usercol='u_id', itemcol='i_id', remove_seen=True)
                
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
