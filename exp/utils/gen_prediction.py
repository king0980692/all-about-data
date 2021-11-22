
import pandas as pd
import numpy as np
import pandas as pd


def gen_ranking_predictions(
    algo,
    data,
    usercol="u_id",
    itemcol="i_id",
    predcol="prediction",
    remove_seen=False,
):
    """Computes predictions of all users and items in data. 
    It can be used for computing ranking metrics like NDCG.

    Args:
        algo : some model class has predict function
        data (pandas.DataFrame): the data from which to get the users and items
        usercol (str): name of the user column
        itemcol (str): name of the item column
        remove_seen (bool): flag to remove (user, item) pairs seen in the training data

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    preds_lst = []
    users = data[usercol].unique()
    items = data[itemcol].unique()

    for user in users:
        for item in items:
            preds_lst.append([user, item, algo.predict_pair(user, item)])

    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])

    if remove_seen:
        tempdf = pd.concat(
            [
                data[[usercol, itemcol]],
                pd.DataFrame(
                    data=np.ones(data.shape[0]), columns=["dummycol"], index=data.index
                ),
            ],
            axis=1,
        )
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        return merged[merged["dummycol"].isnull()].drop("dummycol", axis=1)
    else:
        return all_predictions
