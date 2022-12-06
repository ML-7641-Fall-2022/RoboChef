"""
# Created by ashish1610dhiman at 03/12/22
Contact at ashish1610dhiman@gmail.com
"""

import pandas as pd
import ast
import numpy as np
from functools import reduce


def get_embeddings(algo):
    print (f"user shape:{algo.trainset.n_users}, item shape:{algo.trainset.n_items}")
    user_matrix = algo.pu
    movie_matrix = algo.qi
    return (user_matrix,movie_matrix)

def get_raw_ids(model,type="user"):
    if type =="user":
        return (map(lambda x: model.trainset.to_raw_uid(x), model.trainset.all_users()))
    else:
        return (map(lambda x: model.trainset.to_raw_iid(x), model.trainset.all_items()))

def embed_to_df(model,embed_mat, type="user"):
    embed_df = pd.DataFrame(embed_mat)
    embed_df.columns = [f"latent_{i + 1}" for i in range(len(embed_df.columns))]
    embed_df[f"{type}_id"] = list(get_raw_ids(model,type))
    return embed_df


def _user_items(df_interactions, user_id, item_id="recipe_id"):
    mask_user = df_interactions["user_id"]==user_id
    all_items = set(df_interactions[item_id])
    all_user_items = set(df_interactions.loc[mask_user][item_id])
    return (list(all_items-all_user_items),list(all_user_items))


def get_true_ranking_user(df_test, df_interactions, df_meta, user_id, item_id="recipe_id"):
    """
    df_test --> DataFrame housing items to rank
    df_interactions --> Cal,protein Average created using this dataframe
    df_meta --> Recipe metadata
    user_id --> Gen True ranking for user
    """
    mask_user = df_interactions["user_id"] == user_id
    df_user = df_interactions.loc[mask_user]
    df_user_join = df_user.merge(df_meta, left_on=["recipe_id"], \
                                 right_on=["id"], how="left")
    df_user_join["nutrition_list"] = df_user_join["nutrition"].apply(lambda x: ast.literal_eval(x))
    df_user_join[['calories', 'fat_dv', "sugar_dv", \
                  "sodium_dv", "protein_dv", "sat_fat", "carbs_dv"] \
        ] = pd.DataFrame(df_user_join.nutrition_list.tolist(), index=df_user_join.index)
    df_user_join_subset = df_user_join[["recipe_id", "rating", "calories", "protein_dv", "fat_dv"]]
    df_user_join_subset.index = df_user_join["recipe_id"]
    df_user_join_subset = df_user_join_subset.drop(columns=["recipe_id"])
    user_delta = np.abs(df_user_join_subset - df_user_join_subset.mean()) #TODO
    user_delta = user_delta.reset_index()
    df_user_join_subset_delta = df_user_join_subset.merge(user_delta, on="recipe_id", suffixes=("", "_delta"))
    #subset for only test recipes
    recipe_test = set(df_test.loc[df_test["user_id"] == user_id]["recipe_id"])
    df_user_join_subset_delta.reset_index()
    # print (df_user_join_subset_delta.columns)
    df_user_join_subset_delta = df_user_join_subset_delta[df_user_join_subset_delta["recipe_id"].isin(recipe_test)]
    df_user_join_subset_delta = df_user_join_subset_delta.sort_values(by=["rating", "calories_delta", \
                                                                          "protein_dv_delta", "fat_dv_delta"], \
                                                                      ascending=[False, True, True, True])
    return (list(df_user_join_subset_delta.recipe_id))


def get_prediction_ranking(df_pred):
    df_pred = df_pred.sort_values(by=["uid", "est"], ascending=[True, False])
    df_rank = df_pred.groupby(['uid']).agg({'iid': lambda x: list(x), \
                                         'est': lambda x: list(x)}).reset_index()
    return df_rank

def get_true_ranking_user_baseline(df_test, item_id="recipe_id"):
    """
    df_test --> DataFrame housing items to rank
    """
    df_pred = df_test.sort_values(by=["user_id", "rating"], ascending=[True, False])
    df_rank = df_pred.groupby(['user_id']).agg({'recipe_id': lambda x: list(x), \
                                            'rating': lambda x: list(x)}).reset_index()
    return df_rank


def gen_recipe_list(df_raw,user_id,filters=[]):
    """
    args:
        df_raw --> pick recipe ids from this data frame
        filters --> All filters to apply before
    return:
        recipe set
    """
    if len(filters)>0:
        combined_filter = reduce(lambda x,y: x & y, filters)
        df_filter = df_raw.loc[combined_filter]
    else:
        df_filter = df_raw
    #Remove already interacted items
    not_interacted,interacted = _user_items(df_raw, user_id)
    mask_not_interacted = df_filter["recipe_id"].isin(not_interacted)
    return (df_filter.loc[mask_not_interacted]["recipe_id"].unique())

def get_reccomendation_ids(model, user_id, recipe_ids, k=10000):
    """
    user_id --> user_id to score
    recipe_ids --> recipes to rank
    k --> reccomend top k
    """
    test_set = [[user_id,recipe_id,None]\
                for recipe_id in recipe_ids]
    predictions = model.test(test_set)
    df_pred = pd.DataFrame(predictions)
    df_rank = get_prediction_ranking(df_pred)
    ranked_list = df_rank.iloc[0].iid
    return ranked_list[:min(len(ranked_list),k)]


def recipe_meta_map(df_meta, recipe_ids):
    df_subset = df_meta.loc[df_meta["id"].isin(recipe_ids)]
    cat_recipe = pd.CategoricalDtype(
        recipe_ids,
        ordered=True)
    df_subset['recipe_id'] = df_subset['id'].astype(cat_recipe)
    df_subset = df_subset.sort_values(['id'])
    return df_subset






