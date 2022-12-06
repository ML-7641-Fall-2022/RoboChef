"""
# Created by ashish1610dhiman at 03/12/22
Contact at ashish1610dhiman@gmail.com
"""

import pandas as pd
import ast
import numpy as np

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


def get_true_ranking_user(df_interactions, df_meta, user_id, item_id="recipe_id"):
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
    user_delta = np.abs(df_user_join_subset - df_user_join_subset.mean())
    user_delta.reset_index()
    df_user_join_subset_delta = df_user_join_subset.merge(user_delta, on="recipe_id", suffixes=("", "_delta"))
    df_user_join_subset_delta = df_user_join_subset_delta.sort_values(by=["rating", "calories_delta", \
                                                                          "protein_dv_delta", "fat_dv_delta"], \
                                                                      ascending=[False, True, True, True])
    return (df_user_join_subset_delta)



