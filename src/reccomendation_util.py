"""
# Created by ashish1610dhiman at 03/12/22
Contact at ashish1610dhiman@gmail.com
"""

import pandas as pd
import ast
import numpy as np
from functools import reduce
from src.utilities import *


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


def gen_recipe_list(df_meta,df_interaction,user_id,filters=[]):
    """
    args:
        df_raw --> pick recipe ids from this data frame
        filters --> All filters to apply before
    return:
        recipe set
    """
    if len(filters)>0:
        combined_filter = reduce(lambda x,y: x & y, filters)
        df_filter = df_meta.loc[combined_filter]
    else:
        df_filter = df_meta
    #Remove already interacted items
    not_interacted,interacted = _user_items(df_interaction, user_id)
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
    ranked_preds = df_rank.iloc[0].est
    return ranked_list[:min(len(ranked_list),k)],ranked_preds[:min(len(ranked_preds),k)]


def recipe_meta_map(df_meta, df_pred):
    df_subset = df_meta.merge(df_pred,left_on="id",right_on="recipe_id")
    df_subset = df_subset.sort_values(['rating_pred'],ascending=False)
    return df_subset


def match_ingredients(x,to_match, threshold = 0.25):
    x = set(x)
    to_match = set(to_match)
    match_rate = len(x.intersection(to_match))/len(to_match)
    return match_rate>=threshold


def ad_final_reccom(user_id,ingredient_list,raw_interactions,recipe_metadata,\
                    model_file="../../models/reccomender_model1_svd.pkl",k=20,\
                    extra_filters=[],remove_old=True, threshold = 0.25):
    """
    user_id -> user ID for Recommendation Engine
    ingredient_list -> List of Ingredients from CNN Output
    raw_interactions -> The raw interactions data
    recipe_metadata -> Metadata of recipes
    threshold -> Threshold for ingredients to match
    """
    #Read Model
    model = load_pickle(model_file)
    #find recipes with similar ingredients
    mask_match_ingredients = recipe_metadata["ingredients_list"]\
        .apply(lambda x: match_ingredients(x,ingredient_list, threshold))
    #filter for extra filters
    if len(extra_filters)>0:
        combined_filter_mask = reduce(lambda x, y: x & y, extra_filters)
        mask_final = mask_match_ingredients & combined_filter_mask
    else:
        mask_final = mask_match_ingredients
    recipes_to_rank = recipe_metadata.loc[mask_final]["id"].unique()
    #remove already interacted
    if remove_old:
        # Remove already interacted items
        not_interacted, interacted = _user_items(raw_interactions, user_id)
        recipes_to_rank = list(set(recipes_to_rank)-set(interacted))
    #Score recipes
    rec_ids,rec_preds = get_reccomendation_ids(model, user_id, recipes_to_rank, k)
    df_pred = pd.DataFrame.from_dict({
        "recipe_id":rec_ids,\
        "rating_pred":rec_preds
    })
    meta_sub = recipe_meta_map(recipe_metadata, df_pred)
    return (meta_sub[["name","id","rating_pred","minutes","ingredients_list","nutrition_list"]])







