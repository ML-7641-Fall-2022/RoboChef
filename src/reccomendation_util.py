"""
# Created by ashish1610dhiman at 03/12/22
Contact at ashish1610dhiman@gmail.com
"""

import pandas as pd

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

