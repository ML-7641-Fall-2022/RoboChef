"""
# Created by ashish1610dhiman at 03/12/22
Contact at ashish1610dhiman@gmail.com
"""
import pickle


def ad_train_test_split(df, test_size):
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    user_normal_list = []
    #     user_single = []

    users_unique = df["user_id"].unique()
    for user in tqdm(users_unique):
        mask_user = df["user_id"] == user
        n_interactions_user = mask_user.sum()
        df_user = df.loc[mask_user]
        if n_interactions_user < int(1 / test_size):
            df_user = df_user.sample(frac=1)  # shuffle rows
            if n_interactions_user > 1:
                df_train_user = df_user.iloc[:-1]
                df_test_user = df_user.iloc[-1]
            else:  # if  only one interaction pull in train
                user_normal_list.append(user)
        else:  # if obs>= split size
            user_normal_list.append(user)

    # split normal by sklearn
    df_normal = df[df["user_id"].isin(user_normal_list)]
    df_train_normal, df_test_normal = train_test_split(df_normal, test_size=TEST_SIZE)

    # append all
    df_train = pd.concat([df_train, df_train_normal])
    df_test = pd.concat([df_test, df_test_normal])
    return (df_train, df_test)

    # append all
    df_train = pd.concat([df_train, df_train_normal, df_train_single])
    df_test = pd.concat([df_test, df_test_normal, df_test_single])
    return (df_train, df_test)

def write_pickle(x, filename):
    filehandler = open(filename, 'wb')
    pickle.dump(x, filehandler)
    # print ("Pickle Dump Falied")


def load_pickle(filename):
    file = open(filename, 'rb')
    res = pickle.load(file)
    return (res)