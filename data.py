import numpy as np
import pandas as pd

DATA_SOURCE = "./data/score_0nan_small.csv"
score = pd.read_csv(DATA_SOURCE)
names = score.drop(["学年", "性別"], axis=1).columns


def get_num_data():
    tmp = score
    # 任意の行をとる
    # delete = teams - rows
    rows = ["学年", "性別"]
    tmp = tmp.drop(rows, axis=1)
    return tmp


def get_full_data():
    return score


def get_corrcoef(data, x_label, y_label):
    cor = np.corrcoef(data[x_label], data[y_label])
    return cor[0, 1].round(4)


def pick_up_df(df, genre):
    ans = pd.DataFrame()

    for elem in genre:
        grade = elem[0:2]
        gender = elem[2]
        ans = ans.append(df[(df["学年"] == grade) & (df["性別"] == gender)])


# ジャンルに応じてデータをフィルタリングして返す
def load_filtered_data(data, genre_filter):
    if genre_filter == "女子":
        filtered_data = data[data["性別"].isin(["女"])]
    elif genre_filter == "高1女子":
        filtered_data = data[data["性別"].isin(["女"])]
        filtered_data = filtered_data[filtered_data["学年"].isin(["高1"])]
    else:
        filtered_data = data

    return filtered_data
