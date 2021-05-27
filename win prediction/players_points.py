import numpy as np
import pandas as pd

BBB = pd.read_csv("/home/prem/ipl_data/data1_ball_to_ball.csv")
temp9 = pd.read_csv("/home/prem/points_update.csv")
PLAYERS = pd.read_csv("/home/prem/PycharmProjects/SSSIHL-IPL/datasets/PLAYERS.csv")


def bowler_stats(data):
    # No of wickets
    data1 = data[data['dismissal_kind'] != 'run out']
    wkts = np.sum(data1["is_wicket"])

    # No of Dots
    dots = len(data[data['total_runs'] == 0])

    return pd.Series([wkts, dots], index=['Wickets', 'Dots'])


temp1 = BBB.groupby("bowler").apply(bowler_stats).reset_index().sort_values(by="Wickets", ascending=False)
temp1 = pd.DataFrame(temp1)
temp1 = temp1.rename(columns={'bowler': 'player'})


def batsman_stats(x):
    # 6's
    six = 0
    four = 0
    for b in x["batsman_runs"]:
        if b == 6:
            six = six + 1
        elif b == 4:
            four = four + 1

    return pd.Series([six, four], index=["6's", "4's"])


temp2 = BBB.groupby(["batsman"]).apply(batsman_stats).reset_index()
temp2 = pd.DataFrame(temp2).rename(columns={'batsman': 'player'})

# Calculating No of catches taken by each player
temp3 = BBB.groupby('fielder').apply(lambda a: len(a[a["dismissal_kind"] == "caught"])).reset_index(name='catches')
temp3 = pd.DataFrame(temp3).rename(columns={'fielder': 'player'})

temp3 = temp3[temp3['catches'] != 0]

# Calculating No of Stumpings done by each player
temp4 = BBB.groupby('fielder').apply(lambda a: len(a[a["dismissal_kind"] == "stumped"])).reset_index(name='stumpings')
temp4 = pd.DataFrame(temp4).rename(columns={'fielder': 'player'})
temp4 = temp4[temp4['stumpings'] != 0]

# Merging all the attributes required for calculating points of each player
'''DATA = pd.merge(temp1, temp2,on='player',how = 'outer')
DATA
#DATA = pd.merge(DATA, temp3,on='player',how = 'outer')
DATA = pd.merge(DATA, temp4,on='player',how = 'outer')
DATA.fillna(0)

#DATA["stumpings"].replace(np.nan,0)
#DATA.fillna(0)
#DATA.to_csv("updated_players1")
#len(DATA["player"].unique()) '''


def funct1(data):
    wkts = np.sum(data["Wickets"])
    dots = np.sum(data["Dots"])
    sixes = np.sum(data["6's"])
    fours = np.sum(data["4's"])
    updated_catches = np.sum(data["catches"])
    stumps = np.sum(data["stumpings"])

    return pd.Series([wkts, dots, sixes, fours, updated_catches, stumps],
                     index=["Wkts", "Dots", "6's", "4's", "catches", "stumpings"])

# Calculating the points for each player
'''
temp = temp9.drop("player", axis=1)
temp = temp.groupby("PLAYERS").apply(funct1).reset_index()
temp["points"] = (3.5 * temp["Wkts"]) + (1 * temp["Dots"]) + (2.5 * temp["4's"]) + (3.5 * temp["6's"]) + (
            2.5 * temp["stumpings"]) + (2.5 * temp["catches"])
temp.to_csv("final_points")
print(temp)
'''
