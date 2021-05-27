import numpy as np
import pandas as pd


BBB = pd.read_csv("/home/prem/ipl_data/data1_ball_to_ball.csv")
PLA_POINTS = pd.read_csv("/home/prem/final_points.csv")
MAT = pd.read_csv("/home/prem/PycharmProjects/SSSIHL-IPL/datasets/MATCHES.csv")

BBB["fielder"] = BBB["fielder"].str.upper()
BBB["fielder"] = BBB["fielder"].str.strip(" (sub)")
BBB["batsman"] = BBB["batsman"].str.upper()
BBB["non_striker"] = BBB["non_striker"].str.upper()
BBB["bowler"] = BBB["bowler"].str.upper()


def team_score(l1):  # Calculates Total points earned by the team
    df1 = pd.DataFrame(l1)
    df2 = df1.merge(PLA_POINTS)
    df2 = pd.DataFrame(df2)
    return round((df2['points'].sum() / 11), 2)


def func1(temp):
    temp1 = temp[temp["inning"] == 1]
    temp2 = temp[temp["inning"] == 2]

    # print(type(np.array_str((temp1["batting_team"].unique()))) )

    batting_team = temp1["batting_team"].unique()
    bowling_team = temp1["bowling_team"].unique()

    # team1
    team1 = ()
    l1 = set(temp1["batsman"].unique())
    l2 = set(temp1["non_striker"].unique())
    l3 = set(temp2["bowler"].unique())
    l4 = set(temp2["fielder"].unique())
    team1_list = {'PLAYERS': list(l1.union(l2, l3, l4))}
    team1_score = team_score(team1_list)

    team1 = (batting_team, team1_score)

    # team2
    team2 = ()
    l1 = set(temp2["batsman"].unique())
    l2 = set(temp2["non_striker"].unique())
    l3 = set(temp1["bowler"].unique())
    l4 = set(temp1["fielder"].unique())
    team2_list = {'PLAYERS': list(l1.union(l2, l3, l4))}
    team2_score = team_score(team2_list)
    team2 = (bowling_team, team2_score)

    return pd.Series([team1, team2], index=["batting_team", "bowling_team"])


df4 = BBB.groupby("id").apply(func1).reset_index()

# Adding team1 score and team2 score to the matches dataset
team1_score = []
team2_score = []
for i in range(0, len(df4)):
    if MAT["team1"][i] == df4["batting_team"][i][0]:
        team1_score.append(df4["batting_team"][i][1])
        team2_score.append(df4["bowling_team"][i][1])
    else:
        if MAT["team2"][i] == df4["batting_team"][i][0]:
            team2_score.append(df4["batting_team"][i][1])
            team1_score.append(df4["bowling_team"][i][1])

MAT["team1_score"] = pd.DataFrame(team1_score)
MAT["team2_score"] = pd.DataFrame(team2_score)

# MAT.to_csv("final_matches_data.csv")
print(MAT.head())
