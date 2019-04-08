import numpy as np
import matplotlib as plt
import pandas as pd
from sklearn.cluster import AffinityPropagation
from sklearn import preprocessing
from sklearn import metrics

PATCH = "7.9.186.8155"
MAOKAI = 57

allMatches = pd.read_csv("data/matches.csv")
matches = allMatches.loc[allMatches['version'] == PATCH]
stats1 = pd.read_csv("stat1_columntrim.csv")

allBans = pd.read_csv("data/teambans.csv")
allParticipants = pd.read_csv("data/participants.csv")
participants = allParticipants.loc[allParticipants['matchid'].isin(matches['id'])]
bans = allBans.loc[allBans['matchid'].isin(matches['id'])]
maokaiPlayers = participants.loc[participants['championid'] == MAOKAI]
#junglers = participants.loc[participants['position'] == "JUNGLE"]
champs = pd.read_csv("data/champs.csv")

maokaiGames = matches.loc[matches['id'].isin(maokaiPlayers['matchid'])]
maokaiStats = maokaiPlayers.merge(stats1, left_on = 'id', right_on = 'id').drop("id", axis=1)
#allPlayerStats = participants.merge(stats1, left_on = 'id', right_on = 'id').drop("id", axis=1)

bans.to_csv("bans.csv", index=False)
matches.to_csv("test.csv",index=False)
participants.to_csv("matchdata.csv",index=False)
maokaiStats.to_csv("maokaiStats.csv",index=False)
posDict = {"TOP": 1, "JUNGLE": 2, "MID": 3, "BOT": 4}
roleDict = {"SOLO": 0, "NONE": 0, "DUO_CARRY": 0, "DUO_SUPPORT": 1}
playerDict = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}

maokaiStats['position'] = maokaiStats['position'].map(posDict)
maokaiStats['role'] = maokaiStats['role'].map(roleDict)
maokaiStats['player'] = maokaiStats['player'].map(playerDict)

maokaiStats.to_csv("maokaiStats.csv",index=False)
