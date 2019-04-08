from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

import pandas as pd

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_column', 50)
pd.set_option('display.width', 1000)

# Import stats data from csv file
# custom_dataset = pd.read_csv("stats1.csv", nrows = 100)
# participants = pd.read_csv("participants.csv", nrows = 100)
custom_dataset = pd.read_csv("stat1_columntrim.csv")
participants = pd.read_csv("participants.csv")

stats1 = pd.DataFrame(custom_dataset, columns=['id','win','kills','deaths','assists','totdmgdealt','magicdmgdealt','physicaldmgdealt','truedmgdealt','totdmgtaken','magicdmgtaken','physdmgtaken','truedmgtaken','totminionskilled','neutralminionskilled','ownjunglekills','enemyjunglekills'])
part1 = pd.DataFrame(participants)

print(stats1)
print(part1)

Export = stats1.to_json(r'A:\Github\571Project\main\stats1.json')
Export = part1.to_json(r'A:\Github\571Project\main\part1.json')

#print(custom_dataset)

# custom_dataset.head()
#
# feature_cols = ['win','totdmgdealt']
#
# X = custom_dataset.loc[:, feature_cols]
# X.shape
