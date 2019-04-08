import pandas as pd
from sklearn import preprocessing

dataset = pd.read_csv("tankStats.csv")

df = pd.DataFrame(dataset)

normalize_cols = ['kills','deaths','assists','totdmgdealt','magicdmgdealt','physicaldmgdealt','truedmgdealt','totdmgtaken','magicdmgtaken','physdmgtaken','truedmgtaken','totminionskilled','neutralminionskilled','ownjunglekills','enemyjunglekills']

#new_data[normalize_cols] = df[normalize_cols].apply(lambda x:(x-x.min())/(x.max()-x.min()))

# df[normalize_cols] = StandardScaler().fit_transform(df[normalize_cols])

# x = df[normalize_cols].values.astype(float)
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# df_normalize = pd.DataFrame(x_scaled)
# print(df_normalize)

df[normalize_cols] = round(((df[normalize_cols]-df[normalize_cols].min())/(df[normalize_cols].max()-df[normalize_cols].min())), 5)
Export = df.to_csv('tankStatsNormalized.csv', sep='\t')

print()

#print(df)
