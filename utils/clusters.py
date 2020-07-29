import ast
import pickle
import pandas as pd
import numpy as np
np.random.seed(1234)

import prince
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer


dat = pd.read_csv('../data/brands_with_extras_kmodes.csv')

dat['year_q_bin'] = dat['year_q_bin'].astype(str)
dat['genre_list'] = [ast.literal_eval(str(x)) for x in dat['genre_list']]

genre_set = set()
for row in dat['genre_list']:
    for g in row:
        genre_set.add(g)
dat['released_year'] = pd.to_datetime(dat['released_year'], format='%Y').dt.year.astype(int)

dat_filt = dat[dat['released_year']>=2010].drop_duplicates('replace').loc[:, ['replace', 'genre_list', 'top_genre', 'year_q_bin', 'budget_cat',
                                                                        'production_company_bin', 'rating', 'source', 'franchise', 'genre_grouped']]
prince_features = dat_filt

glist_mlb = MultiLabelBinarizer()
ggroup_ohe = OneHotEncoder()
ybin_ohe = OneHotEncoder()
bcat_ohe = OneHotEncoder()
rating_ohe = OneHotEncoder()
src_ohe = OneHotEncoder()

glist = pd.DataFrame(glist_mlb.fit_transform(prince_features['genre_list']).tolist())
genregroup = pd.DataFrame(ggroup_ohe.fit_transform(prince_features['genre_grouped'].values.reshape(-1, 1)).toarray().tolist())
ybin = pd.DataFrame(ybin_ohe.fit_transform(prince_features['year_q_bin'].astype(str).values.reshape(-1, 1)).toarray().tolist())
bcat = pd.DataFrame(bcat_ohe.fit_transform(prince_features['budget_cat'].values.reshape(-1, 1)).toarray().tolist())
rating = pd.DataFrame(rating_ohe.fit_transform(prince_features['rating'].values.reshape(-1, 1)).toarray().tolist())
src = pd.DataFrame(src_ohe.fit_transform(prince_features['source'].values.reshape(-1, 1)).toarray().tolist())

prince_ohe = pd.concat([prince_features.loc[:, 'replace'], glist, genregroup,
                        ybin, bcat, rating, src], axis=1)

mca = prince.MCA()

mca = mca.fit(prince_ohe.iloc[:, 1:])
res = mca.transform(prince_ohe.iloc[:, 1:])

km3 = KMeans(n_clusters=3, n_init=10000)
preds_optimal_3 = km3.fit_predict(res)
dat['clusters_3'] = preds_optimal_3

km4 = KMeans(n_clusters=4, n_init=10000)
preds_optimal_4 = km4.fit_predict(res)
dat['clusters_4'] = preds_optimal_4

km5 = KMeans(n_clusters=5, n_init=10000)
preds_optimal_5 = km5.fit_predict(res)
dat['clusters_5'] = preds_optimal_5

dat.to_csv('alex_clusters.csv')
pickle.dump(glist_mlb, open('glist_mlb.pkl', 'wb'))
pickle.dump(ggroup_ohe, open('ggroup_ohe.pkl', 'wb'))
pickle.dump(ybin_ohe, open('ybin_ohe.pkl', 'wb'))
pickle.dump(bcat_ohe, open('bcat_ohe.pkl', 'wb'))
pickle.dump(rating_ohe, open('rating_ohe.pkl', 'wb'))
pickle.dump(src_ohe, open('src_ohe.pkl', 'wb'))
pickle.dump(mca, open('mca.pkl', 'wb'))
pickle.dump(km3, open('kmeans3.pkl', 'wb'))
pickle.dump(km4, open('kmeans4.pkl', 'wb'))
pickle.dump(km5, open('kmeans5.pkl', 'wb'))
