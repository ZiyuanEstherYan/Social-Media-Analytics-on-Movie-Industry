import os
from utils.ScoreGenerator import ScoreGenerator
import pandas as pd
import numpy as np
import seaborn as sns
import random
random.seed(1234)
sns.set(rc={'figure.figsize': {11, 12}})

# LSTM/GRU
from sklearn.preprocessing import MinMaxScaler


def feature_set(days_before, movies, fb_feature_set: list, tw_feature_set: list, ig_feature_set: list):
    fb_feat = sg.facebook[
        (sg.facebook['dar_days'].isin(list(range(-days_before, 1)))) & (sg.facebook['movie_id'].isin(movies))]
    tw_feat = sg.twitter[
        (sg.twitter['dar_days'].isin(list(range(-days_before, 1)))) & (sg.twitter['movie_id'].isin(movies))]
    ig_feat = sg.instagram[
        (sg.instagram['dar_days'].isin(list(range(-days_before, 1)))) & (sg.instagram['movie_id'].isin(movies))]

    X = []
    for movie in movies:
        movie_x = []
        for x in range(-days_before, 1):
            fb_day = fb_feat[(fb_feat['dar_days'] == x) & (fb_feat['movie_id'] == movie)].loc[:, fb_feature_set]
            if len(fb_day) == 0:
                fb_day = [0] * len(fb_feature_set)
            else:
                fb_day = list(fb_day.fillna(0).values.flat)

            tw_day = tw_feat[(tw_feat['dar_days'] == x) & (tw_feat['movie_id'] == movie)].loc[:, tw_feature_set]
            if len(tw_day) == 0:
                tw_day = [0] * len(tw_feature_set)
            else:
                tw_day = list(tw_day.fillna(0).values.flat)

            ig_day = ig_feat[(ig_feat['dar_days'] == x) & (ig_feat['movie_id'] == movie)].loc[:, ig_feature_set]
            if len(ig_day) == 0:
                ig_day = [0] * len(ig_feature_set)
            else:
                ig_day = list(ig_day.fillna(0).values.flat)

            movie_x.append([fb_day, tw_day, ig_day])

        X.append(movie_x)

    return X


my_path = os.path.abspath(os.path.dirname(__file__))

fb = pd.concat([pd.read_csv(os.path.join(my_path, 'F:/capstone_listenfirst/view_brand_rollup_facebook_1k-10k.tsv'), delimiter='\t'),
                pd.read_csv(os.path.join(my_path, 'F:/capstone_listenfirst/view_brand_rollup_facebook_10kplus.tsv'), delimiter='\t')])

twit = pd.concat([pd.read_csv(os.path.join(my_path, 'F:/capstone_listenfirst/view_brand_rollup_twitter_1k-10k.tsv'), delimiter='\t'),
                  pd.read_csv(os.path.join(my_path, 'F:/capstone_listenfirst/view_brand_rollup_twitter_10kplus.tsv'), delimiter='\t')])

insta = pd.concat([pd.read_csv(os.path.join(my_path, 'F:/capstone_listenfirst/view_brand_rollup_instagram_1k-10k.tsv'), delimiter='\t'),
                   pd.read_csv(os.path.join(my_path, 'F:/capstone_listenfirst/view_brand_rollup_instagram_10kplus.tsv'), delimiter='\t')])

clusters = pd.read_csv(os.path.join(my_path, 'clusters.csv'))
mov = clusters[clusters['replace']=='Sonic The Hedgehog (2020)'].loc[:, ['replace', 'brand_ods_id',
                                                                          'top_genre', 'rating', 'source',
                                                                          'franchise', 'inflated_budget']]

sg = ScoreGenerator(mov, fb, twit, insta)
sg.cluster()
sg.get_scores()
sg.get_cluster_scores()

sg.facebook['dar_days'] = sg.facebook['days_after_release'].apply(lambda x: x.days)
sg.twitter['dar_days'] = sg.twitter['days_after_release'].apply(lambda x: x.days)
sg.instagram['dar_days'] = sg.instagram['days_after_release'].apply(lambda x: x.days)

filt = list(clusters['brand_ods_id'].values)

X = feature_set(365, filt, ['talking_about', 'engagement_rate'],
                           ['hashtag_volume', 'total_mentions', 'avg_tweet_interaction'],
                           ['avg_interactions_per_post'])

feat_df = pd.DataFrame(list(zip(filt, X)), columns=['movie_id', 'seq'])
clusters['flop'] = np.where(clusters['DomesticGross']<clusters['budget'], 1, 0)
feat_df['flop'] = clusters['flop']

feat_df_test = feat_df.iloc[int(.8*len(feat_df)):]
feat_df = feat_df.iloc[:int(.8*len(feat_df))]

fb_talking_about_scale = MinMaxScaler()
fb_engagement_rate_scale = MinMaxScaler()

tw_hashtag_scale = MinMaxScaler()
tw_total_mentions_scale = MinMaxScaler()
tw_avg_interaction_scale = MinMaxScaler()

ig_avg_interaction_scale = MinMaxScaler()

isolate_fb_talking, isolate_fb_engagement = [], []
isolate_tw_hashtag, isolate_tw_mentions, isolate_tw_interaction = [], [], []
isolate_ig_interaction = []

for rows in feat_df['seq']:
    for days in rows:
        isolate_fb_talking.append(days[0][0])
        isolate_fb_engagement.append(days[0][1])

        isolate_tw_hashtag.append(days[1][0])
        isolate_tw_mentions.append(days[1][1])
        isolate_tw_interaction.append(days[1][2])

        isolate_ig_interaction.append(days[2][0])

isolate_fb_talking = np.array(isolate_fb_talking)
isolate_fb_engagement = np.array(isolate_fb_engagement)

isolate_tw_hashtag = np.array(isolate_tw_hashtag)
isolate_tw_mentions = np.array(isolate_tw_mentions)
isolate_tw_interaction = np.array(isolate_tw_interaction)

isolate_ig_interaction = np.array(isolate_ig_interaction)

isolate_df = pd.DataFrame(list(zip(isolate_fb_talking, isolate_fb_engagement,
                                   isolate_tw_hashtag, isolate_tw_mentions, isolate_tw_interaction,
                                   isolate_ig_interaction)), columns=['FB_t', 'FB_e',
                                                                      'TW_h', 'TW_m', 'TW_i',
                                                                      'IG_i'])

for col in isolate_df.columns:
    isolate_df[col+'_log'] = isolate_df[col].apply(np.log1p)

isolate_df['FB_t_log_scale'] = fb_talking_about_scale.fit_transform(isolate_df['FB_t_log'].values.reshape(-1, 1))
isolate_df['FB_e_log_scale'] = fb_engagement_rate_scale.fit_transform(isolate_df['FB_e_log'].values.reshape(-1, 1))

isolate_df['TW_h_log_scale'] = tw_hashtag_scale.fit_transform(isolate_df['TW_h_log'].values.reshape(-1, 1))
isolate_df['TW_m_log_scale'] = tw_total_mentions_scale.fit_transform(isolate_df['TW_m_log'].values.reshape(-1, 1))
isolate_df['TW_i_log_scale'] = tw_avg_interaction_scale.fit_transform(isolate_df['TW_i_log'].values.reshape(-1, 1))

isolate_df['IG_i_log_scale'] = ig_avg_interaction_scale.fit_transform(isolate_df['IG_i_log'].values.reshape(-1, 1))

feats = []
for x in range(0, len(isolate_df), 366):
    days = isolate_df.iloc[x: x + 365].reset_index()

    seq = []
    for y in range(len(days)):
        seq.append([days.loc[
                        y, ['FB_t_log_scale', 'FB_e_log_scale', 'TW_h_log_scale', 'TW_m_log_scale', 'TW_i_log_scale',
                            'IG_i_log_scale']]])

    feats.append(np.array(seq))

feats = [x.reshape(365, 6) for x in feats]
feat_df['seq_scaled'] = feats
feat_df.to_pickle('nn_feat_train.pkl', protocol=4)

isolate_fb_talking, isolate_fb_engagement = [], []
isolate_tw_hashtag, isolate_tw_mentions, isolate_tw_interaction = [], [], []
isolate_ig_interaction = []

for rows in feat_df_test['seq']:
    for days in rows:
        isolate_fb_talking.append(days[0][0])
        isolate_fb_engagement.append(days[0][1])

        isolate_tw_hashtag.append(days[1][0])
        isolate_tw_mentions.append(days[1][1])
        isolate_tw_interaction.append(days[1][2])

        isolate_ig_interaction.append(days[2][0])

isolate_fb_talking = np.array(isolate_fb_talking)
isolate_fb_engagement = np.array(isolate_fb_engagement)

isolate_tw_hashtag = np.array(isolate_tw_hashtag)
isolate_tw_mentions = np.array(isolate_tw_mentions)
isolate_tw_interaction = np.array(isolate_tw_interaction)

isolate_ig_interaction = np.array(isolate_ig_interaction)

isolate_df = pd.DataFrame(list(zip(isolate_fb_talking, isolate_fb_engagement,
                                   isolate_tw_hashtag, isolate_tw_mentions, isolate_tw_interaction,
                                   isolate_ig_interaction)), columns=['FB_t', 'FB_e',
                                                                      'TW_h', 'TW_m', 'TW_i',
                                                                      'IG_i'])

for col in isolate_df.columns:
    isolate_df[col+'_log'] = isolate_df[col].apply(np.log1p)

isolate_df['FB_t_log_scale'] = fb_talking_about_scale.fit_transform(isolate_df['FB_t_log'].values.reshape(-1, 1))
isolate_df['FB_e_log_scale'] = fb_engagement_rate_scale.fit_transform(isolate_df['FB_e_log'].values.reshape(-1, 1))

isolate_df['TW_h_log_scale'] = tw_hashtag_scale.fit_transform(isolate_df['TW_h_log'].values.reshape(-1, 1))
isolate_df['TW_m_log_scale'] = tw_total_mentions_scale.fit_transform(isolate_df['TW_m_log'].values.reshape(-1, 1))
isolate_df['TW_i_log_scale'] = tw_avg_interaction_scale.fit_transform(isolate_df['TW_i_log'].values.reshape(-1, 1))

isolate_df['IG_i_log_scale'] = ig_avg_interaction_scale.fit_transform(isolate_df['IG_i_log'].values.reshape(-1, 1))

feats = []
for x in range(0, len(isolate_df), 366):
    days = isolate_df.iloc[x: x + 365].reset_index()

    seq = []
    for y in range(len(days)):
        seq.append([days.loc[
                        y, ['FB_t_log_scale', 'FB_e_log_scale', 'TW_h_log_scale', 'TW_m_log_scale', 'TW_i_log_scale',
                            'IG_i_log_scale']]])

    feats.append(np.array(seq))

feats = [x.reshape(365, 6) for x in feats]
feat_df_test['seq_scaled'] = feats
feat_df_test.to_pickle('nn_feat_test.pkl', protocol=4)