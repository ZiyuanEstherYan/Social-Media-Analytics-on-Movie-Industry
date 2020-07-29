from pandas.errors import DtypeWarning
import warnings
import os
import ast
import pickle
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
sys.path.append('../')

warnings.simplefilter(action='ignore', category=DtypeWarning)
my_path = os.path.abspath(os.path.dirname(__file__))
clusters = pd.read_csv(os.path.join(my_path, 'clusters.csv'))
clusters = clusters.rename(columns={'brand_ods_id': 'movie_id'})


class ScoreGenerator:
    """
    To use this class:
    Step 1: Create an isolated Pandas DataFrame object for a brand, which must contain the brand ods id, leading genre,
            MPAA rating, source, franchise status and inflated budget
    Step 2: Create a ScoreGenerator object, pass in data for Facebook, Twitter, Instagram as well the brands that have
            already been fit with a clustering algorithm
            i.e. score_gen = ScoreGenerator(brand_features, fb_data, t_data, i_data)
    Step 3: Call the cluster method to create the clusters
            i.e. score_gen.cluster()
    Step 4: Simply call the get_scores function on the new object just created. The function will automatically update
            the stored datasets
            i.e. score_gen.get_scores()
    Step 5: (Optional) To access the new datasets, simply call the object created and extract the intended social media
            channel
            i.e. score_gen.instagram
    Step 6: If all social media scores are present, call the get_cluster_scores to create a combined set of scores
            for all social media channels
            i.e. score_gen.get_cluster_scores()
    Step 7: (Optional) Call the plots method to create interactive visualizations of the three social media channels
            i.e. score_gen.plots()
    """

    def __init__(self, features: pd.DataFrame, fb_dat: pd.DataFrame=None, twit_dat: pd.DataFrame=None, insta_dat: pd.DataFrame=None):
        """
        Creates the ScoreGenerator object. Force params to pandas DataFrame for easy of use. Only required attr is
        features with columns: {'brand_ods_id', 'top_genre', 'rating', 'source', 'franchise', 'inflated_budget'}. Raise
        value error if not all columns are present.
        When social media channel is inputted, join in the movie release date. Create columns 'days_after_release' for
        each respective movie and social media channel identifier, used when joining the three dataframes
        """
        feature_set = {'replace', 'brand_ods_id', 'top_genre', 'rating', 'source', 'franchise', 'inflated_budget'}
        if not feature_set.issubset(features.columns):
            raise ValueError('Not all feature columns present. Make sure the following columns are present before continuing: {}'.format(feature_set))
        self.features = features
        self.client_name = self.features.iloc[0]['replace']
        self.features = self.features.rename(columns={'brand_ods_id': 'movie_id'})
        self.current_id = [self.features.iloc[0]['movie_id']]
        self.brands = clusters
        # if self.current_id in self.brands['movie_id'].values:
        #     raise KeyError('Client movie ID exists in existing clusters')

        if fb_dat is not None:
            self.facebook = fb_dat.fillna(0)
            self.facebook = pd.merge(self.facebook, self.brands.loc[:, ['movie_id', 'released_on']],
                                     on='movie_id')
            self.facebook['data_for'] = pd.to_datetime(self.facebook['data_for']).dt.date
            self.facebook['released_on'] = pd.to_datetime(self.facebook['released_on']).dt.date
            self.facebook['days_after_release'] = self.facebook['data_for'] - self.facebook['released_on']
            self.facebook['identifier'] = ['facebook'] * len(self.facebook)
        else:
            self.facebook = None

        if twit_dat is not None:
            self.twitter = twit_dat.fillna(0)
            self.twitter = pd.merge(self.twitter, self.brands.loc[:, ['movie_id', 'released_on']],
                                    on='movie_id')
            self.twitter['data_for'] = pd.to_datetime(self.twitter['data_for']).dt.date
            self.twitter['released_on'] = pd.to_datetime(self.twitter['released_on']).dt.date
            self.twitter['days_after_release'] = self.twitter['data_for'] - self.twitter['released_on']
            self.twitter['identifier'] = ['twitter'] * len(self.twitter)
        else:
            self.twitter = None

        if insta_dat is not None:
            self.instagram = insta_dat.fillna(0)
            self.instagram = pd.merge(self.instagram, self.brands.loc[:, ['movie_id', 'released_on']],
                                      on='movie_id')
            self.instagram['data_for'] = pd.to_datetime(self.instagram['data_for']).dt.date
            self.instagram['released_on'] = pd.to_datetime(self.instagram['released_on']).dt.date
            self.instagram['days_after_release'] = self.instagram['data_for'] - self.instagram['released_on']
            self.instagram['identifier'] = ['instagram'] * len(self.instagram)
        else:
            self.instagram = None

        self.cluster_brands = None
        self.cluster_set = None

    def cluster(self):
        """
        Use pretrained models to predict a cluster based on a client movie's attributes
        """
        my_path = os.path.abspath(os.path.dirname(__file__))
        budget_scale = pickle.load(open(os.path.join(my_path, 'budget_scale.pkl'), 'rb'))

        self.brands['inflated_budget'] = budget_scale.fit_transform(
            self.brands['inflated_budget'].values.reshape(-1, 1))

        features_ = self.brands[['replace', 'top_genre', 'rating', 'source', 'franchise', 'inflated_budget']]
        features_ = features_.set_index('replace')

        famd = pickle.load(open(os.path.join(my_path, 'famd.pkl'), 'rb'))
        famd2 = famd.fit(features_)
        clust_brand_extras_famd = famd2.row_coordinates(features_)

        hc = pickle.load(open(os.path.join(my_path, 'hc.pkl'), 'rb'))
        features_pred = hc.fit_predict(clust_brand_extras_famd)

        self.cluster_brands = [x for x in list(self.brands[self.brands['cluster_id']==features_pred[0]]['movie_id']) if x != self.current_id[0]]

    @staticmethod
    def fb_score(dat):
        """
        Generate a score for facebook dataframe
        """
        return ((dat['total_post_interactions'] - dat['total_post_comments']) - \
               (0.5 * dat['total_post_sad_count'] + 0.5 * dat['total_post_angry_count'])) / \
               dat['total_post']

    @staticmethod
    def twit_score(dat):
        """
        Generate a score for twitter dataframe
        """
        return ((dat['total_post_interactions'] - dat['total_replies']) + \
                (dat['hashtag_volume'] + dat['keyword_volume'] + dat['cashtag_volume'])) / \
               (dat['tweets'] * (dat['total_replies'] / dat['followers']))

    @staticmethod
    def insta_score(dat):
        """
        Generate a score for instagram dataframe
        """
        return (dat['total_post_interactions']-dat['total_comments']) / (dat['media_count'] * (dat['total_comments'] / dat['followed_by_count']))

    def get_scores(self):
        """
        Generate scores for each social media channel that is present
        """
        if self.facebook is not None:
            facebook_cluster = self.facebook[(self.facebook['movie_id'].isin(self.cluster_brands)) & \
                                             (self.facebook['total_post'] != 0)]
            facebook_cluster['score'] = facebook_cluster.apply(self.fb_score, axis=1)

            facebook_client = self.facebook[(self.facebook['movie_id'].isin(self.current_id)) & \
                                            (self.facebook['total_post'] != 0)]
            if len(facebook_client) != 0:
                facebook_client['score'] = facebook_client.apply(self.fb_score, axis=1)

            self.facebook = pd.concat([facebook_cluster, facebook_client], ignore_index=True)
            self.facebook['cluster_identifier'] = np.where(self.facebook['movie_id']==self.current_id[0],
                                                           'client',
                                                           'cluster')
            self.facebook = self.facebook.sort_values(['cluster_identifier', 'days_after_release'], ascending=True)

        if self.twitter is not None:
            twitter_cluster = self.twitter[(self.twitter['movie_id'].isin(self.cluster_brands)) & \
                                           (self.twitter['tweets'] != 0) & \
                                           (self.twitter['followers'] != 0) & \
                                           (self.twitter['total_replies'] != 0)]
            twitter_cluster['score'] = twitter_cluster.apply(self.twit_score, axis=1)

            twitter_client = self.twitter[(self.twitter['movie_id'].isin(self.current_id)) & \
                                          (self.twitter['tweets'] != 0) & \
                                          (self.twitter['followers'] != 0) & \
                                          (self.twitter['total_replies'] != 0)]
            if len(twitter_client) != 0:
                twitter_client['score'] = twitter_client.apply(self.twit_score, axis=1)

            self.twitter = pd.concat([twitter_cluster, twitter_client], ignore_index=True)
            self.twitter['cluster_identifier'] = np.where(self.twitter['movie_id']==self.current_id[0],
                                                          'client',
                                                          'cluster')
            self.twitter = self.twitter.sort_values(['cluster_identifier', 'days_after_release'], ascending=True)

        if self.instagram is not None:
            instagram_cluster = self.instagram[(self.instagram['movie_id'].isin(self.cluster_brands)) & \
                                               (self.instagram['media_count'] != 0) & \
                                               (self.instagram['total_comments'] != 0) & \
                                               (self.instagram['followed_by_count'] != 0)]
            instagram_cluster['score'] = instagram_cluster.apply(self.insta_score, axis=1)

            instagram_client = self.instagram[(self.instagram['movie_id'].isin(self.current_id)) & \
                                              (self.instagram['media_count'] != 0) & \
                                              (self.instagram['total_comments'] != 0) & \
                                              (self.instagram['followed_by_count'] != 0)]
            if len(instagram_client) != 0:
                instagram_client['score'] = instagram_client.apply(self.insta_score, axis=1)

            self.instagram = pd.concat([instagram_cluster, instagram_client], ignore_index=True)
            self.instagram['cluster_identifier'] = np.where(self.instagram['movie_id']==self.current_id[0],
                                                            'client',
                                                            'cluster')
            self.instagram = self.instagram.sort_values(['cluster_identifier', 'days_after_release'], ascending=False)

    def get_cluster_scores(self):
        """
        Join the three social media channels and aggregate score by days after release column. Will throw error if
        three social media channels are not present
        """
        if self.facebook is not None and self.twitter is not None and self.instagram is not None:
            self.cluster_set = pd.concat([self.facebook, self.twitter, self.instagram],
                                         ignore_index=True)
            self.cluster_set = pd.DataFrame(self.cluster_set.groupby(['identifier', 'cluster_identifier', 'days_after_release']).mean()['score']).reset_index()
            self.cluster_set['days_after_release'] = self.cluster_set['days_after_release'].apply(lambda x: x.days)
            self.cluster_set = self.cluster_set[self.cluster_set['days_after_release'].isin(list(range(-365, 365)))]

        else:
            raise ValueError('Not all rollups present, call the get_scores method first or pass all three datasets when creating the ScoreGenerator object')

    def plots(self):
        """
        Use plotly graph_objects module to create three subplots (one for each social media channel) comparing the client's
        score from 1 year before release to 1 year after, as well as the the scores for the cluster the client was assigned
        to over the same range. Will throw error if three social media datasets are not present.
        """
        if self.cluster_set is not None:
            fig = make_subplots(rows=3, cols=1,
                                x_title='Days After Release',
                                y_title='Score',
                                shared_xaxes=True,
                                subplot_titles=('Facebook', 'Twitter', 'Instagram'))

            row = 1
            legend_ = True
            for x in ['facebook', 'twitter', 'instagram']:
                if row != 1:
                    legend_ = False
                filt = self.cluster_set[(self.cluster_set['identifier'] == x) & (self.cluster_set['cluster_identifier'] == 'cluster')]
                trace = fig.add_trace(go.Scatter(x=filt['days_after_release'], y=filt['score'],
                                                 legendgroup='group', name='Clustered Movies', showlegend=legend_,
                                                 line=dict(color='firebrick')),
                                                 row=row, col=1)

                filt = self.cluster_set[(self.cluster_set['identifier'] == x) & (self.cluster_set['cluster_identifier'] == 'client')]
                if len(filt) != 0:
                    trace.append_trace(go.Scatter(x=filt['days_after_release'], y=filt['score'],
                                                  legendgroup='group', name='Client', showlegend=legend_,
                                                  line=dict(color='royalblue')),
                                                  row=row, col=1)
                row += 1

            filt_fb = self.cluster_set[(self.cluster_set['identifier'] == 'facebook')]
            filt_tw = self.cluster_set[(self.cluster_set['identifier'] == 'twitter')]
            filt_ig = self.cluster_set[(self.cluster_set['identifier'] == 'instagram')]
            fig.update_layout(
                title='Score Trend Plot for {}'.format(self.client_name),
                width=1000,
                height=500,
                shapes=[
                    dict(type='line', x0=0, x1=0, y0=0, y1=filt_fb['score'].max(), line_width=2, xref='x1', yref='y1'),
                    dict(type='line', x0=0, x1=0, y0=0, y1=filt_tw['score'].max(), line_width=2, xref='x1', yref='y2'),
                    dict(type='line', x0=0, x1=0, y0=0, y1=filt_ig['score'].max(), line_width=2, xref='x1', yref='y3')
                ]
            )
            fig.show()

        else:
            raise ValueError('Call the get_cluster_scores method with all three social media datasets present before calling this method')



