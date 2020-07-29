from pandas.errors import DtypeWarning
import warnings
import os
import json
import pandas as pd
import numpy as np
import regex as re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import plotly.express as px
from plotly.subplots import make_subplots

warnings.simplefilter(action='ignore', category=DtypeWarning)
corpus = set(nltk.corpus.words.words())
stopwords = set(stopwords.words('english'))
stopwords.add('I')

my_path = os.path.abspath(os.path.dirname(__file__))
brands = pd.read_csv(os.path.join(my_path, 'brand_concat.csv'))
clusters = pd.read_csv(os.path.join(my_path, 'clusters.csv'))


class TextPlots:
    """
    Class for creating exploratory plots using techniques such as word frequency charts and wordclouds. Similar to
    ScoreGenerator class;
    Step 1. Create TextPlots object with target movie_id, range of days "after" release for target (negative for days
            before release) and comments data
            i.e. tp = TextPlots(1234, [-365, 365], comments)
    Step 2. Choose between plotting functions to view visualizations based on parameters set. options include
            count_plot, sentiment_plot, wordcloud
            i.e. tp.count_plot()
                 tp.sentiment_plot()
                 tp.wordcloud()
    """

    def __init__(self, movie_id, days_window: list, comments):
        """
        Pass in target movie_id, target days and full comments data. Comments data is automatically filtered based on
        parameters. Raise value error if days out of range
        """
        self.movie_id = movie_id
        dcs_uid = brands[brands['brand_ods_id'] == self.movie_id]['data_profile_dcs_uid'].values

        self.comments = comments
        self.comments = self.comments[self.comments['post_author_dcs_uid'].isin(dcs_uid)]

        self.comments = self.comments[self.comments['comment_message'].notna()]
        if len(self.comments) == 0:
            raise LookupError('No comments found for this movie. Try a larger set of comments or check the movie id inputted')

        self.comments['released_on'] = clusters[clusters['brand_ods_id'] == self.movie_id]['released_on'].iloc[0]
        self.comments['comment_posted_at'] = pd.to_datetime(self.comments['comment_posted_at']).dt.date
        self.comments['released_on'] = pd.to_datetime(self.comments['released_on']).dt.date
        self.comments['days_after_release'] = self.comments['comment_posted_at'] - self.comments['released_on']
        self.comments['days_after_release'] = pd.to_numeric(self.comments['days_after_release'].apply(lambda x: x.days))

        sentiment = []
        for x in self.comments['comment_ace_metadata']:
            try:
                sentiment.append(json.loads(x)['lfm/sentiment_polarity'])
            except:
                sentiment.append('neu')
        self.comments['sentiment'] = sentiment

        if days_window[0] >= days_window[1] or days_window[0] < self.comments['days_after_release'].min() or \
            days_window[1] > self.comments['days_after_release'].max():
            raise KeyError('Days out of bounds, try again with a larger set of comments or adjust the days after release parameter')
        self.day_window = days_window

    def count_plot(self, top_words=25, clean=True):
        """
        Create a word count bar chart. Choose the amount of words to include, as well as whether to clean the comment
        text before plotting. If no words are found on a particular day, the function will automatically find the next
        day with comments by updating the days_after_release parameter. NOTE: this affects the days_after_release
        parameter associated with the current object. Any subsequent calls to class methods will use the updated
        value.
        """
        if top_words > 25:
            warnings.warn('Including more than 25 words on the plot will cause labels to be excluded')

        daily_comments = self.comments[(self.comments['days_after_release'].\
                                            isin(list(range(self.day_window[0], self.day_window[1]+1))))]
        if len(daily_comments) == 0:
            warnings.warn('No comments found for this day, trying future dates until comments are found')

            while len(daily_comments) == 0 and self.day_window[1] <= self.comments['days_after_release'].max():
                if self.day_window[1] > self.comments['days_after_release'].max():
                    raise KeyError('Reached bounds of comment dates available. Make sure all comments are present')
                self.day_window[1] += 1
                daily_comments = self.comments[(self.comments['days_after_release'].\
                                                    isin(list(range(self.day_window[0], self.day_window[1]+1))))]

        print('Now looking at {} to {} days after release'.format(self.day_window[0], self.day_window[1]))

        left = np.where(self.day_window[0] < 0, 'Before', 'After')
        right = np.where(self.day_window[1] < 0, 'Before', 'After')
        if clean:
            daily_comments['clean_comments'] = daily_comments['comment_message'].apply(self.comment_cleaner)
            res = daily_comments['clean_comments'].str.split(expand=True).stack().value_counts().to_dict()
            fig = px.bar(x=list(res.keys())[:top_words], y=list(res.values())[:top_words])
            fig.update_layout(
                title='Top {} Words at {} Days {} Release to {} Days {} Release'.format(top_words,
                                                                                        self.day_window[0], left,
                                                                                        self.day_window[1], right),
                yaxis_title='Count',
                xaxis_tickangle=-45
            )
            fig.show()

        else:
            res = daily_comments['clean_comments'].str.split(expand=True).stack().value_counts().to_dict()
            fig = px.bar(x=list(res.keys())[:top_words], y=list(res.values())[:top_words])
            fig.update_layout(
                title='Top {} Words at {} Days {} Release to {} Days {} Release'.format(top_words,
                                                                                        self.day_window[0], left,
                                                                                        self.day_window[1], right),
                yaxis_title='Count',
                xaxis_tickangle=-45
            )
            fig.show()

    def sentiment_plot(self, top_words=25):
        """
        Similar to count_plot function, but creates separate subplots for positive, neutral and negative polarity. In
        this case, comments are not clean to make sure comments are evenly distributed between sentiments.
        """
        if top_words > 25:
            warnings.warn('Including more than 25 words on the X-axis will cause words to be excluded from the axis')

        daily_comments = self.comments[(self.comments['days_after_release'].\
                                        isin(list(range(self.day_window[0], self.day_window[1] + 1))))]
        if len(daily_comments) == 0:
            warnings.warn('No comments found for this day, trying future dates until comments are found')

            while len(daily_comments) == 0:
                if self.day_window[1] > self.comments['days_after_release'].max():
                    raise KeyError('Reached bounds of comment dates available. Make sure all comments are present')
                self.day_window[1] += 1
                daily_comments = self.comments[(self.comments['days_after_release'].\
                                                isin(list(range(self.day_window[0], self.day_window[1] + 1))))]

            print('Now looking at {} to {} days after release'.format(self.day_window[0], self.day_window[1]))

        if 'pos' not in daily_comments['sentiment'].values or 'neu' not in daily_comments['sentiment'].values or \
                'neg' not in daily_comments['sentiment'].values:
            warnings.warn('No negative or positive sentiments found on this day, trying future dates until positive or negative comments are found')

            while 'pos' not in daily_comments['sentiment'].values or 'neu' not in daily_comments['sentiment'].values or \
                    'neg' not in daily_comments['sentiment'].values:
                if self.day_window[1] > self.comments['days_after_release'].max():
                    raise KeyError('Reached bounds of comment dates available. Make sure all comments are present')
                self.day_window[1] += 1
                daily_comments = self.comments[(self.comments['days_after_release']. \
                                                isin(list(range(self.day_window[0], self.day_window[1] + 1))))]

        print('Now looking at {} to {} days after release'.format(self.day_window[0], self.day_window[1]))

        res_positive = daily_comments[(daily_comments['sentiment']=='pos')]['comment_message'].str.split(expand=True)\
            .stack().value_counts().to_dict()
        res_neutral = daily_comments[(daily_comments['sentiment']=='neu')]['comment_message'].str.split(expand=True)\
            .stack().value_counts().to_dict()
        res_negative = daily_comments[daily_comments['sentiment']=='neg']['comment_message'].str.split(expand=True)\
            .stack().value_counts().to_dict()

        fig = make_subplots(rows=3, cols=1,
                            y_title='Count',
                            subplot_titles=('Positive', 'Neutral', 'Negative'))
        trace = fig.add_trace(px.bar(x=list(res_positive.keys())[:top_words], y=list(res_positive.values())[:top_words]).data[0],
                              row=1, col=1)
        fig.append_trace(px.bar(x=list(res_neutral.keys())[:top_words], y=list(res_neutral.values())[:top_words]).data[0],
                         row=2, col=1)
        fig.append_trace(px.bar(x=list(res_negative.keys())[:top_words], y=list(res_negative.values())[:top_words]).data[0],
                         row=3, col=1)

        left = np.where(self.day_window[0] < 0, 'Before', 'After')
        right = np.where(self.day_window[1] < 0, 'Before', 'After')
        fig.update_layout(
            title='Top {} Words at {} Days {} Release to {} Days {} Release'.format(top_words,
                                                                                    self.day_window[0], left,
                                                                                    self.day_window[1], right)
        )
        fig.show()

    def wordcloud(self):
        """
        Generate wordcloud using range of days 1 year before to 1 year after release by default. Will only use cleaned
        and lemmatized text
        """
        if 'clean_comments' not in self.comments.columns:
            self.comments['clean_comments'] = self.comments['comment_message'].apply(self.comment_cleaner)

        self.comments = self.comments[self.comments['clean_comments'].notna()]

        if 'text_lemmatized' not in self.comments.columns:
            self._proc(self.comments)

        long_str = ','.join(list(self.comments['text_lemmatized']))

        wc = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue',
                       width=1600, height=800)
        wc.generate(long_str)
        return wc.to_image()

    @staticmethod
    def _proc(dat):
        """
        Apply lemmatization to cleaned comments
        """
        def lemma(text):
            lemmatizer = WordNetLemmatizer()
            w_tokenizer = WhitespaceTokenizer()
            return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

        dat['text_lemmatized'] = dat['clean_comments'].apply(lemma)
        dat['text_lemmatized'] = dat['text_lemmatized'].apply(' '.join)

    @staticmethod
    def comment_cleaner(text):
        """
        Clean comments using regex and filtering
        """
        text = re.sub("[^\w\s]", "", text)
        text = " ".join([x.lower() for x in text.split(' ') if x.lower() in corpus and x.lower() not in stopwords and len(x) > 1])
        if text == '':
            return np.nan
        return text