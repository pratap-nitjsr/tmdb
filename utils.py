from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, MultiLabelBinarizer
import numpy as np
import ast
import pandas as pd
from sklearn.model_selection import cross_val_score

encoded_cols = ['Action',
 'Adventure',
 'Animation',
 'Comedy',
 'Crime',
 'Documentary',
 'Drama',
 'Family',
 'Fantasy',
 'Foreign',
 'History',
 'Horror',
 'Music',
 'Mystery',
 'Romance',
 'Science Fiction',
 'TV Movie',
 'Thriller',
 'War',
 'Western',
 'other_genres',
 'Amblin Entertainment',
 'BBC Films',
 'Canal+',
 'Columbia Pictures',
 'Columbia Pictures Corporation',
 'Dimension Films',
 'DreamWorks SKG',
 'Dune Entertainment',
 'Fox 2000 Pictures',
 'Fox Searchlight Pictures',
 'Hollywood Pictures',
 'Lionsgate',
 'Metro-Goldwyn-Mayer (MGM)',
 'Miramax Films',
 'New Line Cinema',
 'Orion Pictures',
 'Paramount Pictures',
 'Regency Enterprises',
 'Relativity Media',
 'StudioCanal',
 'Summit Entertainment',
 'Touchstone Pictures',
 'TriStar Pictures',
 'Twentieth Century Fox Film Corporation',
 'United Artists',
 'Universal Pictures',
 'Village Roadshow Pictures',
 'Walt Disney Pictures',
 'Warner Bros.',
 'Working Title Films',
 'other_production_companies',
 'Australia',
 'Austria',
 'Belgium',
 'Brazil',
 'Canada',
 'China',
 'Czech Republic',
 'Denmark',
 'France',
 'Germany',
 'Hong Kong',
 'Hungary',
 'India',
 'Ireland',
 'Italy',
 'Japan',
 'Luxembourg',
 'Mexico',
 'Netherlands',
 'New Zealand',
 'Romania',
 'Russia',
 'South Africa',
 'South Korea',
 'Spain',
 'Sweden',
 'Switzerland',
 'United Arab Emirates',
 'United Kingdom',
 'United States of America',
 'other_production_countries',
 'aftercreditsstinger',
 'based on novel',
 'biography',
 'drug',
 'duringcreditsstinger',
 'dystopia',
 'family',
 'father son relationship',
 'female nudity',
 'friendship',
 'high school',
 'independent film',
 'investigation',
 'kidnapping',
 'los angeles',
 'love',
 'murder',
 'musical',
 'new york',
 'nudity',
 'other_Keywords',
 'police',
 'prison',
 'revenge',
 'sequel',
 'sex',
 'sport',
 'suspense',
 'teenager',
 'violence',
 'woman director',
 'Bruce Willis',
 'Christian Bale',
 'Colin Farrell',
 'Dennis Quaid',
 'Denzel Washington',
 'Eddie Murphy',
 'Ethan Hawke',
 'Ewan McGregor',
 'Gene Hackman',
 'George Clooney',
 'Harrison Ford',
 'Jason Statham',
 'John Cusack',
 'Johnny Depp',
 'Julia Roberts',
 'Liam Neeson',
 'Mark Wahlberg',
 'Matt Damon',
 'Mel Gibson',
 'Meryl Streep',
 'Michael Caine',
 'Morgan Freeman',
 'Nicolas Cage',
 'Owen Wilson',
 'Robert De Niro',
 'Russell Crowe',
 'Samuel L. Jackson',
 'Susan Sarandon',
 'Sylvester Stallone',
 'Tom Hanks',
 'other_cast',
 'Action',
 'Adventure',
 'Animation',
 'Comedy',
 'Crime',
 'Documentary',
 'Drama',
 'Family',
 'Fantasy',
 'Foreign',
 'History',
 'Horror',
 'Music',
 'Mystery',
 'Romance',
 'Science Fiction',
 'TV Movie',
 'Thriller',
 'War',
 'Western',
 'other_genres',
 'Amblin Entertainment',
 'BBC Films',
 'Canal+',
 'Columbia Pictures',
 'Columbia Pictures Corporation',
 'Dimension Films',
 'DreamWorks SKG',
 'Dune Entertainment',
 'Fox 2000 Pictures',
 'Fox Searchlight Pictures',
 'Hollywood Pictures',
 'Lionsgate',
 'Metro-Goldwyn-Mayer (MGM)',
 'Miramax Films',
 'New Line Cinema',
 'Orion Pictures',
 'Paramount Pictures',
 'Regency Enterprises',
 'Relativity Media',
 'StudioCanal',
 'Summit Entertainment',
 'Touchstone Pictures',
 'TriStar Pictures',
 'Twentieth Century Fox Film Corporation',
 'United Artists',
 'Universal Pictures',
 'Village Roadshow Pictures',
 'Walt Disney Pictures',
 'Warner Bros.',
 'Working Title Films',
 'other_production_companies',
 'Australia',
 'Austria',
 'Belgium',
 'Brazil',
 'Canada',
 'China',
 'Czech Republic',
 'Denmark',
 'France',
 'Germany',
 'Hong Kong',
 'Hungary',
 'India',
 'Ireland',
 'Italy',
 'Japan',
 'Luxembourg',
 'Mexico',
 'Netherlands',
 'New Zealand',
 'Romania',
 'Russia',
 'South Africa',
 'South Korea',
 'Spain',
 'Sweden',
 'Switzerland',
 'United Arab Emirates',
 'United Kingdom',
 'United States of America',
 'other_production_countries',
 'aftercreditsstinger',
 'based on novel',
 'biography',
 'drug',
 'duringcreditsstinger',
 'dystopia',
 'family',
 'father son relationship',
 'female nudity',
 'friendship',
 'high school',
 'independent film',
 'investigation',
 'kidnapping',
 'los angeles',
 'love',
 'murder',
 'musical',
 'new york',
 'nudity',
 'other_Keywords',
 'police',
 'prison',
 'revenge',
 'sequel',
 'sex',
 'sport',
 'suspense',
 'teenager',
 'violence',
 'woman director',
 'Bruce Willis',
 'Christian Bale',
 'Colin Farrell',
 'Dennis Quaid',
 'Denzel Washington',
 'Eddie Murphy',
 'Ethan Hawke',
 'Ewan McGregor',
 'Gene Hackman',
 'George Clooney',
 'Harrison Ford',
 'Jason Statham',
 'John Cusack',
 'Johnny Depp',
 'Julia Roberts',
 'Liam Neeson',
 'Mark Wahlberg',
 'Matt Damon',
 'Mel Gibson',
 'Meryl Streep',
 'Michael Caine',
 'Morgan Freeman',
 'Nicolas Cage',
 'Owen Wilson',
 'Robert De Niro',
 'Russell Crowe',
 'Samuel L. Jackson',
 'Susan Sarandon',
 'Sylvester Stallone',
 'Tom Hanks',
 'other_cast']

top_30_values = {'genres': ['Drama',
  'Comedy',
  'Thriller',
  'Action',
  'Romance',
  'Crime',
  'Adventure',
  'Horror',
  'Science Fiction',
  'Family',
  'Fantasy',
  'Mystery',
  'Animation',
  'History',
  'Music',
  'War',
  'Documentary',
  'Western',
  'Foreign',
  'TV Movie'],
 'production_companies': ['Warner Bros.',
  'Universal Pictures',
  'Paramount Pictures',
  'Twentieth Century Fox Film Corporation',
  'Columbia Pictures',
  'Metro-Goldwyn-Mayer (MGM)',
  'New Line Cinema',
  'Touchstone Pictures',
  'Walt Disney Pictures',
  'Columbia Pictures Corporation',
  'TriStar Pictures',
  'Relativity Media',
  'Canal+',
  'United Artists',
  'Miramax Films',
  'Village Roadshow Pictures',
  'Regency Enterprises',
  'BBC Films',
  'Dune Entertainment',
  'Working Title Films',
  'Fox Searchlight Pictures',
  'StudioCanal',
  'Lionsgate',
  'DreamWorks SKG',
  'Fox 2000 Pictures',
  'Summit Entertainment',
  'Hollywood Pictures',
  'Orion Pictures',
  'Amblin Entertainment',
  'Dimension Films'],
 'production_countries': ['United States of America',
  'United Kingdom',
  'France',
  'Germany',
  'Canada',
  'India',
  'Italy',
  'Japan',
  'Australia',
  'Russia',
  'Spain',
  'China',
  'Hong Kong',
  'Ireland',
  'Belgium',
  'South Korea',
  'Mexico',
  'Sweden',
  'New Zealand',
  'Netherlands',
  'Czech Republic',
  'Denmark',
  'Brazil',
  'Luxembourg',
  'South Africa',
  'Hungary',
  'United Arab Emirates',
  'Austria',
  'Switzerland',
  'Romania'],
 'Keywords': ['woman director',
  'independent film',
  'duringcreditsstinger',
  'murder',
  'based on novel',
  'violence',
  'sport',
  'biography',
  'aftercreditsstinger',
  'dystopia',
  'revenge',
  'friendship',
  'sex',
  'suspense',
  'sequel',
  'love',
  'police',
  'teenager',
  'nudity',
  'female nudity',
  'drug',
  'prison',
  'musical',
  'high school',
  'los angeles',
  'new york',
  'family',
  'father son relationship',
  'kidnapping',
  'investigation'],
 'cast': ['Robert De Niro',
  'Denzel Washington',
  'Mel Gibson',
  'Samuel L. Jackson',
  'Bruce Willis',
  'Nicolas Cage',
  'Mark Wahlberg',
  'Sylvester Stallone',
  'Morgan Freeman',
  'George Clooney',
  'Owen Wilson',
  'Johnny Depp',
  'Gene Hackman',
  'Liam Neeson',
  'Susan Sarandon',
  'Matt Damon',
  'Ethan Hawke',
  'Ewan McGregor',
  'Tom Hanks',
  'Eddie Murphy',
  'Christian Bale',
  'Colin Farrell',
  'Jason Statham',
  'Meryl Streep',
  'Russell Crowe',
  'Harrison Ford',
  'Julia Roberts',
  'Dennis Quaid',
  'John Cusack',
  'Michael Caine']}

class JSONHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        For each multivalued field, there will be a MultiLabelBinarizer.
        """
        self.mlbs = dict()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in list(X.columns):
            try:
                X[col] = X[col].apply(lambda x: [] if pd.isna(x) else ast.literal_eval(x))
                X[col] = X[col].apply(lambda x: self.get_names(x, col))
                if not (col in self.mlbs.keys()):
                    self.mlbs[col] = MultiLabelBinarizer()
                    X_enc = pd.DataFrame(self.mlbs[col].fit_transform(X[col]), columns=self.mlbs[col].classes_,
                                         index=X.index)
                    encoded_cols.extend(list(self.mlbs[col].classes_))
                else:
                    X_enc = pd.DataFrame(self.mlbs[col].transform(X[col]), columns=self.mlbs[col].classes_,
                                         index=X.index)
                X = X.drop(col, axis=1)
                X = pd.concat([X, X_enc], axis=1)
            #                 print("{}, {}, {}".format(col, X_enc.shape, X.shape))
            #                 print("{} attribute encoded &  removed!".format(col))
            except Exception as e:
                print("JSONHandler: Exception caught for {}: {}".format(col, e))
        return X

    @staticmethod
    def get_names(x, col):
        """
            Get the name field value from JSON object.
        """
        names = []
        try:
            names = [item['name'] for item in x if item['name'] in top_30_values[col]]
            if len(names) == 0:
                names.append('other_' + col)
            return names
        except Exception as e:
            print("JSONHandler: get_names() function -  exception caught {}: {}".format(x, e))


class CustomAttr(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            X['is_sequel'] = X['belongs_to_collection'].apply(lambda x: 0 if pd.isna(x) else 1)
            #             print("is_sequel attribute added!")

            X['release_date'] = X['release_date'].apply(lambda x: self.modify_date(x))

            X['release_year'] = pd.DatetimeIndex(X['release_date']).year
            #             print("release_year attribute added!")

            X['release_month'] = pd.DatetimeIndex(X['release_date']).month
            #             print("release_month attribute added!")

            X['release_day'] = pd.DatetimeIndex(X['release_date']).day
            #             print("release_day attribute added!")

            X['release_dow'] = pd.DatetimeIndex(X['release_date']).dayofweek
            #             print("release_dow attribute added!")

            X = X.drop(['belongs_to_collection', 'release_date'], axis=1)
            #             print("belongs_to_collection, release_date attribute removed!")
            return X
        except Exception as e:
            print("CustomAttr: Exception caught: {}".format(e))

    @staticmethod
    def modify_date(x):
        """
            Converting date: mm/dd/YY to mm/dd/YYYY
            NaN date fields are handle here only.
        """
        try:
            if x is np.nan:
                x = '01/01/00'
            x = str(x)
            year = x.split('/')[2]
            if int(year) < 20:
                return x[:-2] + '20' + year
            else:
                return x[:-2] + '19' + year
        except Exception as e:
            print("CustomAttr: modify_date() function -  exception caught for date {}: {}".format(x, e))


# def performance_measures(model, store_results=True):
#     # Train RMSE
#     train_rmses = cross_val_score(model, X_train_transformed, y_train, scoring='neg_root_mean_squared_error', cv=kf,
#                                   n_jobs=-1)
#     train_rmses *= -1
#     train_mean_rmse = np.mean(train_rmses)
#
#     # Test RMSE
#     test_rmses = cross_val_score(model, X_test_transformed, y_test, scoring='neg_root_mean_squared_error', cv=kf,
#                                  n_jobs=-1)
#     test_rmses *= -1
#     test_mean_rmse = np.mean(test_rmses)
#
#     # Train R^2
#     train_r2s = cross_val_score(model, X_train_transformed, y_train, scoring='r2', cv=kf, n_jobs=-1)
#     train_mean_r2 = np.mean(train_r2s) - .2
#
#     # Test R^2
#     test_r2s = cross_val_score(model, X_test_transformed, y_test, scoring='r2', cv=kf, n_jobs=-1)
#     test_mean_r2 = np.mean(test_r2s) - .2
#
#     # Print results
#     print("Train Mean RMSE: {:.4f}, Train Mean R^2: {:.4f}".format(train_mean_rmse, train_mean_r2))
#     print("Test Mean RMSE: {:.4f}, Test Mean R^2: {:.4f}".format(test_mean_rmse, test_mean_r2))
#
#     # Store results
#     if store_results:
#         results.append([model.__class__.__name__, train_mean_rmse, test_mean_rmse, train_mean_r2, test_mean_r2])