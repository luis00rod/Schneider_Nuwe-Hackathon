# This program classifies the type of volcano based on the vibrations detected by sensors.

# Getting libraries
import pickle
import re
import json
import PyPDF2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from decimal import Decimal
from sklearn.model_selection import train_test_split

class Schneider:
    """
    This class is used to classify the type of volcano based on the vibrations detected by sensors. Contains methods to
    preprocess the data, train the model, and test the model, as well as methods to save the predictions and the model.
    The model used is a Random Forest Classifier.
    """

    def __init__(self):
        """
        Inizialize the class and define the model to use
        """
        self.dict_csv_nuwe = {'countryName': 'countryName', 'eprtrSectorName': 'eptrSectorName',
                              'EPRTRAnnexIMainActivityLabel': 'EPRTRAnnexIMainActivityLabel',
                              'FacilityInspireID': 'FacilityInspireID', 'facilityName': 'facilityName',
                              'City': 'City', 'targetRelease': 'targetRelease', 'pollutant': 'pollutant',
                              'reportingYear': 'reportingYear', 'MONTH': 'MONTH', 'DAY': 'DAY',
                              'CONTINENT': 'CONTINENT',
                              'max_wind_speed': 'max_wind_speed', 'avg_wind_speed': 'avg_wind_speed',
                              'min_wind_speed': 'min_wind_speed', 'max_temp': 'max_temp', 'avg_temp': 'avg_temp',
                              'min_temp': 'min_temp',
                              'DAY WITH FOGS': 'DAY_WITH_FOG', 'REPORTER NAME': 'REPORTER_NAME', 'CITY ID': 'CITY_ID'}
        self.model = None
        self.model_path = "../models/"
        self.data_test = pd.read_csv('../data/test_x.csv')
        self.prepare_test_data()
        self.data_train1 = pd.read_csv('../data/train1.csv')
        self.data_train2 = pd.read_csv('../data/train2.csv', sep=";")
        self.data_train3 = pd.read_json('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/first')
        self.data_train4 = pd.read_json(
            'http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/second')
        self.data_train5 = pd.read_json('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/third')
        self.data_csv_json = self.create_df_from_json_and_csv()
        self.df_pdf = self.create_df_from_pdf()
        self.data_csv_json = self.decimal_refactor(self.data_csv_json)
        self.df_pdf = self.decimal_refactor(self.df_pdf)
        self.data_train = pd.concat([self.df_pdf, self.data_csv_json]).reset_index()
        self.data_train = self.strip_blank_spaces(self.data_train, 1)
        self.data_local_train = pd.DataFrame()
        self.data_local_test = pd.DataFrame()
        self.best_params = dict()
        self.rdn_number = np.random.randint(200)

    def prepare_test_data(self):
        self.data_test = self.data_test.rename(columns=self.dict_csv_nuwe)
        columns_to_drop = ['facilityName', 'FacilityInspireID', 'CITY_ID', 'targetRelease', 'CONTINENT', 'test_index',
                           'City', 'EPRTRAnnexIMainActivityCode', 'REPORTER_NAME', 'countryName', 'eptrSectorName',
                           'EPRTRAnnexIMainActivityLabel']
        self.data_test = self.decimal_refactor(self.data_test)
        self.data_test = self.strip_blank_spaces(self.data_test, 0)
        self.data_test = self.data_test.drop(columns_to_drop, axis=1)

    @staticmethod
    def decimal_refactor(df):
        df['max_wind_speed'] = df['max_wind_speed'].astype(str).apply(lambda x: x.replace(',', '.')) \
            .apply(lambda x: Decimal(x))
        df['min_wind_speed'] = df['min_wind_speed'].astype(str).apply(lambda x: x.replace(',', '.')) \
            .apply(lambda x: Decimal(x))
        df['avg_wind_speed'] = df['avg_wind_speed'].astype(str).apply(lambda x: x.replace(',', '.')) \
            .apply(lambda x: Decimal(x))
        df['max_temp'] = df['max_temp'].astype(str).apply(lambda x: x.replace(',', '.')) \
            .apply(lambda x: Decimal(x))
        df['min_temp'] = df['min_temp'].astype(str).apply(lambda x: x.replace(',', '.')) \
            .apply(lambda x: Decimal(x))
        df['avg_temp'] = df['avg_temp'].astype(str).apply(lambda x: x.replace(',', '.')) \
            .apply(lambda x: Decimal(x))
        return df

    @staticmethod
    def strip_blank_spaces(df, opt):
        df['countryName'] = df['countryName'].str.replace(" ", "")
        df['eptrSectorName'] = df['eptrSectorName'].str.replace(" ", "")
        df['EPRTRAnnexIMainActivityLabel'] = df['EPRTRAnnexIMainActivityLabel']. \
            str.replace(" ", "")
        df['City'] = df['City'].str.replace(" ", "")
        df['targetRelease'] = df['targetRelease'].str.replace(" ", "")
        df['CONTINENT'] = df['CONTINENT'].str.replace(" ", "")
        if opt == 1:
            df['pollutant'] = df['pollutant'].str.replace(" ", "")

        return df

    @staticmethod
    def load_pdf(idd):

        pdfFileObj = open('../data/train6/pdfs' + str(idd) + '.pdf', 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
        pageObj = pdfReader.getPage(0)
        pdf = pageObj.extractText()
        pdf = pdf.replace('\n \n', ', ').replace('\n', '').replace(' ', '')

        return pdf

    @staticmethod
    def regex(str_pdf):
        pattern = 'REPORTCONTAMINACIÓNnº:(\S+)FACILITYNAME:(\S+)FacilityInspireID:(\S+)COUNTRY:(\S+)CONTINENT:(\S+)' \
                  'CITY:(\S+)EPRTRSectorCode:(\S+)eprtrSectorName:(\S+)MainActivityCode:(\S+)targetRealase:(\S+)pollu' \
                  'tant:(\S+)emissions:(\S+)DAY:(\S+)MONTH:(\S+)YEAR:(\S+)METEOROLOGICALCONDITIONSmax_wind_speed:' \
                  '(\S+)min_wind_speed:(\S+)avg_wind_speed:(\S+)max_temp:(\S+)min_temp:(\S+)avg_temp:(\S+)DAYSFOG:' \
                  '(\S+)REPORTERNAME:(\S+)CITY_ID(\S+)'

        list_pdf_info = re.split(pattern, str_pdf)
        list_pdf_info[2] = list_pdf_info[2][:-5]
        list_pdf_info = list_pdf_info[2:-1]
        del (list_pdf_info[21])
        return list_pdf_info

    def create_df_from_pdf(self):

        columns_in_pdf = ['facilityName', 'FacilityInspireID', 'countryName', 'CONTINENT', 'City', 'EPRETRSectorCode',
                          'eptrSectorName', 'EPRTRAnnexIMainActivityLabel', 'targetRelease', 'pollutant', 'emissions',
                          'DAY', 'MONTH', 'reportingYear', 'max_wind_speed', 'min_wind_speed', 'avg_wind_speed',
                          'max_temp', 'min_temp', 'avg_temp', 'DAY_WITH_FOG', 'CITY_ID']

        df_pdf = pd.DataFrame(columns=columns_in_pdf)
        str_pdf = ''
        for i in range(15, 97):
            num = '815' + str(i)
            try:
                str_pdf = self.load_pdf(num)
            except FileNotFoundError:
                print('FileNotFoundError')
            list_pdf_info = self.regex(str_pdf)
            df_pdf.loc[len(df_pdf)] = list_pdf_info

        str_pdf = self.load_pdf('-1')
        list_pdf_info = self.regex(str_pdf)
        df_pdf.loc[len(df_pdf)] = list_pdf_info

        return df_pdf.drop(['emissions', 'EPRETRSectorCode'], axis=1)

    def create_df_from_json_and_csv(self):

        dict_json_nuwe = {'CITY ID': 'CITY_ID', 'CONTINENT': 'CONTINENT', 'City': 'City', 'DAY': 'DAY',
                          'DAY WITH FOGS': 'DAY_WITH_FOG', 'EPRTRAnnexIMainActivityCode': 'EPRTRAnnexIMainActivityCode',
                          'EPRTRAnnexIMainActivityLabel': 'EPRTRAnnexIMainActivityLabel',
                          'EPRTRSectorCode': 'EPRETRSectorCode', 'FacilityInspireID': 'FacilityInspireID',
                          'MONTH': 'MONTH', 'REPORTER NAME': 'REPORTER_NAME', 'avg_temp': 'avg_temp',
                          'avg_wind_speed': 'avg_wind_speed', 'countryName': 'countryName',
                          'eprtrSectorName': 'eptrSectorName', 'facilityName': 'facilityName', 'max_temp': 'max_temp',
                          'max_wind_speed': 'max_wind_speed', 'min_temp': 'min_temp',
                          'min_wind_speed': 'min_wind_speed',
                          'pollutant': 'pollutant', 'reportingYear': 'reportingYear', 'targetRelease': 'targetRelease'}
        # Join data
        columns_non_in_csv = ['EPRETRSectorCode', 'EPRTRAnnexIMainActivityCode', '']

        self.data_train1 = self.data_train1.rename(columns=self.dict_csv_nuwe)
        self.data_train2 = self.data_train2.rename(columns=self.dict_csv_nuwe)
        self.data_train3 = self.data_train3.rename(columns=dict_json_nuwe).drop(columns_non_in_csv, axis=1)
        self.data_train4 = self.data_train4.rename(columns=dict_json_nuwe).drop(columns_non_in_csv, axis=1)
        self.data_train5 = self.data_train5.rename(columns=dict_json_nuwe).drop(columns_non_in_csv, axis=1)
        data_frames = [self.data_train1, self.data_train2, self.data_train3, self.data_train4, self.data_train5]

        return pd.concat(data_frames, axis=0).drop(['REPORTER_NAME'], axis=1)

    def preprocess_data(self):
        """
        This function preprocesses the data to be used in the model. Perform a normal scaling to both,
        train and test data set.
        The training set is split into two different sets, based on the class distribution:
            1.- 80% of the data is just for local training.
            2.- 20% of the data is only for local testing.

        """

        def _map(x):
            if x == 'Nitrogenoxides(NOX)':
                return 0
            elif x == 'Carbondioxide(CO2)':
                return 1
            elif x == 'Methane(CH4)':
                return 2
            else:
                return None

        self.data_train['target'] = self.data_train['pollutant'].apply(_map)
        columns_to_drop = ['facilityName', 'FacilityInspireID', 'CITY_ID', 'targetRelease', 'CONTINENT', 'pollutant',
                           'City', 'countryName', 'eptrSectorName', 'EPRTRAnnexIMainActivityLabel']
        self.data_train.drop(columns_to_drop, axis=1, inplace=True)

        # Scale the training data
        scale = StandardScaler()
        scale.fit(self.data_train.drop("target", axis=1))
        x_scaled = scale.transform(self.data_train.drop("target", axis=1))
        columns_names = self.data_train.columns
        self.data_train = pd.concat([pd.DataFrame(x_scaled), self.data_train.target], axis=1)
        self.data_train.columns = columns_names

        # Scale the test data
        scale.fit(self.data_test)
        x_test_scaled = scale.transform(self.data_test)
        columns_names = self.data_test.columns
        self.data_test = pd.DataFrame(x_test_scaled)
        self.data_test.columns = columns_names

    def train_model(self):
        """
        This function trains the model.
        :return:
        """

        X = self.data_train.drop(['target'], axis=1)
        y = self.data_train.target

        # Split dataset into train and test
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.rdn_number)

        self.model = RandomForestClassifier(criterion='entropy')
        self.model.fit(x_train, y_train)

        y_predict = self.model.predict(x_test)

        # Evaluate the model
        f1_score = metrics.f1_score(y_test, y_predict, average='macro')
        print("Local TEST", f1_score)

    def evaluate_model(self):
        """
        This function evaluates the model over the local test set (to calculate the final f1_score) as well as the
        test set from the competition, which targets is unknown. Finally, save the results for competition in the file
        "y_pred" in the data folder and the model.
        """

        # Save the results
        y_pred = self.model.predict(self.data_test)
        output = pd.DataFrame(columns=['test_index', 'pollutant'])
        output['pollutant'] = y_pred
        output['test_index'] = self.data_test.index

        output.to_csv('../data/y_pred.csv', header=True, index=False)
        output.to_json('../data/y_pred.json', orient="split")


sc = Schneider()
sc.preprocess_data()
sc.train_model()
sc.evaluate_model()
