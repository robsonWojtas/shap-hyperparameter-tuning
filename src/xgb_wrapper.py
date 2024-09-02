import numpy as np
from xgboost import XGBClassifier
from masters_utils import get_params_dataset
import pandas as pd
import shap
from shap.maskers import Independent
from sklearn.model_selection import train_test_split
from threading import Thread
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import OneSidedSelection
from imblearn.over_sampling import SMOTE
from sklearn import datasets
import matplotlib.pyplot as plt
import pickle


class XGBWrapper:
    results = {}

    def __call__(self, x, *args, **kwargs):
        print('XGBWrapper called')
        self.results = {}
        threads = []
        for idx, chunk in enumerate(np.array_split(x, 6)):
            thread = Thread(target=self.get_model_results, args=(chunk, idx))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        results = []
        for v in self.results.values():
            results = results + v

        return pd.DataFrame(np.array(results), columns=['y'])

    def __init__(self, x_train, y_train, x_val, y_val, param_names, int_params):
        print('init XGBWrapper')
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.param_names = param_names
        self.int_params = int_params

    def fit(self, params):
        params = self.params_to_dict(params)
        model = XGBClassifier(**params)
        model.fit(self.x_train, self.y_train, verbose=False)
        return model

    def predict(self, model):
        print('predict output')
        return model.predict(self.x_val)

    def params_to_dict(self, row):
        params_dict = dict(zip(self.param_names, row))
        for param in self.int_params:
            params_dict[param] = int(params_dict[param])
        return params_dict

    def get_model_accuracy(self, model):
        predictions = model.predict(self.x_val)
        return accuracy_score(self.y_val, predictions)

    def get_model_results(self, x, thread_number):
        self.results[thread_number] = []
        x_len = len(x) - 1
        counter = 0
        for row in x.values:
            model = self.fit(row)
            result = self.get_model_accuracy(model)
            self.results[thread_number].append(result)
            print('Thread no. ', thread_number, ' Iteration: ', counter, 'of', x_len, ': ', result)
            counter += 1


def get_prepared_adult_data():
    dataset = pd.read_csv('../works/data/adult.csv')
    dataset['income'] = dataset['income'].map({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K': 1})
    dataset['workclass'] = dataset['workclass'].replace(['?'], 'Unknown')
    dataset['marital-status'] = dataset['marital-status'].replace(
        ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
    dataset['marital-status'] = dataset['marital-status'].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'],
                                                                  'Single')
    dataset['marital-status'] = dataset['marital-status'].map({'Married': 0, 'Single': 1})
    dataset['marital-status'] = dataset['marital-status']
    dataset.drop(labels=['gender', 'workclass', 'education', 'occupation', 'relationship', 'race', 'native-country'],
                 axis=1, inplace=True)
    return dataset


def get_balanced_adult_data():
    dataset = pd.read_csv('../works/data/adult.csv')
    encoder = LabelEncoder()
    dataset['income'] = encoder.fit_transform(dataset['income'])
    dataset['native-country'] = np.where(dataset['native-country'] == 'United-States', 1, 0)
    dataset['marital-status'] = dataset['marital-status'].replace(
        {' Married-civ-spouse': 'Married', ' Never-married': 'Single',
         ' Separated': 'Divorced', ' Married-spouse-absent': 'Divorced',
         ' Divorced': 'Divorced',
         ' Married-AF-spouse': 'Divorced', ' Widowed': 'Widowed'})
    dataset['workclass'] = np.where(dataset['workclass'] == 'Private', 1, 0)
    dataset['gender'] = np.where(dataset['gender'] == 'Male', 1, 0)
    dataset['race'] = np.where(dataset['race'] == 'White', 1, 0)
    education_mapping = {'Preschool': 0, '1st-4th': 1, '5th-6th': 2, '7th-8th': 3, '9th': 4, '10th': 5,
                         '11th': 6, '12th': 7, 'HS-grad': 8, 'Some-college': 0, 'Assoc-acdm': 10,
                         'Assoc-voc': 11, 'Bachelors': 12, 'Prof-school': 13, 'Masters': 14, 'Doctorate': 15
                         }
    dataset['education'] = dataset['education'].map(education_mapping)
    relationship_ordered = dataset.groupby(['relationship'])['income'].count().sort_values().index
    relationship_ordered = {k: i for i, k in enumerate(relationship_ordered, 0)}
    dataset['relationship'] = dataset['relationship'].map(relationship_ordered)
    occupation_ordered = dataset.groupby(['occupation'])['income'].count().sort_values().index
    occupation_ordered = {k: i for i, k in enumerate(occupation_ordered, 0)}
    dataset['occupation'] = dataset['occupation'].map(occupation_ordered)
    marital_ordered = dataset.groupby(['marital-status'])['income'].count().sort_values().index
    marital_ordered = {k: i for i, k in enumerate(marital_ordered, 0)}
    dataset['marital-status'] = dataset['marital-status'].map(marital_ordered)
    dataset.drop('fnlwgt', axis=1, inplace=True)  # it is not a useful feature for predicting the wage class
    scaler = StandardScaler()
    scaled_features_balanced_dataset = scaler.fit_transform(dataset.drop('income', axis=1))
    scaled_features_balanced_dataset = pd.DataFrame(scaled_features_balanced_dataset,
                                                    columns=dataset.drop('income', axis=1).columns)
    # undersampling the train set
    under = OneSidedSelection()
    x_train_res, y_train_res = under.fit_resample(scaled_features_balanced_dataset, dataset['income'])

    # oversampling the train set
    sm = SMOTE()
    x_train_res, y_train_res = sm.fit_resample(x_train_res, y_train_res)

    x_train_res = pd.DataFrame(x_train_res, columns=dataset.drop('income', axis=1).columns)

    # creating the final train
    final_balanced_dataset = pd.concat([x_train_res, y_train_res], axis=1)
    return final_balanced_dataset


if __name__ == '__main__':
    # # default_params = [200, 0.3, 6.0, 1, 0]
    # #############################################################
    # #     ADULT INCOME DATASET     ##############################
    # #############################################################
    # #############################################################
    # #     NOT BALANCED     ######################################
    # #############################################################
    # # numeric_columns = ['marital-status', 'age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss',
    # #                    'hours-per-week']
    # # data = get_prepared_adult_data()
    # # X = data[numeric_columns]
    # # Y = data.income
    # 
    # #############################################################
    # #     BALANCED         ######################################
    # #############################################################
    # # data = get_balanced_adult_data()
    # # Y = data.income
    # # data.drop(labels=['income'], axis=1, inplace=True)
    # # X = data
    # 
    # #############################################################
    # #     IRIS DATASET     ######################################
    # #############################################################
    # # iris = datasets.load_iris()
    # # X = pd.DataFrame(iris.data, columns=iris.feature_names)
    # # Y = pd.DataFrame(iris.target)
    # 
    # #############################################################
    # #     WINES DATASET     ######################################
    # #############################################################
    # # wine = datasets.load_wine()
    # # X = pd.DataFrame(wine.data, columns=wine.feature_names)
    # # Y = pd.DataFrame(wine.target)
    # 
    # #############################################################
    # #     DIABETES DATASET     ##################################
    # #############################################################
    # # diabetes = pd.read_csv('../works/data/diabetes.csv')
    # # Y = diabetes.Outcome
    # # diabetes.drop(labels=['Outcome'], axis=1, inplace=True)
    # # X = diabetes
    # 
    # #############################################################
    # #     BREAST CANCER DATASET     #############################
    # #############################################################
    # breast_cancer = datasets.load_wine()
    # X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    # Y = pd.DataFrame(breast_cancer.target)
    # 
    # #############################################################
    # #     PARAMS PREPARATION     ################################
    # #############################################################
    # columns = ['n_estimators',
    #            'learning_rate',
    #            'max_depth',
    #            'min_child_weight',
    #            'min_split_loss'
    #            ]
    # #
    # # 'reg_lambda'
    # # 'gamma',
    # int_columns = ['n_estimators', 'max_depth']
    # 
    # # pierwszy zestaw danych
    # # n_estimators = [1, 10, 50, 100, 200, 300, 500, 1000]
    # # learning_rate = [0.000001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 1]
    # # max_depth = [1, 2, 3, 4, 5, 6, 7, 8]
    # # min_child_weight = [0.0001, 0.01, 0.1, 1, 3, 10, 50, 100]
    # # min_split_loss = [0, 0.0001, 0.01, 0.1, 1, 10, 50]
    # # lam = [1, 1.5, 3]
    # 
    # # drugi zestaw danych
    # n_estimators = [1, 100, 300]
    # learning_rate = [0.000001, 0.3, 1]
    # max_depth = [1, 10, 20]
    # min_child_weight = [0.1, 5, 10]
    # min_split_loss = [0, 1, 10]
    # lam = [1, 1.5, 3]
    # 
    # #############################################################
    # #     EXPERIMENTS     #######################################
    # #############################################################
    # train_X, val_X, train_y, val_y = train_test_split(X, Y, test_size=0.3, random_state=42)
    # 
    # wrapper = XGBWrapper(x_train=train_X, y_train=train_y, x_val=val_X, y_val=val_y,
    #                      param_names=columns, int_params=int_columns)
    # 
    # input_data = get_params_dataset([n_estimators, learning_rate, max_depth, min_child_weight, min_split_loss], columns)
    # # masker = shap.maskers.Independent(input_data)
    # masker = shap.maskers.Independent(input_data, max_samples=10)  # zmniejszenie ilości obliczeń po stronie maskera
    # explainer = shap.Explainer(wrapper, masker, feature_names=columns)
    # shap_values = explainer(input_data)
    # 
    # #############################################################
    # #     RESULT PLOT     #######################################
    # #############################################################
    # shap_values.base_values = shap_values.base_values[0]
    # shap_values.base_values = shap_values.base_values[0]
    # fig = plt.figure()
    # shap.plots.waterfall(shap_values[0], show=False)
    # plt.gcf().set_size_inches(18, 7)
    # plt.show()

    #############################################################
    #     EXPERIMENTS - MULTIPLE RUN AND RANKINGS    ############
    #############################################################

    #############################################################
    #     PARAMS PREPARATION     ################################
    #############################################################
    columns = ['n_estimators',
               'learning_rate',
               'max_depth',
               'min_child_weight',
               'min_split_loss'
               ]
    #
    # 'reg_lambda'
    # 'gamma',
    int_columns = ['n_estimators', 'max_depth']

    # pierwszy zestaw danych
    # n_estimators = [1, 10, 50, 100, 200, 300, 500, 1000]
    # learning_rate = [0.000001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.8, 1]
    # max_depth = [1, 2, 3, 4, 5, 6, 7, 8]
    # min_child_weight = [0.0001, 0.01, 0.1, 1, 3, 10, 50, 100]
    # min_split_loss = [0, 0.0001, 0.01, 0.1, 1, 10, 50]
    # lam = [1, 1.5, 3]

    # drugi zestaw danych
    n_estimators = [1, 100, 300]
    learning_rate = [0.000001, 0.3, 1]
    max_depth = [1, 10, 20]
    min_child_weight = [0.1, 5, 10]
    min_split_loss = [0, 1, 10]
    lam = [1, 1.5, 3]

    input_data = get_params_dataset([n_estimators, learning_rate, max_depth, min_child_weight, min_split_loss], columns)
    masker = shap.maskers.Independent(input_data, max_samples=20)  # zmniejszenie ilości obliczeń po stronie maskera

    scaler = StandardScaler()
    sm = SMOTE()
    #############################################################
    #     IRIS DATASET     ######################################
    #############################################################
    iris = datasets.load_iris()
    X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
    Y_iris = pd.DataFrame(iris.target)
    X_iris_tmp = X_iris.copy()
    X_iris = pd.DataFrame(scaler.fit_transform(X_iris_tmp), columns=X_iris.columns)
    X_iris, Y_iris = sm.fit_resample(X_iris, Y_iris)

    train_X, val_X, train_y, val_y = train_test_split(X_iris, Y_iris, test_size=0.2, random_state=0)
    wrapper_iris = XGBWrapper(x_train=train_X, y_train=train_y, x_val=val_X, y_val=val_y,
                              param_names=columns, int_params=int_columns)
    explainer_iris = shap.Explainer(wrapper_iris, masker, feature_names=columns)
    shap_values_iris = explainer_iris(input_data)

    #############################################################
    #     WINES DATASET     ######################################
    #############################################################
    wine = datasets.load_wine()
    X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
    Y_wine = pd.DataFrame(wine.target)
    X_wine_tmp = X_wine.copy()
    X_wine = pd.DataFrame(scaler.fit_transform(X_wine_tmp), columns=X_wine.columns)
    X_wine, Y_wine = sm.fit_resample(X_wine, Y_wine)

    train_X, val_X, train_y, val_y = train_test_split(X_wine, Y_wine, test_size=0.2, random_state=0)
    wrapper_wine = XGBWrapper(x_train=train_X, y_train=train_y, x_val=val_X, y_val=val_y,
                              param_names=columns, int_params=int_columns)
    explainer_wine = shap.Explainer(wrapper_wine, masker, feature_names=columns)
    shap_values_wine = explainer_wine(input_data)

    #############################################################
    #     DIABETES DATASET     ##################################
    #############################################################
    diabetes = pd.read_csv('../works/data/diabetes.csv')
    Y_diabetes = diabetes.Outcome
    diabetes.drop(labels=['Outcome'], axis=1, inplace=True)
    X_diabetes = diabetes
    X_diabetes_tmp = X_diabetes.copy()
    X_diabetes = pd.DataFrame(scaler.fit_transform(X_diabetes_tmp), columns=X_diabetes.columns)
    X_diabetes, Y_diabetes = sm.fit_resample(X_diabetes, Y_diabetes)

    train_X, val_X, train_y, val_y = train_test_split(X_diabetes, Y_diabetes, test_size=0.2, random_state=0)
    wrapper_diabetes = XGBWrapper(x_train=train_X, y_train=train_y, x_val=val_X, y_val=val_y,
                                  param_names=columns, int_params=int_columns)
    explainer_diabetes = shap.Explainer(wrapper_diabetes, masker, feature_names=columns)
    shap_values_diabetes = explainer_diabetes(input_data)

    #############################################################
    #     BREAST CANCER DATASET     #############################
    #############################################################
    breast_cancer = datasets.load_breast_cancer()
    X_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    Y_bc = pd.DataFrame(breast_cancer.target)
    X_bc_tmp = X_bc.copy()
    X_bc = pd.DataFrame(scaler.fit_transform(X_bc_tmp), columns=X_bc.columns)
    X_bc, Y_bc = sm.fit_resample(X_bc, Y_bc)

    train_X, val_X, train_y, val_y = train_test_split(X_bc, Y_bc, test_size=0.2, random_state=0)
    wrapper_bc = XGBWrapper(x_train=train_X, y_train=train_y, x_val=val_X, y_val=val_y,
                            param_names=columns, int_params=int_columns)
    explainer_bc = shap.Explainer(wrapper_bc, masker, feature_names=columns)
    shap_values_bc = explainer_bc(input_data)

    shap_values_std_array = [np.std(shap_values_iris.values, axis=0), np.std(shap_values_wine.values, axis=0),
                             np.std(shap_values_diabetes.values, axis=0), np.std(shap_values_bc.values, axis=0)]

    shap_values_std_df = pd.DataFrame(shap_values_std_array, columns=columns)
    mean_values = shap_values_std_df.mean()
    rank = mean_values.sort_values(ascending=False).index.tolist()
    #############################################################
    #     SAVING USEFUL VARIABLES     ###########################
    #############################################################
    with open('workspaces/xgb_ranking.pkl', 'wb') as file:
        pickle.dump(
            (shap_values_iris, shap_values_wine, shap_values_diabetes, shap_values_bc, shap_values_std_df, rank), file)
