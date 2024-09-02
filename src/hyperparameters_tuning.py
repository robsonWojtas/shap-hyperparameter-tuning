from sklearn import datasets
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import OneSidedSelection
import random
import pickle


def increase_parameter(value, multiplier):
    return value * multiplier


def load_adult():
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
    y = final_balanced_dataset.income
    final_balanced_dataset.drop(labels=['income'], axis=1, inplace=True)
    x = final_balanced_dataset
    return x, y


def load_iris():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target)
    scaler = StandardScaler()
    sm = SMOTE()
    x_tmp = x.copy()
    x = pd.DataFrame(scaler.fit_transform(x_tmp), columns=x.columns)
    x, y = sm.fit_resample(x, y)
    return x, y


def load_wines():
    wine = datasets.load_wine()
    x = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = pd.DataFrame(wine.target)
    scaler = StandardScaler()
    sm = SMOTE()
    x_tmp = x.copy()
    x = pd.DataFrame(scaler.fit_transform(x_tmp), columns=x.columns)
    x, y = sm.fit_resample(x, y)
    return x, y


def load_breast_cancer():
    breast_cancer = datasets.load_breast_cancer()
    x = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = pd.DataFrame(breast_cancer.target)
    scaler = StandardScaler()
    sm = SMOTE()
    x_tmp = x.copy()
    x = pd.DataFrame(scaler.fit_transform(x_tmp), columns=x.columns)
    x, y = sm.fit_resample(x, y)
    return x, y


def load_diabetes():
    diabetes = pd.read_csv('../works/data/diabetes.csv')
    y = diabetes.Outcome
    diabetes.drop(labels=['Outcome'], axis=1, inplace=True)
    x = diabetes
    scaler = StandardScaler()
    sm = SMOTE()
    x_tmp = x.copy()
    x = pd.DataFrame(scaler.fit_transform(x_tmp), columns=x.columns)
    x, y = sm.fit_resample(x, y)
    return x, y


def load_california():
    x = pd.read_csv('../works/data/housing.csv')
    x.dropna(subset=['total_bedrooms'], inplace=True)
    x['median_house_value'] = np.where(x['median_house_value'] >= 179700, 1, 0)
    x.drop(labels=['ocean_proximity'], axis=1, inplace=True)
    y = x.median_house_value
    x.drop(labels=['median_house_value'], axis=1, inplace=True)
    return x, y


def get_random_parameters(parameters):
    rand_params = {}
    for key, val in parameters.items():
        rand_params[key] = val[random.randint(0, len(val) - 1)]
    return rand_params


if __name__ == '__main__':
    # {'mode': 'xgb', 'kernel': '', 'data': 'adult'}, {'mode': 'svm', 'kernel': 'linear', 'data': 'adult'},
    #         {'mode': 'svm', 'kernel': 'poly', 'data': 'adult'}, {'mode': 'svm', 'kernel': 'rbf', 'data': 'adult'},
    #         {'mode': 'svm', 'kernel': 'sigmoid', 'data': 'adult'},
    setups = [
        {'mode': 'svm', 'kernel': 'linear', 'data': 'california'},
        {'mode': 'svm', 'kernel': 'poly', 'data': 'california'},
        {'mode': 'svm', 'kernel': 'rbf', 'data': 'california'},
        {'mode': 'svm', 'kernel': 'sigmoid', 'data': 'california'},
        {'mode': 'xgb', 'kernel': '', 'data': 'california'}, ]

    for setup in setups:
        print('SETUP:', setup)

        mode = setup['mode']  # 'xgb'  # 'svm'
        dataset_name = setup['data']
        test_size = 0.2
        random_state = 0
        metrics_filename = 'metryki_' + setup['mode'] + '_' + setup['data'] + '_' + setup['kernel'] + '_random.png'
        tuning_filename = 'strojenie_' + setup['mode'] + '_' + setup['data'] + '_' + setup['kernel'] + '_random.png'
        best_filename = 'najlepsze' + setup['mode'] + '_' + setup['data'] + '_' + setup['kernel'] + '_random.png'

        # SVM parameters and presets
        kernel = setup['kernel']
        svm_kernels = [setup['kernel']]

        if setup['kernel'] == 'linear':
            svm_default_values = {'kernel': 'linear', 'tol': 0.001, 'max_iter': 10000, 'C': 5}
            svm_param_value_lists = {
                'C': [5, 4, 3, 2, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01],
                'max_iter': [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000],
                'tol': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.5]
            }
        if setup['kernel'] == 'poly':
            svm_default_values = {'kernel': 'poly', 'tol': 0.001, 'max_iter': 10000, 'C': 1, 'degree': 8, 'coef0': 0}
            svm_param_value_lists = {
                'degree': [8, 7, 6, 5, 4, 3, 2, 1],
                'coef0': [10, 5, 4, 3, 2, 1, 0.5, 0],
                'C': [0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6, 10],
                'max_iter': [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000],
                'tol': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.5]
            }
        if setup['kernel'] == 'rbf':
            svm_default_values = {'kernel': 'rbf', 'tol': 0.001, 'max_iter': 10000, 'C': 5}
            svm_param_value_lists = {
                'C': [5, 4, 3, 2, 1, 0.5, 0.2, 0.1, 0.01],
                'max_iter': [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000],
                'tol': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.5]
            }
        if setup['kernel'] == 'sigmoid':
            svm_default_values = {'kernel': 'sigmoid', 'tol': 0.001, 'max_iter': 10000, 'C': 1, 'coef0': 0}
            svm_param_value_lists = {
                'coef0': [0, 0.2, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5],
                'C': [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5],
                'max_iter': [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1500, 1000],
                'tol': [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2, 0.3, 0.5]
            }
        if mode == 'svm':
            svm_hyperparameters = list(svm_param_value_lists.keys())

        # XGB parameters and presets
        xgb_default_values = {'n_estimators': 300, 'learning_rate': 0.1, 'min_child_weight': 50, 'gamma': 20,
                              'max_depth': 2}
        xgb_param_value_lists = {
            'learning_rate': [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],  # , 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5],
            'n_estimators': [300, 400, 500, 600, 700, 800, 900, 1000],  # 1000, 900, 800, 700, 600, 500, 400, 300],  # ,
            'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            'min_child_weight': [50, 30, 20, 15, 10, 5, 4, 3, 2, 1, 0.5, 0.3, 0.1, 0],
            # 0, 0.1, 0.3, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 30, 50],
            # 30, 20, 15, 10, 5, 4, 3, 2, 1, 0.5, 0.3, 0.1, 0],
            'gamma': [20, 15, 10, 7, 5, 3, 1, 0],  # 0, 1, 3, 5, 7, 10, 15,
        }
        xgb_hyperparameters = list(xgb_param_value_lists.keys())

        if mode == 'svm':
            params = svm_hyperparameters
        elif mode == 'xgb':
            params = xgb_hyperparameters
        else:
            ValueError('Incorrect')

        results = []
        param_frame = []
        best_indices = []
        iteration = 0

        # dataset loading
        if dataset_name == 'iris':
            X, Y = load_iris()
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        elif dataset_name == 'wines':
            X, Y = load_wines()
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        elif dataset_name == 'diabetes':
            X, Y = load_diabetes()
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        elif dataset_name == 'breast cancer':
            X, Y = load_breast_cancer()
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        elif dataset_name == 'california':
            X, Y = load_california()
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        else:
            X, Y = load_adult()
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

        # learning loop
        if mode == 'svm':
            for k in svm_kernels:
                param_values = svm_default_values.copy()
                param_values['kernel'] = k
                for param in params:
                    print('SVM, parametr: ', param)
                    best_result = 0
                    best_result_index = 0
                    if len(svm_param_value_lists[param]) > 0:
                        for i in range(0, len(svm_param_value_lists[param])):

                            # choose param value
                            param_values[param] = svm_param_value_lists[param][i]

                            # model training
                            model = SVC(**param_values)
                            model.fit(x_train, y_train.values.ravel())

                            # result computing
                            predictions = model.predict(x_test)
                            result = accuracy_score(y_test, predictions)

                            # save results and used params
                            results.append(result)
                            param_frame.append(param_values.copy())

                            # save info about the best param value and its index
                            if result >= best_result:
                                best_result = result
                                best_result_index = i
                            iteration += 1
                        # keep the best value of a parameter
                        best_indices.append(best_result_index)
                        param_values[param] = svm_param_value_lists[param][best_result_index]
        else:
            param_values = xgb_default_values.copy()
            for param in params:
                print('XGB, parametr: ', param)
                best_result = 0
                best_result_index = 0
                if len(xgb_param_value_lists[param]) > 0:
                    for i in range(0, len(xgb_param_value_lists[param])):

                        # choose param value
                        param_values[param] = xgb_param_value_lists[param][i]

                        # model training
                        model = XGBClassifier(**param_values)
                        model.fit(x_train, y_train.values.ravel(), verbose=False)

                        # result computing
                        predictions = model.predict(x_test)
                        result = accuracy_score(y_test, predictions)

                        # save results and used params
                        results.append(result)
                        param_frame.append(param_values.copy())

                        # save info about the best param value and its index
                        if result >= best_result:
                            best_result = result
                            best_result_index = i
                        iteration += 1
                    # keep the best value of a parameter
                    best_indices.append(best_result_index)
                    param_values[param] = xgb_param_value_lists[param][best_result_index]

        param_frame = pd.DataFrame(param_frame)
        result_frame = pd.DataFrame(results)

        # compare with random search
        rand_search_results = []
        random_search_param_frame = []
        best_rand_result = 0
        best_rand_result_idx = 0
        if mode == 'svm':
            for i in range(0, iteration):
                params = get_random_parameters(svm_param_value_lists)
                rand_model = SVC(kernel=kernel, **params)
                rand_model.fit(x_train, y_train.values.ravel())
                # result computing
                rand_predictions = rand_model.predict(x_test)
                rand_result = accuracy_score(y_test, rand_predictions)

                # save results and used params
                rand_search_results.append(rand_result)
                random_search_param_frame.append(params)
                if rand_result >= best_rand_result:
                    best_rand_result = rand_result
                    random_search_best_params = params
                    best_rand_result_idx = i
        else:
            for i in range(0, iteration):
                params = get_random_parameters(xgb_param_value_lists)
                rand_model = XGBClassifier(**params)
                rand_model.fit(x_train, y_train.values.ravel())
                # result computing
                rand_predictions = rand_model.predict(x_test)
                rand_result = accuracy_score(y_test, rand_predictions)

                # save results and used params
                rand_search_results.append(rand_result)
                random_search_param_frame.append(params)
                if rand_result >= best_rand_result:
                    best_rand_result = rand_result
                    random_search_best_params = params
                    best_rand_result_idx = i

        # result plot
        plt.plot(list(range(0, iteration)), results, color='b', label='SHAP')
        plt.plot(list(range(0, iteration)), rand_search_results, color='r', label='Random')

        major_xticks = np.arange(0, iteration, 10)
        minor_xticks = np.arange(0, iteration, 1)

        major_yticks = np.arange(0, 1.1, 0.1)
        minor_yticks = np.arange(0, 1.1, 0.05)

        plt.xticks(major_xticks)
        plt.gca().xaxis.set_minor_locator(ticker.FixedLocator(minor_xticks))

        plt.yticks(major_yticks)
        plt.gca().yaxis.set_minor_locator(ticker.FixedLocator(minor_yticks))

        plt.grid(True, which='both')

        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        plt.xlabel("Iteracja")
        plt.ylabel("Celność", labelpad=5)
        plt.legend(loc='lower right')
        ax.tick_params(axis='y', pad=15)
        plt.savefig(tuning_filename, dpi=300)
        plt.show()

        # best result comparison
        shap_best = 0
        rand_best = 0
        best_shap_list = []
        best_rand_list = []
        for i in range(0, iteration):
            k = results[i]
            j = rand_search_results[i]
            if k > shap_best:
                shap_best = k
            if j > rand_best:
                rand_best = j
            best_shap_list.append(shap_best)
            best_rand_list.append(rand_best)

        # result plot
        plt.plot(list(range(0, iteration)), best_shap_list, color='b', label='SHAP')
        plt.plot(list(range(0, iteration)), best_rand_list, color='r', label='Random')

        major_xticks = np.arange(0, iteration, 10)
        minor_xticks = np.arange(0, iteration, 1)

        major_yticks = np.arange(0, 1.1, 0.1)
        minor_yticks = np.arange(0, 1.1, 0.05)

        plt.xticks(major_xticks)
        plt.gca().xaxis.set_minor_locator(ticker.FixedLocator(minor_xticks))

        plt.yticks(major_yticks)
        plt.gca().yaxis.set_minor_locator(ticker.FixedLocator(minor_yticks))

        plt.grid(True, which='both')

        ax = plt.gca()
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        plt.xlabel("Iteracja")
        plt.ylabel("Celność", labelpad=5)
        plt.legend(loc='lower right')
        ax.tick_params(axis='y', pad=15)
        plt.savefig(best_filename, dpi=300)
        plt.show()

        # tuning validation loop
        optimal_accuracies = []
        random_search_accuracies = []
        default_accuracies = []
        optimal_f1scores = []
        random_search_f1scores = []
        default_f1scores = []
        optimal_recalls = []
        random_search_recalls = []
        default_recalls = []
        optimal_precisions = []
        random_search_precisions = []
        default_precisions = []

        num_of_checks = 50
        k = list(range(0, num_of_checks))
        for i in range(0, num_of_checks):
            print('Iteracja: ', i)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

            if mode == 'svm':
                optimal_model = SVC(**param_values)
                random_model = SVC(kernel=kernel, **random_search_best_params)
                default_model = SVC(kernel=kernel, max_iter=param_values['max_iter'])

                optimal_model.fit(x_train, y_train.values.ravel())
                random_model.fit(x_train, y_train.values.ravel())
                default_model.fit(x_train, y_train.values.ravel())

                optimal_predictions = optimal_model.predict(x_test)
                random_predictions = random_model.predict(x_test)
                default_predictions = default_model.predict(x_test)

                optimal_accuracies.append(accuracy_score(y_test, optimal_predictions))
                random_search_accuracies.append(accuracy_score(y_test, random_predictions))
                default_accuracies.append(accuracy_score(y_test, default_predictions))

                optimal_f1scores.append(f1_score(y_test, optimal_predictions))
                random_search_f1scores.append(f1_score(y_test, random_predictions))
                default_f1scores.append(f1_score(y_test, default_predictions))

                optimal_recalls.append(recall_score(y_test, optimal_predictions))
                random_search_recalls.append(recall_score(y_test, random_predictions))
                default_recalls.append(recall_score(y_test, default_predictions))

                optimal_precisions.append(precision_score(y_test, optimal_predictions))
                random_search_precisions.append(precision_score(y_test, random_predictions))
                default_precisions.append(precision_score(y_test, default_predictions))
            else:
                optimal_model = XGBClassifier(**param_values)
                random_model = XGBClassifier(**random_search_best_params)
                default_model = XGBClassifier()

                optimal_model.fit(x_train, y_train.values.ravel())
                random_model.fit(x_train, y_train.values.ravel())
                default_model.fit(x_train, y_train.values.ravel())

                optimal_predictions = optimal_model.predict(x_test)
                random_predictions = random_model.predict(x_test)
                default_predictions = default_model.predict(x_test)

                optimal_accuracies.append(accuracy_score(y_test, optimal_predictions))
                random_search_accuracies.append(accuracy_score(y_test, random_predictions))
                default_accuracies.append(accuracy_score(y_test, default_predictions))

                optimal_f1scores.append(f1_score(y_test, optimal_predictions))
                random_search_f1scores.append(f1_score(y_test, random_predictions))
                default_f1scores.append(f1_score(y_test, default_predictions))

                optimal_recalls.append(recall_score(y_test, optimal_predictions))
                random_search_recalls.append(recall_score(y_test, random_predictions))
                default_recalls.append(recall_score(y_test, default_predictions))

                optimal_precisions.append(precision_score(y_test, optimal_predictions))
                random_search_precisions.append(precision_score(y_test, random_predictions))
                default_precisions.append(precision_score(y_test, default_predictions))

        # plots
        accuracies = [optimal_accuracies, random_search_accuracies, default_accuracies]
        f1scores = [optimal_f1scores, random_search_f1scores, default_f1scores]
        recalls = [optimal_recalls, random_search_recalls, default_recalls]
        precisions = [optimal_precisions, random_search_precisions, default_precisions]

        # box plots here...
        fig, axs = plt.subplots(2, 2)
        labels = ['SHAP', 'Losowe', 'Domyślne']
        bplot0 = axs[0, 0].boxplot(accuracies, vert=True, labels=labels, patch_artist=True)
        axs[0, 0].set_title('Celność')
        axs[0, 0].yaxis.grid(True)
        bplot1 = axs[0, 1].boxplot(f1scores, vert=True, labels=labels, patch_artist=True)
        axs[0, 1].set_title('Miara F1')
        axs[0, 1].yaxis.grid(True)
        bplot2 = axs[1, 0].boxplot(recalls, vert=True, labels=labels, patch_artist=True)
        axs[1, 0].set_title('Czułość')
        axs[1, 0].yaxis.grid(True)
        bplot3 = axs[1, 1].boxplot(precisions, vert=True, labels=labels, patch_artist=True)
        axs[1, 1].set_title('Precyzja')
        axs[1, 1].yaxis.grid(True)
        plt.tight_layout()
        plt.savefig(metrics_filename, dpi=300)
        plt.show()

        #############################################################
        #     SAVING USEFUL VARIABLES     ###########################
        #############################################################
        with open('workspaces/' + setup['mode'] + '_' + setup['data'] + '_' + setup['kernel'] + '_workspace.pkl',
                  'wb') as file:
            pickle.dump(
                (results,  # 0
                 param_values,  # 1
                 random_search_best_params,  # 2
                 rand_search_results,  # 3
                 param_frame,  # 4
                 random_search_param_frame,  # 5
                 iteration,  # 6
                 best_result,  # 7
                 best_result_index,  # 8
                 best_rand_result,  # 9
                 best_rand_result_idx,  # 10
                 accuracies,  # 11
                 f1scores,  # 12
                 recalls,  # 13
                 precisions,  # 14
                 best_shap_list,  # 15
                 best_rand_list), file)  # 16

# #############################################################
# #     GENERATE PLOTS WITH RANGES ON THEM     ################
# #############################################################
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
#
# setup = {'mode': 'svm', 'kernel': 'linear', 'data': 'adult'}
# tuning_filename = 'strojenie_' + setup['mode'] + '_' + setup['data'] + '_' + setup['kernel'] + '_random_ranges.png'
# best_filename = 'najlepsze' + setup['mode'] + '_' + setup['data'] + '_' + setup['kernel'] + '_random_ranges.png'
#
# with open("./masters_utils/workspaces/xgb_adult_workspace.pkl", 'rb') as f:
#     data = pickle.load(f)
# results = data[0]
# rand_search_results = data[3]
# iteration = data[6]
# best_shap_list = data[15]
# best_rand_list = data[16]
# plt.plot(list(range(0, iteration)), results, color='b', label='SHAP')
# plt.plot(list(range(0, iteration)), rand_search_results, color='r', label='Random')
# major_xticks = np.arange(0, iteration, 10)
# minor_xticks = np.arange(0, iteration, 1)
#
# major_yticks = np.arange(0, 1.1, 0.1)
# minor_yticks = np.arange(0, 1.1, 0.05)
#
# plt.xticks(major_xticks)
# plt.gca().xaxis.set_minor_locator(ticker.FixedLocator(minor_xticks))
#
# plt.yticks(major_yticks)
# plt.gca().yaxis.set_minor_locator(ticker.FixedLocator(minor_yticks))
#
# plt.grid(True, which='both')
#
# ax = plt.gca()
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['bottom'].set_position('zero')
# plt.xlabel("Iteracja")
# plt.ylabel("Celność", labelpad=5)
# plt.legend(loc='lower right')
# ax.tick_params(axis='y', pad=15)
#
# XGB
# plt.axvspan(0,7, color='blue', alpha=0.05)
# plt.axvspan(7,15, color='blue', alpha=0.15)
# plt.axvspan(15,27, color='blue', alpha=0.05)
# plt.axvspan(27,41, color='blue', alpha=0.15)
# plt.axvspan(41,48, color='blue', alpha=0.05)
#
# ax.text(3, 0.05,'learning_rate', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(11, 0.05,'n_estimators', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(21, 0.05,'max_depth', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(34, 0.05,'min_child_weight', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(45, 0.2,'gamma', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)

# linear
# plt.axvspan(0,10, color='blue', alpha=0.05)
# plt.axvspan(10,21, color='blue', alpha=0.15)
# plt.axvspan(21,28, color='blue', alpha=0.05)
# ax.text(3, 0.05,'C', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(11, 0.05,'max_iter', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(21, 0.2,'tol', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)

# poly
# plt.axvspan(0,7, color='blue', alpha=0.05)
# plt.axvspan(7,15, color='blue', alpha=0.15)
# plt.axvspan(15,25, color='blue', alpha=0.05)
# plt.axvspan(25,36, color='blue', alpha=0.15)
# plt.axvspan(36,43, color='blue', alpha=0.05)
# ax.text(3, 0.05,'degree', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(11, 0.05,'coef0', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(20, 0.05,'C', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(31, 0.05,'max_iter', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(39, 0.2,'tol', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)

# rbf
# plt.axvspan(0,8, color='blue', alpha=0.05)
# plt.axvspan(8,19, color='blue', alpha=0.15)
# plt.axvspan(19,27, color='blue', alpha=0.05)
# ax.text(4, 0.05,'C', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(13, 0.05,'max_iter', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(23, 0.2,'tol', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)

# sigmoid
# plt.axvspan(0,9, color='blue', alpha=0.05)
# plt.axvspan(9,20, color='blue', alpha=0.15)
# plt.axvspan(20,31, color='blue', alpha=0.05)
# plt.axvspan(31,39, color='blue', alpha=0.15)
# ax.text(4, 0.05,'coef0', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(14, 0.05,'C', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(25, 0.05,'max_iter', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)
# ax.text(35, 0.2,'tol', ha='left', va='bottom', rotation='vertical', color='blue', alpha=0.5)


# plt.savefig(tuning_filename, dpi=300)

# plt.show()