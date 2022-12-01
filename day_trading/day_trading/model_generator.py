#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 15:52:52 2022
@author: Juan Beleño
"""
import warnings
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')


class DayTradingModelGenerator:
    def catboost_objective(self, trial: optuna.Trial, train_data, train_labels):
        param = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
            'depth': trial.suggest_int('depth', 3, 9, 1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 1e0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
            'random_state': trial.suggest_categorical('random_state', [1]),
            'loss_function': trial.suggest_categorical('loss_function', ['CrossEntropy']),
            'eval_metric': trial.suggest_categorical('eval_metric', ['Precision']),
            'silent': trial.suggest_categorical('silent', [True]),
            'early_stopping_rounds': trial.suggest_categorical('early_stopping_rounds', [100])
        }

        # Conditional Hyper-Parameters
        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float(
                "bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        model = CatBoostClassifier(**param)
        cv = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
        return cross_val_score(model, train_data, train_labels, cv=cv, scoring='precision', error_score=0.0).mean()

    def tree_objective(self, trial: optuna.Trial, train_data, train_labels):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 64)
        }
        model = DecisionTreeClassifier(**params)
        cv = StratifiedKFold(n_splits=3, random_state=1, shuffle=True)
        return cross_val_score(model, train_data, train_labels, cv=cv, scoring='precision', error_score=0.0).mean()

    def get_best_model(self, train_data, train_labels):
        # Wrap the objective inside a lambda and call objective inside it
        def func(trial): return self.tree_objective(
            trial, train_data, train_labels)

        # Pass func to Optuna studies
        study = optuna.create_study(direction='maximize')
        study.optimize(func, n_trials=30)

        # Best parameters
        best_params = study.best_trial.params
        '''
        best_params = {'learning_rate': 0.09898706607536945, 'depth': 6, 'l2_leaf_reg': 0.3860948723188682, 'colsample_bylevel': 0.07482814420154311, 'boosting_type': 'Plain',
                       'bootstrap_type': 'MVS', 'min_data_in_leaf': 11, 'one_hot_max_size': 3, 'random_state': 1, 'loss_function': 'CrossEntropy', 'eval_metric': 'Precision', 'silent': True, 'early_stopping_rounds': 100}
        '''
        print(f'Best params: {best_params}')

        # Train best model
        best_model = DecisionTreeClassifier(**best_params)
        best_model.fit(train_data, train_labels)

        return best_model
