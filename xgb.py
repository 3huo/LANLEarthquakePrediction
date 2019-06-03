# -*- coding: utf-8 -*-
import argparse
import time
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataloader import ActionData
from models import EmbedNet,Base_model,GhostVLAD_layer,Resnet_VLAD,Final_model
import logging
import os
import numpy as np
import xgboost as xgb

def main():

  params = {#'num_leaves': 32,
      #'min_data_in_leaf': 79,
      #'objective': 'gamma',
      'max_depth': 7,
      #'learning_rate': 0.01,
      #"boosting": "gbdt",
      #"bagging_freq": 5,
      #"bagging_fraction": 0.8126672064208567,
      #"bagging_seed": 11,
      #"metric": 'mae',
      #"verbosity": -1,
      #'reg_alpha': 0.1302650970728192,
      #'reg_lambda': 0.3603427518866501,
      #'feature_fraction': 0.1
      'tree_method':'approx',
      'subsample': 0.6,
      'eval_matric':'merror',
      'objective':'multi:softmax',
      'lambda':1,
      'eta':0.1,
      'silent':1,
      'num_class': 5,
      'nthread':8,
      'seed': 42
     }

  train_f = np.load('/data/wangyancheng/ActionReconition/train_feature_model_33.npy')
  train_l = np.load('/data/wangyancheng/ActionReconition/train_label_model_33.npy')
  test_f = np.load('/data/wangyancheng/ActionReconition/test_feature_model_33.npy')
  test_l = np.load('/data/wangyancheng/ActionReconition/test_label_model_33.npy')


  # train_f = np.transpose(train_f)
  # test_f = np.transpose(test_f)
  train_data = xgb.DMatrix(data=train_f, label=train_l)
  valid_data = xgb.DMatrix(data=test_f, label=test_l)
  print(train_f.shape)
  print(train_l.shape)
  print(test_f.shape)
  print(test_l.shape)
  watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
  model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=1, params=params)
  #model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, params=params)
  #model = xgb.train(params, train_data, 1)
  model.save_model('002.model')
  y_pred_valid = model.predict(xgb.DMatrix(test_f), ntree_limit=model.best_ntree_limit)
  #y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)
      #         pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}:".format(
#             process_time/(process_time+prepare_time), epoch, opt.epochs))
  acc = (y_pred_valid == test_l).sum()/len(y_pred_valid)
  print(y_pred_valid)
  print(test_l)
  print(acc)
if __name__ == '__main__':
    main()
        
        