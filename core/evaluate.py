"""Test script for ATDA."""

import torch.nn as nn
import os
import pandas
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import cohen_kappa_score
import torch
import torch.nn.functional as F
'''
Created on 4 Jun 2019

@author: xiwang
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
t = 0.5


def cal_metrics(path_pred, path_true, checkpoint):
    if not os.path.exists('Record'):
        os.mkdir('Record')
    w_id = open('Record/iter%d.txt' % (checkpoint), 'w')
    right_num = 0
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    prob_id = open(path_pred, 'r')
    gt_id = open(path_true, 'r')
    prob_lines = prob_id.readlines()
    gt_lines = gt_id.readlines()
    nums = len(prob_lines)
    print nums
    for i in xrange(nums):
        pred = (float(prob_lines[i].split(',')[-1]) > t)
        gt = float(gt_lines[i].split(',')[-1])
        if pred == gt:
            right_num += 1
            if gt == 0:
                TN += 1
            else:
                TP += 1
        else:
            if gt == 0:
                FP += 1
            else:
                FN += 1
    acc = (right_num * 1.0) / (nums * 1.0)
    sensitivity = (TP * 1.0) / (1.0 * (TP + FN))
    specificity = (TN * 1.0) / ((FP + TN) * 1.0)
    if TP + FP == 0:
        precision = 0
    else:
        precision = (TP * 1.0) / (1.0 * (TP + FP))
    if sensitivity + precision == 0:
        F1 = 0
    else:
        F1 = 2 * sensitivity * precision / (sensitivity + precision)

    from sklearn.metrics import roc_auc_score
    labels = np.loadtxt(path_true, delimiter=',')
    preds = np.loadtxt(path_pred, delimiter=',')
    auc = roc_auc_score(labels, preds)
    predictions = np.argmax(preds, axis=1)
    groundTruth = np.argmax(labels, axis=1)
    kappa = cohen_kappa_score(groundTruth, predictions)
    print 'Threshold:%.4f\tAccuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f' % (t, acc, sensitivity, specificity, precision, F1, auc, kappa)
    print 'TN: %d\t FN:%d\t TP: %d\t FP: %d\n' % (TN, FN, TP, FP)
    w_id.write('Threshold: %.4f\n' % (t))
    w_id.write('acc: %.4f\n' % (acc))
    w_id.write('sensitivity: %.4f\n' % (sensitivity))
    w_id.write('specificity: %.4f\n' % (specificity))
    w_id.write('TN: %d\n' % (TN))
    w_id.write('TP: %d\n' % (TP))
    w_id.write('FN: %d\n' % (FN))
    w_id.write('FP: %d\n' % (FP))
    prob_id.close()
    gt_id.close()
    w_id.write('AUC:%.4f' % (auc))
    w_id.close()
    return acc, sensitivity, specificity, precision, F1, auc, kappa, TN, FN, TP, FP


def cal_metrics_Volume_average(result_dir, path1, path2, testlist, checkpoint):
    volume_list = []
    name_id = open('%s/iter%d_VolName_avg.txt' %
                   (result_dir, checkpoint), 'w')
    for line in testlist:
        if '_C_' in line:
            prefix = line.split('/')[-1].split(',')[0].split('_C_')[0]
        elif '_H_' in line:
            prefix = line.split('/')[-1].split(',')[0].split('_H_')[0]
        if prefix not in volume_list:
            volume_list.append(prefix)
            name_id.write('%s\n' % (prefix))
    name_id.close()
    volume_pred_map = {}
    volume_label_map = {}
    for item in volume_list:
        volume_label_map[item] = []
        volume_pred_map[item] = []
    pred_r = open(path1)
    label_r = open(path2)
    name_lines = testlist
    pred_lines = pred_r.readlines()
    label_lines = label_r.readlines()
    pred_r.close()
    label_r.close()
    for i in xrange(len(name_lines)):
        line = name_lines[i]
        if '_C_' in line:
            prefix = line.split('/')[-1].split(',')[0].split('_C_')[0]
        elif '_H_' in line:
            prefix = line.split('/')[-1].split(',')[0].split('_H_')[0]
#         prefix = line.split('/')[-1].split(',')[0].split('_')[0]
        pred_ = [float(e) for e in pred_lines[i].split(',')]
#         print pred_
        label_ = [float(e) for e in label_lines[i].split(',')]
        volume_label_map[prefix].append(label_)
        volume_pred_map[prefix].append(pred_)
    volume_pred_id = open('%s/iter%d_VolPred_avg.txt' %
                          (result_dir, checkpoint), 'w')
    volume_label_id = open('%s/iter%d_VolLabel_avg.txt' %
                           (result_dir, checkpoint), 'w')
#     print volume_pred_map
    for item in volume_list:
        pred_l = volume_pred_map[item]
        label_l = volume_label_map[item]
        volPred = [1 - np.mean(np.array(pred_l)[:, 1]),
                   np.mean(np.array(pred_l)[:, 1])]
        volLabel = [1 - np.mean(np.array(label_l)[:, 1]),
                    np.mean(np.array(label_l)[:, 1])]
        volume_pred_id.write('%f,%f\n' % (1 - np.mean(np.array(pred_l)[:, 1]),
                                          np.mean(np.array(pred_l)[:, 1])))
        volume_label_id.write('%f,%f\n' % (1 - np.mean(np.array(label_l)[:, 1]),
                                           np.mean(np.array(label_l)[:, 1])))
    volume_pred_id.close()
    volume_label_id.close()
    path1 = '%s/iter%d_VolPred_avg.txt' % (result_dir, checkpoint)
    path2 = '%s/iter%d_VolLabel_avg.txt' % (result_dir, checkpoint)
    return cal_metrics(path1, path2, checkpoint)


def cal_metrics_Volume_presence(result_dir, path1, path2, testlist, checkpoint):
    volume_list = []
    name_id = open('%s/iter%d_VolName_max.txt' %
                   (result_dir, checkpoint), 'w')

    for line in testlist:
        if '_C_' in line:
            prefix = line.split('/')[-1].split(',')[0].split('_C_')[0]
        elif '_H_' in line:
            prefix = line.split('/')[-1].split(',')[0].split('_H_')[0]
#         prefix = line.split('/')[-1].split(',')[0].split('_')[0]
        if prefix not in volume_list:
            volume_list.append(prefix)
            name_id.write('%s\n' % (prefix))
    name_id.close()
    volume_pred_map = {}
    volume_label_map = {}
    print len(volume_list)
    for item in volume_list:
        volume_label_map[item] = []
        volume_pred_map[item] = []
    pred_r = open(path1)
    label_r = open(path2)
    name_lines = testlist
    pred_lines = pred_r.readlines()
    label_lines = label_r.readlines()
    pred_r.close()
    label_r.close()
    for i in xrange(len(name_lines)):
        line = name_lines[i]
        if '_C_' in line:
            prefix = line.split('/')[-1].split(',')[0].split('_C_')[0]
        elif '_H_' in line:
            prefix = line.split('/')[-1].split(',')[0].split('_H_')[0]
#         prefix = line.split('/')[-1].split(',')[0].split('_')[0]
#         print prefix
        pred_ = [float(e) for e in pred_lines[i].split(',')]
#         print pred_
        label_ = [float(e) for e in label_lines[i].split(',')]
        volume_label_map[prefix].append(label_)
        volume_pred_map[prefix].append(pred_)
    volume_pred_id = open('%s/iter%d_VolPred_max.txt' %
                          (result_dir, checkpoint), 'w')
    volume_label_id = open('%s/iter%d_VolLabel_max.txt' %
                           (result_dir, checkpoint), 'w')
#     print volume_pred_map
    for item in volume_list:
        pred_l = volume_pred_map[item]
        label_l = volume_label_map[item]
        volPred = [1 - np.max(np.array(pred_l)[:, 1]),
                   np.max(np.array(pred_l)[:, 1])]
        volLabel = [1 - np.max(np.array(label_l)[:, 1]),
                    np.max(np.array(label_l)[:, 1])]
        volume_pred_id.write('%f,%f\n' % (1 - np.max(np.array(pred_l)[:, 1]),
                                          np.max(np.array(pred_l)[:, 1])))
        volume_label_id.write('%f,%f\n' % (1 - np.max(np.array(label_l)[:, 1]),
                                           np.max(np.array(label_l)[:, 1])))
    volume_pred_id.close()
    volume_label_id.close()
    path1 = '%s/iter%d_VolPred_max.txt' % (result_dir, checkpoint)
    path2 = '%s/iter%d_VolLabel_max.txt' % (result_dir, checkpoint)
    return cal_metrics(path1, path2, checkpoint)


def validate(args, model, data_loader,  bar):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    model.eval()
    results_list = []
    labels_list = []
    # evaluate network
    stat_record = [0, 0, 0, 0]
    with torch.no_grad():
        for sample_batched in data_loader:
            bar.move()
            images = sample_batched['image']
            labels = sample_batched['label']

            # convert into torch.autograd.Variable
            images = images.cuda()
            labels = labels.cuda()

            preds = model(images)
            preds = F.softmax(preds)

            pred_cls = preds.data.cpu().numpy()
            label_cls = toOneHot(labels.data.cpu().numpy())

            results_list.extend(pred_cls)
            labels_list.extend(label_cls)
            stat = compute_acc(pred_cls, label_cls)
            stat_record = [stat_record[i] + stat[i] for i in range(4)]

            bar.log(job='Validating, acc = {}'.format(
                    str(stat_record)))

        results_arr = np.array(results_list)
        labels_arr = np.array(labels_list)

        acc, sensitivity, specificity, precision,  F1, auc, kappa, TN, FN, TP, FP = metrics(
            results_arr, labels_arr)
        return acc, sensitivity, specificity, precision,  F1, auc, kappa


def test(args, model, data_loader,  bar, testlist):
    """Evaluate classifier on source or target domains."""
    # set eval state for Dropout and BN layers
    model.eval()
    result_root = args.result_root
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    result_d1 = '%s/%s' % (result_root, args.net)
    if not os.path.exists(result_d1):
        os.mkdir(result_d1)
    result_d2 = '%s/%d' % (result_d1, args.checkpoint)
    if not os.path.exists(result_d2):
        os.mkdir(result_d2)
    results_list = []
    labels_list = []
    # evaluate network
    stat_record = [0, 0, 0, 0]
    with torch.no_grad():
        for sample_batched in data_loader:
            bar.move()
            images = sample_batched['image']
            labels = sample_batched['label']

            # convert into torch.autograd.Variable
            images = images.cuda()
            labels = labels.cuda()

            preds = model(images)
            preds = F.softmax(preds)

            pred_cls = preds.data.cpu().numpy()
            label_cls = toOneHot(labels.data.cpu().numpy())

            results_list.extend(pred_cls)
            labels_list.extend(label_cls)
            stat = compute_acc(pred_cls, label_cls)
            stat_record = [stat_record[i] + stat[i] for i in range(4)]

            bar.log(job='Testing, acc = {}'.format(
                    str(stat_record)))

        results_arr = np.array(results_list)
        labels_arr = np.array(labels_list)
        path1 = '%s/iter%d_pred.txt' % (
            result_d2, args.checkpoint)
        path2 = '%s/iter%d_label.txt' % (
            result_d2, args.checkpoint)

        pandas.DataFrame(np.array(results_arr)).to_csv(
            path1, header=None, index=None)
        pandas.DataFrame(np.array(labels_arr)).to_csv(
            path2, header=None, index=None)

        acc, sensitivity, specificity, precision,  F1, auc, kappa, TN, FN, TP, FP = metrics(
            results_arr, labels_arr)

        acc1, sensitivity1, specificity1, precision1,  F11, auc1, kappa1, TN1, FN1, TP1, FP1 = cal_metrics_Volume_average(result_d2, path1, path2,
                                                                                                                          testlist, args.checkpoint)
        acc2, sensitivity2, specificity2, precision2,  F12, auc2, kappa2, TN2, FN2, TP2, FP2 = cal_metrics_Volume_presence(result_d2, path1, path2,
                                                                                                                           testlist, args.checkpoint)
        w_id = open('%s/%d_metrics_all.csv' %
                    (result_d2, args.checkpoint), 'w')
        w_id.write(
            ',checkpoint,acc, sensitivity, specificity, precision,  F1, auc, kappa,TN, FN, TP, FP\n')
        w_id.write('image-level,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n' %
                   (args.checkpoint, acc, sensitivity, specificity, precision,  F1, auc, kappa, TN, FN, TP, FP))
        w_id.write('lesion-level(average),%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n' %
                   (args.checkpoint, acc1, sensitivity1, specificity1, precision1,  F11, auc1, kappa1, TN1, FN1, TP1, FP1))
        w_id.write('lesion-level(presence),%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n' %
                   (args.checkpoint, acc2, sensitivity2, specificity2, precision2,  F12, auc2, kappa2, TN2, FN2, TP2, FP2))
        w_id.close()
        return acc, sensitivity, specificity, precision,  F1, auc, kappa


def toOneHot(arr):
    arr_list = []
    for i in xrange(len(arr)):
        if arr[i] == 0:
            arr_list.append([1, 0])
        elif arr[i] == 1:
            arr_list.append([0, 1])
    return arr_list


def compute_acc(x, y):
    # print(x)
    # print(y)
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    false_indices = []
    if not isinstance(x, list):
        x = x.tolist()
    for i in range(len(x)):
        d = x[i]
        if not isinstance(x[i], list):
            d = x[i].tolist()
        idx = d.index(max(x[i]))
        if idx == 0:
            if y[i][idx] == 1:
                TN = TN + 1
            else:
                FN = FN + 1
                # false_indices.append(i)
        elif idx == 1:
            if y[i][idx] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
                # false_indices.append(i)

    # return [TP,FP,TN,FN], false_indices
    return [TP, FP, TN, FN]


def metrics(preds, labels):
    t_list = [0.5]
    for t in t_list:
        right_num = 0
        FP = 0
        FN = 0
        TP = 0
        TN = 0
        nums = len(preds)
        for i in xrange(nums):
            pred = preds[i][1] > t
            gt = labels[i][1]
            if pred == gt:
                right_num += 1
                if gt == 0:
                    TN += 1
                else:
                    TP += 1
            else:
                if gt == 0:
                    FP += 1
                else:
                    FN += 1
        acc = (right_num * 1.0) / (nums * 1.0)
        sensitivity = (TP * 1.0) / (1.0 * (TP + FN))
        specificity = (TN * 1.0) / ((FP + TN) * 1.0)
        predictions = np.argmax(preds, axis=1)
        groundTruth = np.argmax(labels, axis=1)
        kappa = cohen_kappa_score(groundTruth, predictions)
        if TP + FP == 0:
            precision = 0
        else:
            precision = (TP * 1.0) / (1.0 * (TP + FP))
        if sensitivity + precision == 0:
            F1 = 0
        else:
            F1 = 2 * sensitivity * precision / (sensitivity + precision)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, preds)
        print 'Threshold:%.4f\tAccuracy:%.4f\tSensitivity:%.4f\tSpecificity:%.4f\tPrecision:%.4f\tF1:%.4f\tAUC: %.4f\tKappa score:%.4f' % (t, acc, sensitivity, specificity, precision, F1, auc, kappa)
        print 'TN: %d\t FN:%d\t TP: %d\t FP: %d\n' % (TN, FN, TP, FP)
        return acc, sensitivity, specificity, precision,  F1, auc, kappa, TN, FN, TP, FP
