import argparse
import json
import os
import sys
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
from evaluation.metrics import Config, DetectionMetricDataList, DetectionMetrics
from evaluation.functions import (accumulate, calc_ap, calc_tp, class_pr_curve,
                       class_tp_curve, dist_pr_curve, summary_plot)

TP_METRICS = ['trans_err', 'scale_err', 'orient_err'] 

class Evaluate():
    def __init__(self, dataset_split='val', config_file="tools/config.json" ):
        
        with open(config_file, 'r') as f:
            data = json.load(f)
        self.config = Config.deserialize(data)
        
        self.count = 0
        self.dataset_split = dataset_split
        self.dets = []
        self.predictions = {}
        self.gts = {}
        self.output_dir = "/export/amsvenkat/project/3d-multi-agent-tracking/SMOKE/logs/logs_v8.1/eval/"
        # self.config = None
        # self.plot_dir = plot_dir
        # self.preds_path = preds_path
        # self.gts_path = gts_path
        self.plot_dir = self.config.plot_path
        self.preds_path = self.config.preds_path
        self.gts_path = self.config.gts_path
        # self.dataset_split = self.config.dataset_split

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_predictions(self, index):
        preds =  defaultdict(list)
        
        for i, index in enumerate(index):
            # result where the predictions are stored
            with open(str(self.preds_path) + str(index) + ".txt", 'r') as f:
                list_items = []
                lines = f.readlines()
                for line in lines:
                    line = line.split(' ')
                    line = [ float(i) if i not in ['Car', 'Pedestrian' , 'Cyclist', 'DontCare'] else str(i)  for i in line ]
                    dictionary = [{'sample_frame': index, 'class_name': line[0], 'dimension': [line[8], line[9], line[10]],
                                  'location_cam': [line[11], line[12], line[13]], 'rotation': line[14], 'score': line[15]}]
                    list_items.extend(dictionary)
            preds[index].extend(list_items)
        #print(preds)
        return preds

    def load_gts(self, index):
        gts = defaultdict(list)
        #print(str(index).lstrip('0')) 

        for i, index in enumerate(index):
            t =str(index).lstrip('0') 
            # label files where the ground truth is stored
            with open(str(self.gts_path) +str(t) + ".txt", 'r') as f:
                list_items = []
                lines = f.readlines()
                for line in lines:
                    line = line.split(' ')
                    line = [ float(i) if i not in ['Car', 'Pedestrian'] else str(i)  for i in line ]
                    dictionary = [{'sample_frame': t ,'class_name': line[0], 'dimension': [line[8], line[9], line[10]],
                                  'location_cam': [line[11], line[12], line[13]], 'rotation': line[14], 'score': line[15]}]
                    list_items.extend(dictionary)
            gts[t].extend(list_items)
        return gts

    def get_count_of_dataset(self):
        #folder_path = "/export/amsvenkat/project/3d-multi-agent-tracking/data/train_v5/ImageSets/"
        folder_path = self.config.imgset_path
        if self.dataset_split == 'train':
            path = os.path.join(folder_path, 'train_copy.txt')
        elif self.dataset_split == 'val':
            path = os.path.join(folder_path, 'val.txt')
        elif self.dataset_split == 'test':
            path = os.path.join(folder_path, 'test.txt')

        with open(path, 'r') as f:
            lines = f.readlines()
            indexes = []
            self.count = len(lines)
            for i in range(self.count):
                indexes.append(lines[i].strip())
            indexes = np.array(indexes)
        return indexes
    
    def evaluate(self, predictions, gts, metrics,  metric_data_list):
        gts_list = []
        pred_list = []
        for k in gts.keys():
            gts_list.extend(gts[k])
            pred_list.extend(predictions[k])
        
        #print(config)
        metric_data_list = DetectionMetricDataList()
        
        for class_category in self.config.class_names:
            for dist_th in self.config.dist_ths:
                md = accumulate(pred_list, gts_list, class_category, dist_th,predictions, gts)
                metric_data_list.set(class_category, dist_th, md)
        
        metrics = DetectionMetrics(self.config)
        #print(metrics)
        for class_name in self.config.class_names:
            # Compute APs.
            for dist_th in self.config.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.config.min_recall, self.config.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.config.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.config.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # # Compute evaluation time.
        # #metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list
    
    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        #if self.verbose:
        print('Rendering PR and TP curves')

        def savepath(name):
            path = os.path.join(self.plot_dir, self.dataset_split)
            if not os.path.exists(path):
                os.makedirs(path)
            return os.path.join(path, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.config.min_precision, min_recall=self.config.min_recall,
                     dist_th_tp=self.config.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.config.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.config.min_precision, self.config.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_tp_curve(md_list, metrics, detection_name, self.config.min_recall, self.config.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.config.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.config.min_precision, self.config.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def main(self):

        # # config_file = "/export/amsvenkat/project/3d-multi-agent-tracking/SMOKE/evaluation/config.json"
        # with open(config_file, 'r') as f:
        #     data = json.load(f)
        # self.config = Config.deserialize(data)

        index = eval.get_count_of_dataset()
        
        gts = self.load_gts(index)
        predictions = self.load_predictions(index)
        assert len(predictions) == len(gts), "Number of predictions and ground truth are not equal"
        print("Number of predictions and ground truth are equal" , len(predictions))
    
        all_metrics, metric_data_list  = self.evaluate(predictions, gts)
        
        self.render(all_metrics, metric_data_list)
        print('Saving metrics to: %s' % self.output_dir)

        metrics_summary = all_metrics.serialize()
        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        #print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('ObjectClass\tAP\tATE\tASE\tAOE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
        ))

    def main_train(self, predictions, gts):

        predictions = predictions
        gts = gts
        assert len(predictions) == len(gts), "Number of predictions and ground truth are not equal"
        print("Number of predictions and ground truth are equal" , len(predictions))
        
        config_file = "/export/amsvenkat/project/3d-multi-agent-tracking/SMOKE/evaluation/config.json"
        
        with open(config_file, 'r') as f:
            data = json.load(f)
        self.config = Config.deserialize(data)
        
        all_metrics = DetectionMetrics(self.config)
        metric_data_list = DetectionMetricDataList()
        all_metrics, metric_data_list  = self.evaluate(predictions, gts, all_metrics, metric_data_list)
        metrics_summary = all_metrics.serialize()
        mAP = metrics_summary['mean_ap']
        error ={}
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            error[tp_name] =tp_val

        return mAP, error

if __name__ == "__main__":

    # choose test or validation set for evaluation
    dataset_split = 'test'

    config_file = sys.argv[1]

    plot_dir= "/export/amsvenkat/project/3d-multi-agent-tracking/SMOKE/logs/logs_v8.1/eval/plot"
    preds_path = "/export/amsvenkat/project/3d-multi-agent-tracking/SMOKE/logs/logs_v8.1/inference/carla_test/data/"
    gts_path = "/export/amsvenkat/project/data/train_v5/label_1/"

    # eval = Evaluate(dataset_split, preds_path, gts_path, plot_dir, config_file=config_file)
    eval = Evaluate(dataset_split=dataset_split,config_file=config_file)
    eval.main()
