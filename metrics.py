import numpy as np
import json
from os.path import join
import torch
import logging
import tempfile
from datetime import timedelta
from time import time

from kor_rouge_metric import Rouge

from pytorch_metric_learning.losses import BaseMetricLossFunction

class MatchRougeMetric(BaseMetricLossFunction): #for test
    def __init__(self, data, n_total,output_path, save_name, score=None, save =False):
        super(MatchRougeMetric, self).__init__()

        self.data        = data
        self.n_total     = n_total
        self.cur_idx = 0
        self.ext = []
        self.start = time()
        self.rouge = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,)

        self.path = output_path
        self.save_name = save_name
        self.save = save

    def evaluate(self, score):

        batch_size = score.size(0)

        for i in range(batch_size):
            ext = int(torch.max(score[i], dim=0).indices) # batch_size = 1
            self.ext.append(ext)
            print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                  i, self.n_total, self.cur_idx/self.n_total*100, timedelta(seconds=int(time()-self.start))
                ), end='')
    
    def save_output(self, dec_list, ref, path, name):
        with open(join(self.path, '{}.txt'.format(self.name)), 'w') as f:
            print('model output :',self.dec_list, file=f)
            print('reference summary :',self.dec_list, file=f)

    def get_metric(self, reset=True):
        R_1_sum = 0
        R_2_sum = 0
        R_L_sum = 0
        print('\nStart calculate each text !!!')
        for i, ext in enumerate(self.ext):
            sent_ids = self.data.loc[i]['indices'][ext]
            dec=[]
            for j in sent_ids:
                dec.append(self.data.loc[i]['text'][j])#self.cur_idx

            ref = self.data.loc[i]['summary']
            
            #저장
            if self.save == True:
                self.save_output(dec,ref, self.path, self.save_name)           
            
            dec = ''.join(dec)
            
            self.cur_idx += 1

            R_1, R_2, R_L = self.eval_rouge(dec, ref)
        
            R_1_sum += R_1
            R_2_sum += R_2
            R_L_sum += R_L
        
        R_1_mean = R_1_sum/self.cur_idx
        R_2_mean = R_2_sum/self.cur_idx 
        R_L_mean = R_L_sum/self.cur_idx 

        print('Start evaluating ROUGE score !!!')
        
        eval_result = {'ROUGE-1': R_1_mean, 'ROUGE-2': R_2_mean, 'ROUGE-L':R_L_mean} 
        
        if reset == True:
            self.cur_idx = 0
            self.ext = []
            self.data = []
            self.start = time()

        return eval_result
        
    def eval_rouge(self, dec, ref):
        if dec == '' or ref == '':
            return 0.0
        scores = self.rouge.get_scores(dec, ref)
        R_1 = scores['rouge-1']['f']
        R_2 = scores['rouge-2']['f']
        R_L = scores['rouge-l']['f']

        return R_1, R_2, R_L

#test를 위한 evaluate 함수 만들기
    def custom_compute_metrics_test(self,pred):
        preds = pred.predictions[0]
        preds = torch.Tensor(preds)
        test_metric = MatchRougeMetric(data=self.data, n_total = len(self.data))
        test_metric.evaluate(preds)
        eval_result = test_metric.get_metric()
        return eval_result




'''compute_metrics는 evaluation에 사용할 metric을 계산하는 함수이다. 
모델의 output인 EvalPrediction을 input으로 받아 metric을 dictionary 형태로 return하는 함수가 되야 한다.'''

class ValidMetric(BaseMetricLossFunction): #for validation

    def __init__(self, output_path, data, score=None):
        super(ValidMetric, self).__init__()
 
        self.output_path = output_path
        self.data = data

        self.top1_correct = 0
        self.top6_correct = 0
        self.top10_correct = 0
         
        self.rouge = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,)
        self.ROUGE = 0.0
        self.Error = 0
        self.cur_idx = 0
    
    # an approximate method of calculating ROUGE
    def fast_rouge(self, dec, ref):

        if dec == '' or ref == '':
            return 0.0
        scores = self.rouge.get_scores(dec, ref)

        return (scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3

    def evaluate(self, score):

        batch_size = score.size(0)

        #score에서 각 차원의 최대값 인덱스를 뽑아내는 것.

        self.top1_correct += int(torch.sum(torch.max(score, dim=1).indices == 0))
        self.top6_correct += int(torch.sum(torch.max(score, dim=1).indices <= 5))
        self.top10_correct += int(torch.sum(torch.max(score, dim=1).indices <= 9))

        # Fast ROUGE
        for i in range(batch_size):
            max_idx = int(torch.max(score[i], dim=0).indices)
            if max_idx >= len(self.data.loc[self.cur_idx]['indices']):
                self.Error += 1 # Check if the candidate summary generated by padding is selected
                self.cur_idx += 1
                continue
            ext_idx = self.data.loc[self.cur_idx]['indices'][max_idx]
            ext_idx.sort()
            dec = []
            ref = ''.join(self.data.loc[self.cur_idx]['summary'])
            for j in ext_idx:
                dec.append(self.data.loc[self.cur_idx]['text'][j])
            dec = ''.join(dec)

            self.ROUGE += self.fast_rouge(dec, ref)
            self.cur_idx += 1

    def get_metric(self, reset=True):

        top1_accuracy = self.top1_correct / self.cur_idx
        top6_accuracy = self.top6_correct / self.cur_idx
        top10_accuracy = self.top10_correct / self.cur_idx

        ROUGE = self.ROUGE / self.cur_idx

        eval_result = {'top1_accuracy': top1_accuracy, 'top6_accuracy': top6_accuracy, 'top10_accuracy': top10_accuracy,
                       'Error': self.Error, 'ROUGE': ROUGE}
        
        with open(join(self.output_path, 'train_info.txt'), 'a') as f:
            print('top1_accuracy = {}, top6_accuracy = {},Error = {}, ROUGE = {}'.format(
                top1_accuracy, top6_accuracy,self.Error, ROUGE),file=f)

        if reset:
            self.top1_correct = 0
            self.top6_correct = 0
            self.top10_correct = 0
            self.ROUGE = 0.0
            self.Error = 0
            self.cur_idx = 0
        return eval_result

    def custom_compute_metrics(self,pred):
        preds = pred.predictions[0]
        preds = torch.Tensor(preds)
        val_metric = ValidMetric(output_path=self.output_path, data=self.data)
        val_metric.evaluate(preds)
        eval_result = val_metric.get_metric()
        return eval_result