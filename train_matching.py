import sys
import argparse
import os
import json
import torch
from time import time
from datetime import timedelta
from os.path import join, exists
from torch.optim import Adam, AdamW

from utils import custom_collate_fn, split_data

from model import MatchSum
from metrics import ValidMetric, MatchRougeMetric
from transformers import EarlyStoppingCallback, TrainingArguments,Trainer

class MyTrainer(Trainer):
    # loss_name 이라는 인자를 추가로 받아 self에 각인 시켜줍니다.
    def __init__(self, loss_name, margin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name # 각인!
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False):

        # config에 저장된 loss_name에 따라 다른 loss 계산 
        if self.loss_name == 'MarginRankingLoss':
            custom_loss = torch.nn.MarginRankingLoss(self.margin)

        # equivalent to initializing TotalLoss to 0

        outputs = model(**inputs)

        score = outputs['score']
        summary_score = outputs['summary_score']

        # here is to avoid that some special samples will not go into the following for loop
        ones = torch.ones(score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss = loss_func(score, score, ones)

        # candidate loss
        n = score.size(1)
        for i in range(1, n):
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones(pos_score.size()).cuda(score.device)
            loss_func = torch.nn.MarginRankingLoss(self.margin * i)
            TotalLoss += loss_func(pos_score, neg_score, ones)

        # gold summary loss
        pos_score = summary_score.unsqueeze(-1).expand_as(score)
        neg_score = score
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        ones = torch.ones(pos_score.size()).cuda(score.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        TotalLoss += loss_func(pos_score, neg_score, ones)
        
        return (TotalLoss, outputs) if return_outputs else TotalLoss

def configure_training(args):
    devices = [int(gpu) for gpu in args.gpus.split(',')]
    params = {}
    params['encoder']       = args.encoder
    params['candidate_num'] = args.candidate_num
    params['batch_size']    = args.batch_size
    params['accum_count']   = args.accum_count
    params['max_lr']        = args.max_lr
    params['margin']        = args.margin
    params['warmup_steps']  = args.warmup_steps
    params['n_epochs']      = args.n_epochs
    params['valid_steps']   = args.valid_steps
    params['save_path'] = args.save_path
    params['data_path'] = args.data_path
    params['output_path'] = args.output_path
    params['save'] = args.save
    return devices, params

def train_model(args):
   
    devices, train_params = configure_training(args)
    # load summarization datasets
    #print('Information of dataset is:')

    trainer_args=TrainingArguments(
    # checkpoint
    output_dir=args.save_path,
    # Model Save & Load
    save_strategy = "no", # 각 epoch 마지막에 저장 
    load_best_model_at_end=True, # train 종료시 best model 로드할지 여부
    # Dataset
    num_train_epochs=args.n_epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    # Evaluation 
    evaluation_strategy = "no",# 각 epoch 마지막에 평가
    metric_for_best_model = 'ROUGE',
    #for train
    warmup_steps=args.warmup_steps,               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,
    # Randomness
    seed=42,)

    if not exists(args.output_path):
        os.makedirs(args.output_path)
    if not exists(args.save_path):
        os.makedirs(args.save_path)

    #데이터 load 함수
    data = split_data(args.data_path,args.candidate_num)

    eval_validation_dataset = data['eval_validation_dataset']
    train_dataset = data['train_dataset']
    validation_dataset = data['validation_dataset']

    print('Devices is:')
    print(devices)

    # configure model
    model = MatchSum(args.candidate_num, args.encoder)

    optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    eps=1e-8)
    
    val_metric = ValidMetric(output_path=args.output_path, data=eval_validation_dataset)

    assert 16 % len(devices) == 0

    trainer = MyTrainer(model= model,args=trainer_args,train_dataset=train_dataset, eval_dataset=validation_dataset,
                        compute_metrics= val_metric.custom_compute_metrics,optimizers=(optimizer,None),
                        data_collator=custom_collate_fn,loss_name='MarginRankingLoss', margin = 0.01 )

    print('Start training with the following hyper-parameters:')
    print(train_params)
    trainer.train()
    torch.save(model, join(args.save_path,'model.pt'))

def test_model(args):
    test_args=TrainingArguments(
    # checkpoint
    output_dir=args.save_path,
    # Model Save & Load
    save_strategy = "no", # 각 epoch 마지막에 저장 
    load_best_model_at_end=True, # train 종료시 best model 로드할지 여부
    # Dataset
    num_train_epochs=args.n_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    # Evaluation 
    evaluation_strategy = "no",# 각 epoch 마지막에 평가
    metric_for_best_model = 'ROUGE-1',
    #for train
    warmup_steps=args.warmup_steps, # number of warmup steps for learning rate scheduler
    weight_decay=0.01,
    # Randomness
    seed=42,)

    #데이터 load 함수
    data = split_data(args.data_path,args.candidate_num)
   
    eval_test_dataset = data['eval_test_dataset']
    test_dataset = data['test_dataset']

    models = os.listdir(args.save_path)
    print('Information of dataset is:')
    print(test_dataset)

    #test_dataset = dataset
    # need 1 gpu for testing
    device = [0]
    
    batch_size = 1

    for i, cur_model in enumerate(models):
        
        print('Current model is {}'.format(cur_model))

        # load model
        model = torch.load(join(args.save_path, cur_model))

        optimizer = AdamW(
        model.parameters(),
        lr=2e-5,
        eps=1e-8)

        # configure testing
        test_metric = MatchRougeMetric(data=eval_test_dataset, n_total = len(eval_test_dataset),
                    output_path=args.output_path, save_name=f'save_file{i}',save =args.save )
        tester = MyTrainer(model= model,args=test_args,train_dataset=test_dataset, eval_dataset=test_dataset,
                        compute_metrics= test_metric.custom_compute_metrics_test, optimizers=(optimizer,None),
                        data_collator=custom_collate_fn,loss_name='MarginRankingLoss',margin = 0.01 )
        tester.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training/testing of MatchSum'
    )
    parser.add_argument('--mode', required=True,
                        help='training or testing of MatchSum', type=str)

    parser.add_argument('--save_path', required=True,
                        help='root of the model', type=str)
    # example for gpus input: '0,1,2,3'
    parser.add_argument('--gpus', required=True,
                        help='available gpus for training(separated by commas)', type=str)
    parser.add_argument('--encoder', required=True,
                        help='the encoder for matchsum (bert/roberta)', type=str)

    parser.add_argument('--batch_size', default=16,
                        help='the training batch size', type=int)
    parser.add_argument('--accum_count', default=2,
                        help='number of updates steps to accumulate before performing a backward/update pass', type=int)
    parser.add_argument('--candidate_num', default=20,
                        help='number of candidates summaries', type=int)
    parser.add_argument('--max_lr', default=2e-5,
                        help='max learning rate for warm up', type=float)
    parser.add_argument('--margin', default=0.01,
                        help='parameter for MarginRankingLoss', type=float)
    parser.add_argument('--warmup_steps', default=10000,
                        help='warm up steps for training', type=int)
    parser.add_argument('--n_epochs', default=5,
                        help='total number of training epochs', type=int)
    parser.add_argument('--valid_steps', default=1000,
                        help='number of update steps for validation and saving checkpoint', type=int)
    parser.add_argument('--data_path', required=True,
                        help='data path for load data(json)', type=str)                        
    parser.add_argument('--output_path', required=True,
                        help='output path for save outputs(summary),train_info.txt(Rouge score)', type=str)
    parser.add_argument('--save', default=False,
                        help='choose save the outputs(bool), defalut is False')

    args = parser.parse_known_args()[0]

    if args.mode == 'train':
        print('Training process of MatchSum !!!')
        train_model(args)
    else:
        print('Testing process of MatchSum !!!')
        test_model(args)