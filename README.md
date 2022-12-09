# MATCHSUM_Kor
MatchSum model - Korean version(Text Summarization)

- Extractive Summarization as Text Matching([paper](https://https://arxiv.org/pdf/2004.08795.pdf))
- Github(https://github.com/maszhongming/MatchSum)

## Dependencies

- kobert-transformers           0.5.1
- transformers                  4.25.1
- konlpy                        0.6.0(mecab)
- pytorch-metric-learning       1.6.3
- torch                         1.13.0

## Data
AI-HUB 문서요약 텍스트(신문기사, 기고문, 잡지기사, 법원 판결문 원문, 요약 3문장)
[Link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=97)

  
  
## 1. get_candidate

MatchSum 모델은 기본적으로 pruning 과정을 이미 거친 데이터를 input 데이터로 사용합니다.  
본 논문에서는 BERTEXT모델을 사용하여 pruning과정을 진행합니다.

get_candidate.py를 실행하기 위해서는 아래의 column이 데이터에 정의되어 있어야 합니다.
- ['src_txt'] = 문서 본문 (list) ex)[ '문장1', '문장2',...,'문장..']
- ['abstractive'] = 문서 중요 요약문 (list) ex) ['문장']
- ['sum_sents_idxes'] = pruning하여 추출한 문서의 중요 문장 5개의 인덱스 (list) ex) [인덱스1,인덱스2,인덱스3,인덱스4,인덱스5]  
  
  
## 2. Train
 
arguments 설명:
- mode (필수) : MatchSum train 혹은 test mode 설정
- save_path (필수) : 체크포인트 및 모델 저장 경로
- gpus (필수) : train에 사용할 수 있는 gpus
- encoder (필수) : matchsum모델에 사용할 encoder(bert)
- batch_size : train batch size, default=16
- accum_count : number of updates steps to accumulate before performing a backward/update pass', default=2,
- candidate_num : 사용할 candidates summaries 개수, default=20
- max_lr : max learning rate for warm up, default=2e-5
- margin : parameter for MarginRankingLoss, default=0.01
- warmup_steps : warm up steps for training, default=10000
- n_epochs : train epochs, default=5
- valid_steps : number of update steps for validation and saving checkpoint, default=1000
- data_path (필수) : data 저장 경로 및 data 이름(.json)              
- output_path (필수) : model outputs(summary) 저장경로 및 Rouge score가 작성될 train_info.txt 저장 경로
- save : model outputs 저장 유무, fault=False

```
#ex)
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_matching.py --mode=train --encoder='monologg/kobert' --save_path=./bert --gpus=0,1,2,3 --data_path=./data/prun_data.json --output_path=./output
```

video memory가 부족한 경우, batch_size, candidate_num를 조정하여 사용하시기 바랍니다.(utils.custom_collate_fn에 max_len, max_len_text을 조정할수도 있습니다.(문서의 길이가 긴 경우 비추천))
  
  
## 3. Test

```
#ex)
CUDA_VISIBLE_DEVICES=0 python train_matching.py --mode=test --encoder='monologg/kobert' --save_path=./bert/ --gpus=0 --data_path=./data/prun_data.json --output_path=./output
```
  
  
## 4. 결과

컴퓨팅 환경 문제로 약 2만개 행의 데이터만을 사용하고 candidate_num=8, batch_size=4로 제한하여 짧게 5epoch 정도 훈련하였을때  

 - ROUGE-1 : 0.383
 - ROUGE-2 : 0.195
 - ROUGE-L : 0.279  

  
  
## 5. 수정된 부분

- fastNLP Trainer --> huggingface Trainer
- fastNLP Loss, Metrics --> pytorch_metric_learning.losses
- pyrouge --> kor_rouge_metric
- dataloader.py --> Trainer 내의 collate_fn

