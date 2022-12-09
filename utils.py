import json
import pandas as pd
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from kobert_transformers import get_tokenizer

class CustomDataset(Dataset):
    """
    - input_data: list of tokens
    """
    
    def __init__(self, input_data1:list,input_data2:list,input_data3:list) -> None:
        self.X = input_data1
        self.Y = input_data2
        self.Z = input_data3
        #self.X = [input_data.text_id.to_list(), input_data.candidate_id.to_list(), input_data.summary_id.to_list()]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index], self.Z[index]

def custom_collate_fn(batch):
    """
    - batch: list of Dicts ({'text_id':tensorized_text,'candidate_id': token_cand,  'summary_id':tensorized_summ})
    
    한 배치 내 문장들을 tokenizing 한 후 텐서로 변환함. 
    이때, dynamic padding (즉, 같은 배치 내 토큰의 개수가 동일할 수 있도록, 부족한 문장에 [PAD] 토큰을 추가하는 작업)을 적용
    
    한 배치 내 레이블(target)은 텐서화 함.
    
    """
    max_len=180
    max_len_text=512 #원래 512
    tokenizer = get_tokenizer()

    text_list, cand_list, summ_list = [], [], []
    token_cand = []
    for input_text, input_cand, input_summ in batch:
        text_list.append(input_text)
        cand_list.append(input_cand)
        summ_list.append(input_summ)
    
    tensorized_text = tokenizer(
    text_list,
    add_special_tokens=True,
    padding="longest",  # 배치내 가장 긴 문장을 기준으로 부족한 문장은 [PAD] 토큰을 추가
    truncation=True, # max_length를 넘는 문장은 이 후 토큰을 제거함
    max_length=max_len_text,
    return_tensors='pt' # 토크나이즈된 결과 값을 텐서 형태로 반환
    )
    
    for candidate in cand_list:
        tensorized_cand = tokenizer(
        candidate,
        add_special_tokens=True,
        padding='max_length',  # 배치내 가장 긴 문장을 기준으로 부족한 문장은 [PAD] 토큰을 추가 ("longest")
        truncation=True, # max_length를 넘는 문장은 이 후 토큰을 제거함
        max_length=max_len,
        return_tensors='pt' # 토크나이즈된 결과 값을 텐서 형태로 반환
        )

        token_cand.append(tensorized_cand)

    tensorized_summ = tokenizer(
    summ_list,
    add_special_tokens=True,
    padding="longest",  # 배치내 가장 긴 문장을 기준으로 부족한 문장은 [PAD] 토큰을 추가
    truncation=True, # max_length를 넘는 문장은 이 후 토큰을 제거함
    max_length=max_len,
    return_tensors='pt' # 토크나이즈된 결과 값을 텐서 형태로 반환
    )
    labels= torch.rand(len(tensorized_summ), 1)
    result ={'text_id':tensorized_text,'candidate_id': token_cand, 'summary_id':tensorized_summ,'labels':labels}
    return result

def split_data(prun_data_path, candidate_num):
    '''
    1. candidate 개수를 조정
    2. train, validation, test 데이터 분리
    3. 훈련에 필요한 dataset으로 변환

    생성되는 변수
    eval_validation_dataset, eval_test_dataset = trainer evaluate에 필요한 데이터프레임
    train_dataset, validation_dataset, test_dataset = trainer train에 필요한 데이터셋
    '''
    #json 불러오기
    with open(prun_data_path) as f: 
        prun_data = json.load(f)
        prun_data = prun_data['data']
        prun_data =pd.DataFrame(prun_data)

    #candidate개수 조정(defalut : 20개)
    prun_data['candidate_summary'] = prun_data['candidate_summary'].apply(lambda x: x[:candidate_num])
    prun_data['indices'] = prun_data['indices'].apply(lambda x: x[:candidate_num])

    #data split
    dataset_size = len(prun_data)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.1)
    test_size = dataset_size - train_size - validation_size

    #evaluate 부분에서 reset된 index가 필요
    eval_validation_dataset = prun_data[train_size:train_size+validation_size].reset_index(drop=True)
    eval_test_dataset = prun_data[train_size+validation_size:].reset_index(drop=True)

    # text, summary 문자열 합쳐주기
    prun_data['text'] = prun_data['text'].apply(lambda x: ' '.join(x))
    prun_data['summary'] = prun_data['summary'].apply(lambda x: ' '.join(x))

    #train에 사용할 데이터셋
    train_dataset = prun_data[:train_size].reset_index(drop=True)
    validation_dataset = prun_data[train_size:train_size+validation_size].reset_index(drop=True)
    test_dataset = prun_data[train_size+validation_size:].reset_index(drop=True)

    print('train :',train_size, len(train_dataset))
    print('validation :',validation_size,len(validation_dataset))
    print('test :',test_size,len(test_dataset))

    #Dataset 설정
    train_dataset = CustomDataset(train_dataset['text'], train_dataset['candidate_summary'], train_dataset['summary'])
    validation_dataset = CustomDataset(validation_dataset['text'], validation_dataset['candidate_summary'], validation_dataset['summary'])
    test_dataset = CustomDataset(test_dataset['text'], test_dataset['candidate_summary'], test_dataset['summary'])

    data = {'eval_validation_dataset':eval_validation_dataset,'eval_test_dataset':eval_test_dataset,
    'train_dataset':train_dataset,'validation_dataset':validation_dataset,'test_dataset':test_dataset }

    return data