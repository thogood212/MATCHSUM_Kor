#tokenize하는 부분을 도려내고 canddidate 문서 자체를 저장
#tokenize하는 부분은 collate_fn부분에서 변경하는 것으로 변경
from itertools import combinations
from kor_rouge_metric import Rouge

def get_candidates(tokenizer, idx,df):
    MAX_LEN=180
    MAX_LEN_text = 512
    Final_rouge = Rouge(
            metrics=["rouge-n", "rouge-l"],
            max_n=2,
            limit_length=True,
            length_limit=1000,
            length_limit_type="words",
            use_tokenizer=True,
            apply_avg=True,
            apply_best=False,
            alpha=0.5,  # Default F1_score
            weight_factor=1.2,
        )
    # load data
    data = {}
    #original text data
    data['text'] = df.loc[idx]['src_txt']
    #gold summary
    data['summary'] = df.loc[idx]['abstractive']

    # get candidate summaries
    #각각 문서에서 5개의 중요한 문장을 truncate합니다.(using koBertSum)
    # 2,3개의 문장을 선택하여 combinations을 사용하여 C(5,2)+C(5,3)=20 총 20개의 후보군 요약문을 생성합니다.
    # 본 모델은 기본적으로 koBertSum 모델을 사전에 활용하여야 get candidate를 할 수 있습니다.
    # 데이터프레임(df)에 ['sum_sents_idxes']열에 5개의 문장인덱스가 존재해야합니다.
    #pruned text(5) index
    sent_id = df.loc[idx]['sum_sents_idxes']
    indices = list(combinations(sent_id, 2))
    indices += list(combinations(sent_id, 3))

    # get ROUGE score for each candidate summary and sort them in descending order
    score = []
    for i in indices:
        i = list(i)
        i.sort()

        # write dec
        dec = []
        for j in i:
            sent = data['text'][j]
            dec.append(sent)
        #dec는 candidate summary당 문장 모음
        rouge = Final_rouge.get_scores(df.loc[idx]['sum_sents_tokenized'], dec)
        
        rouge1 = float(rouge['rouge-1']['f'])
        rouge2 = float(rouge['rouge-2']['f'])
        rougel = float(rouge['rouge-l']['f'])
        rouge = (rouge1 + rouge2 + rougel) / 3
        score.append((i, rouge))
    
    #Rouge score가 높은 순서대로 정렬
    score.sort(key=lambda x : x[1], reverse=True)
    
    # write candidate indices and score
    data['ext_idx'] = sent_id
    data['indices'] = []
    data['score'] = []
    for i, R in score:
        data['indices'].append(list(map(int, i)))
        data['score'].append(R)

    # get candidate_text
    candidate_summary = []
    for i in data['indices']:
        cur_summary = []
        for j in i:
            cur_summary.append(data['text'][j])
        cur_summary = ' '.join(cur_summary)
        candidate_summary.append(cur_summary)
        
    data['candidate_summary'] = candidate_summary

    return data