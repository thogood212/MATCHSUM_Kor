import torch
from torch import nn
from torch.nn import init

from transformers import BertModel

class MatchSum(nn.Module):
    
    def __init__(self, candidate_num, encoder, hidden_size=768):
        super(MatchSum, self).__init__()
        
        self.hidden_size = hidden_size
        self.candidate_num  = candidate_num
        
        self.encoder = BertModel.from_pretrained(encoder)

    def forward(self, text_id, candidate_id, summary_id, labels):
        
        batch_size = text_id['input_ids'].size(0)

        # get document embedding
        outputs = self.encoder(input_ids=text_id['input_ids'], attention_mask=text_id['attention_mask'],
                           token_type_ids=text_id['token_type_ids'])
        
        last_hidden_states = outputs[0]
        doc_emb = last_hidden_states[:,0,:]

        assert doc_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]
        
        # get summary embedding
        outputs = self.encoder(input_ids=summary_id['input_ids'], attention_mask=summary_id['attention_mask'],
                           token_type_ids=summary_id['token_type_ids'])
        last_hidden_states = outputs[0]
        summary_emb = last_hidden_states[:, 0, :]

        assert summary_emb.size() == (batch_size, self.hidden_size) # [batch_size, hidden_size]

        # get summary score
        summary_score = torch.cosine_similarity(summary_emb, doc_emb, dim=-1)

        # get candidate embedding
        cand_input_ids = []
        cand_attention_mask = []
        cand_token_type_ids = []

        for candidate in candidate_id:
            cand_input_ids.append(candidate['input_ids'])
            cand_attention_mask.append(candidate['attention_mask'])
            cand_token_type_ids.append(candidate['token_type_ids'])
        # stack embeddings
        cand_input_ids = torch.stack(cand_input_ids).view(-1,len(cand_input_ids[0][0]))
        cand_attention_mask = torch.stack(cand_attention_mask).view(-1,len(cand_attention_mask[0][0]))
        cand_token_type_ids = torch.stack(cand_token_type_ids).view(-1,len(cand_token_type_ids[0][0]))


        outputs = self.encoder(input_ids=cand_input_ids, attention_mask=cand_attention_mask,
                           token_type_ids=cand_token_type_ids)
        last_hidden_states = outputs[0]
        candidate_emb = last_hidden_states[:, 0, :].view(batch_size, self.candidate_num, self.hidden_size)  # [batch_size, candidate_num, hidden_size]

        assert candidate_emb.size() == (batch_size, self.candidate_num, self.hidden_size)
        
        # get candidate score
        doc_emb = doc_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, doc_emb, dim=-1) # [batch_size, candidate_num]
        assert score.size() == (batch_size, self.candidate_num)

        #score는 문장단위 score , summary_score는 요약 단위 score
        return {'score': score, 'summary_score': summary_score}