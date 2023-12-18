import os
import sys

import torch
import random



# This worked!!!!!
# C_MAG = {
#         'c_high': (0.2,'<C_HIGH>'),# Testing at full throttle to start. dont confuse model with multiple goals.
     
#     }

C_CONTINOUS_SCALE_LOSS_FACTOR = 0.01 #uSE TO ENSURE loss is balanced with reconstruction loss 

C_MAG = {
        # 'c_high': (0.05,'<C_HIGH>'),# Testing at full throttle to start. dont confuse model with multiple goals.
        'c_med': (0.2,'<C_MED>'),
        # 'c_low': (0.5,'<C_LOW>'),
        # 'c_repeat': (1.0,'<C_REP>'),
        # 'c_cont': (0.01,'<C_CONT>'), # Much less pressure on the eos loss here. To try and get a dynamic loss...

    }

MAG_VAL_IDX = 0
MAG_TOKEN_IDS = 1

C_TOKENS = {
    "c_BOS": "<C_BOS>",
    "c_EOS": "<C_EOS>"
    }

C_TOKEN_SHIFT = 3

class CondenserTokenizer(torch.nn.Module):
    def __init__(self,tokenizer, magnetude_dict = C_MAG, tokens = C_TOKENS, dim = 4096, use_internal_embeddings = True,device='cuda'):
        super(CondenserTokenizer, self).__init__()
        tmp = {k:t for k,(v,t) in magnetude_dict.items()}
        self.tokens = {**tokens,**tmp}
        self.magnetude = magnetude_dict
        self.embedding_dim = dim
        self.use_internal_embeddings = use_internal_embeddings
        self.num_tokens = len(self.tokens)
        if use_internal_embeddings:
            self.embed = torch.nn.parameter.Parameter(torch.empty((self.num_tokens,4096),dtype = torch.float32,device = device), requires_grad=True)
            self.reset_param() # need to add loading function later.    

        self.tokenizer = self.add_tokenizer(tokenizer)
        self.tok_idx_lookup = {self.len_o + i:t for i,(k,t) in enumerate(self.tokens.items()) }
        #return self.add_model(model) #primarily adding model to ensure pulling from the embedding table doesn't cause errors
        # better solution would be if the embedding table would work....
        


    def reset_param(self):
        torch.nn.init.normal_(self.embed)
        

    def forward(self,txt):
        '''
        Use as the tokenization step.
        '''
        txt,mags = self.insert_condenser_tokens(txt)
        inputs = self.tokenizer(txt,return_tensors='pt', padding=True)
        inputs['mags'] = mags
        return inputs 

    def add_model(self,model):
        model.resize_token_embeddings(len(self.tokenizer),pad_to_multiple_of=128)
        return model
    
    def add_tokenizer(self,tokenizer): # do not use with PEFT
        self.len_o = len(tokenizer)
        num_added_toks = tokenizer.add_tokens([t for k,t in self.tokens.items()],special_tokens=True)
        return tokenizer

    def insert_condenser_tokens(self,text):
        rate = random.choices(list(self.magnetude.keys()),k=len(text)) # choose a random condenser rate for each string
        c_text = [self.tokens['c_BOS'] + self.magnetude[r][MAG_TOKEN_IDS] + s + self.tokens['c_EOS'] for (s,r) in zip(text,rate)] #
        c_text = [self.tokens['c_BOS'] + self.magnetude[r][MAG_TOKEN_IDS] + s + self.tokens['c_EOS'] for (s,r) in zip(text,rate)] #
        return c_text,rate
    
    def extract_embeddings(self,tokens,embedding_table,insert_replacement_embs = True):
        """
        generate embeddings from a token list and replace ctoken embeddings with internal
        representations. We are doing this because PEFT models will not optimize embeddings...
        for some unknown reason.
        """
        emb = embedding_table(tokens) 
        if insert_replacement_embs:
            for k in self.tok_idx_lookup.keys():
                emb[tokens == k] = self.embed[k - self.len_o].half()
        return emb

    def get_token_shift(self):
        return C_TOKEN_SHIFT
    
    def get_mag_tensor(self,mags,device = 'auto'):
        vals = [self.magnetude[m][MAG_VAL_IDX] for m in mags]
        res = torch.tensor(vals,device = device).unsqueeze(-1)
        return res


    def get_scale_tensor(self,mags,device = 'auto'):
        vals = [self.magnetude[m][MAG_TOKEN_IDS] == '<C_CONT>' for m in mags]
        res = C_CONTINOUS_SCALE_LOSS_FACTOR*torch.tensor(vals,device = device,dtype = torch.float).unsqueeze(-1)
        res = res + torch.tensor([not v for v in vals],device = device,dtype = torch.float).unsqueeze(-1)
        # print(res)
        # print(vals)
        # print([not v for v in vals])

        return res        
    
    def get_dummy():
        return

    def get_embedding_weights(self,embedding_table,tokenizer):
        tok = tokenizer("".join([t for k,t in self.tokens.items()]),add_special_tokens=False)["input_ids"]
        return embedding_table.weight[tok]