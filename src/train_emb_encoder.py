
import os
import sys


import torch
import random

import einops
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)

from torch.utils.data import DataLoader

#local
from utils import (
    create_model_and_tokenizer, 
    load_data,
    decode_logits

)

from architectures import WordEmbeddingAligner

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def select_batch(x,embedding_table,model,bs,size,bos_token,tokenizer):
    #x = torch.randint(0,size,(bs,))
    x = tokenizer(x,add_special_tokens = True,return_tensors='pt')['input_ids']
    #print(x.shape)
    w_emb = embedding_table(x)[:,1:]
    #print(w_emb.shape)
    #m_token = torch.concatenate([bos_token, x.unsqueeze(0)],dim=1)
    with torch.no_grad():
        out = model(x,output_hidden_states = True) #randomness will effect the attentions here. probably fine (could set att to 0..)
    m_emb = out.hidden_states[-1][0,:-1]
    #Instead of input embs, target the pred words
    t_tokens = torch.argmax(out.logits[0,:-1],dim=1)
    t_emb = embedding_table(t_tokens)
    
    return w_emb.to(device),m_emb.to(device),x,out,t_emb


def cosine_loss(pred,y):
    return 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(pred, y)))


def reconstruction_loss(pred,y):
    x = torch.square(pred - y)
    x = torch.sum(x,dim=1) # SUM makes the numbers to big. equivalent
    x = torch.mean(torch.sqrt(x))
    return x


def comp_cosine(w_emb,m_emb,pred,t_emb):
    w_m = torch.mean(torch.abs(torch.nn.functional.cosine_similarity(w_emb, m_emb)))
    w_p = torch.mean(torch.abs(torch.nn.functional.cosine_similarity(w_emb, pred)))
    t_m = torch.mean(torch.abs(torch.nn.functional.cosine_similarity(t_emb, m_emb)))
    t_p = torch.mean(torch.abs(torch.nn.functional.cosine_similarity(t_emb, pred)))
    t_w = torch.mean(torch.abs(torch.nn.functional.cosine_similarity(t_emb, w_emb)))
    
    
    print(f"LAST LAYER emb to INPUT emb cosine = {w_p:.4f}")
    print(f"ENCODER OUT emb to INPUT emb cosine = {w_m:.4f}")
    print(f"LAST LAYER emb to TARGT emb cosine = {t_p:.4f}")
    print(f"ENCODER OUT emb to TARGT emb cosine = {t_m:.4f}")
    print(f"TARGT emb to INPUT emb cosine = {t_w:.4f}")

    print("Embedding component ABSOLUTE MEANS:")
    print("Last Layer:",torch.mean(torch.abs(m_emb)).cpu().detach().numpy())
    print("Word:",torch.mean(torch.abs(w_emb)).cpu().detach().numpy())
    print("Target:",torch.mean(torch.abs(t_emb)).cpu().detach().numpy())
    print("Encoder (Pred):",torch.mean(torch.abs(pred)).cpu().detach().numpy())
    





def main():
    #try SGD
    overfit = False
    size = 32000
    checkpoint_step = 500
    lr = 1.0e-5
    #OUTPUT_DIR = '../models/w_encoder/lr1e-5_L2_2L256_Adam_fudge0.03_25' # try 1024
    OUTPUT_DIR = '../models/w_encoder/lr1e-5_L2_2L1024_L2_fudge0.03_28'
    
    
    # load last checkpoint if exists
    if  os.path.isdir(OUTPUT_DIR):
        w_encoder = torch.load(os.path.join(OUTPUT_DIR,"checkpoint"))
        OUTPUT_DIR = OUTPUT_DIR + "_continue"
    else:
        w_encoder = WordEmbeddingAligner(device,normalize = False,fudge_factor=0.03,n_layers=2,h_dim = 1024)
    print(w_encoder)

    writer = SummaryWriter(log_dir = os.path.join( OUTPUT_DIR ,'logs/train'))
    bs = 16
    epochs = 100

  
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    model, tokenizer = create_model_and_tokenizer(model_id,device_map=None)
    dataset = load_data()


    embedding_table = model.get_input_embeddings()
    #optimizer = torch.optim.Adam(w_encoder.parameters(), lr=lr)
    # param = {'layer_%d'%i:l.parameters() for i,l in enumerate(w_encoder.layers)}
    # param['model'] = w_encoder.parameters()
    # optimizer = torch.optim.SGD([w_encoder.parameters()] + param, lr=lr)

    params = [{'params': l.parameters(), 'lr': lr, 'weight_decay': 0}  for l in w_encoder.layers]
    params += [{'params': w_encoder.parameters(), 'lr': lr, 'weight_decay': 0} ]
    #optimizer = torch.optim.Adam(params)
    optimizer = torch.optim.AdamW(params)


    bos_token = tokenizer(tokenizer.bos_token,add_special_tokens = False,return_tensors='pt')['input_ids']
    dataloader = DataLoader(dataset['train'], batch_size=1,shuffle = True)
    if overfit:
        dataloader = DataLoader(dataset['train'], batch_size=1)  
        overfit_batch = next(iter(dataloader))


    ll = rl = cl = 0
    #for step in range(steps):
    #with torch.autocast(device_type='cuda'):
    for e in range(epochs):
        progress_bar = tqdm(enumerate(dataloader))
        for step,batch in progress_bar:
            progress_bar.set_description(f"rec loss: {rl:.5f} cos loss: {cl:.5f} loss: {ll:.5f}")
            if overfit:
                batch = overfit_batch

            w_emb,m_emb,x,out,t_emb = select_batch(batch['answers'],embedding_table,model,bs,size,bos_token,tokenizer)
            #print(w_emb)
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            pred = w_encoder(m_emb.float())
            #print(pred[0,:10])
            #print(pred.shape,w_emb.shape)
            #loss = reconstruction_loss(pred, w_emb)
            #print(t_emb.shape,pred.shape)
            
            #Idea: Oscillitory Loss contributions
            #have kr(i)*Lr + kc(i)*Lc
            # Where kr(i) and kc(i) are some oscilating functions that
            # Allow the model to focus on 1 of the 2 contributions at a time
            # and oscillate between both focuses.
            # The idea being the model can find solution subspaces that optimize the one problem
            # and then find a subspace in that solution space that optimizes the other
            # instead of trying to solve both problems at once.
            # This could make the optimization problem more tractable.
            # If this works could generalize to N loss contributions
            # Also could Allow the model to learn how to distribute loss contributions in combo of this.
            rec_loss = reconstruction_loss(pred, t_emb)
            cos_loss = cosine_loss(pred, t_emb)
            loss = rec_loss #+ cos_loss #rec_loss
            loss.retain_grad()
            # print(loss.grad)
            loss.backward()
            # print(loss.grad)
            optimizer.step()
            # print(loss.grad)
            # print(pred)
            #print(w_encoder.out_layer.weight[0,:10])
            for layer in w_encoder.layers:
                pass
                #print(layer.weight[0,:10])
            #print(w_encoder.in_layer.weight[0,:10])

            for p in w_encoder.parameters():
                    pass
                    # print(p.grad)
                    #print(p)
            ll = loss.cpu().detach().numpy()
            rl = rec_loss.cpu().detach().numpy()
            cl = cos_loss.cpu().detach().numpy()
            writer.add_scalar('Loss/train',ll, step +len(dataloader)*e)
            writer.add_scalar('Loss/rec',rl, step +len(dataloader)*e)
            writer.add_scalar('Loss/cos',cl, step +len(dataloader)*e)
    
            out_token = torch.argmax(out.logits[0],dim=1)
            if step % checkpoint_step == 0:
                
                comp_cosine(w_emb,m_emb,pred,t_emb)
                print()
                print(OUTPUT_DIR)
                torch.save(w_encoder, os.path.join(OUTPUT_DIR,"checkpoint"))

                # for p in w_encoder.parameters():
                #     print(p)

if __name__ == "__main__":
    main()