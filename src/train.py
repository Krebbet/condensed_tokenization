#accelerate launch --config_file ../config/ds_config.yaml train.py
# TOP PRIORITIES:
# -- Try Llama encoder with only ONE compression rate.
# -- Try Phi2 to get embeddings working with a bigger encoder.
# -- play around with minstral 7 and 8x7. Can I get this one working? Also try using this as base model.

# -- Try larger embedding model where WE optimization will work, Llama causes NAN for some unknown reason....
# -- make embedding model different then base model (and get WE optimizing! This may be why we can't get length loss working...)
# -- check length loss is working... looks like lengths are not what they should be when loss is 0.
# TO DO: 
# - Refactor code:
#   - Clean up training file
#   - put training step and related code into something other than utils.
#   - Create model config and save to model save when checkpointing
#   - COmments code
#   - TODO's should be eliminated.
# - Test weighted loss function to make sure - weighting is working AND attention mask is working.
# - Evaluate current results. (Check how others have been evaluating.)
# - Evaluate the data generation step. Make sure I/O is being generated appropriately.
# - figure out how to get ctokens to optimize!!!!! This is super frustrating.  
# - Once above done extend length of answer tokens significantly.
# - get deepspeed working correctly on this model (or any other model....)
# - Add in different levels of text condensation
# - Create a training method where we encode random chunks of text in a longer input.
#   Idea being we can use this at any point in an text. 
# - Extend to encoding large sets of text (500+ tokens)
#   - Is it best to do this in chunks or all at once?

import os
import torch
from datasets import load_dataset
import numpy as np
import pickle as pkl


from peft import PeftConfig, LoraConfig

import einops
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments
)
from peft.tuners.lora import LoraLayer
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training


from torch.utils.data import DataLoader

from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter



#local
from utils import (
    create_model_and_tokenizer, 
    training_step,
    prepare_data,
    load_data,
    decode_logits

)

from condenser_tokens import CondenserTokenizer
from architectures import (
    WordEmbeddingAligner,
    TokenCondenserWrapper,
    NeuralNetwork, # remove once I train a new WordEmbeddingAligner
)

from model_utils import (
    checkpoint_model_objects,
    load_existing_model,
    create_encoder_objects,
    create_roberta_encoder,
    create_minstral_objects,
)
    
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
from utils import (
    SAVE_STATE_PATH,
    SAVE_STATE_META_PATH,
    SAVE_STATE_CTOKENIZER_PATH,
    SAVE_ENCODER_PATH
)

#################################
"""
There are the following objects:
 
model / tokenizer: need to store what these are but they don't change

encoder_model: right now it is a peft version of base, eventually want this to be 
    any LM I want (ideally small).

w_encoder: small FC model to align output embeddings to model. Should be optional.

ctokenizer: object to store additional embeddings (not nec.) and to handle 
    adding spec. tokens for variable encoding size.

    NOTE: Embs do not optimize and cant figure out why... If I get them in I get NAN's


TokenCondenserWrapper: wraps encoder model with w encoder...


Q: how should I organize these objects to abstract some of what I am doing logically and
cleanly.

"""

### TRAINING TRIES:
# Elongate weighted CBE LOSS ( acc goes down from token lik 3-8)
# try add_y_bos=True in train step. Will adda pad token between emb and y
#       first token ACC is bad... maybe because LLM needs a token to normalize behaviour?

def main():
    # Setup Huggingface Accelerate Object
    #accelerator = Accelerator()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #device = accelerator.device
    print(device)

    DEBUG = False
    if DEBUG:
        print("CURRENTLY IN DEBUG MODE!!!!")


    #exclude_param_list = ['base_model.model.lm_head.weight','base_model.model.model.embed_tokens.weight']
    exclude_param_list = []

    # RUN VARS
    ENCODER_TYPE = 'base_peft' #roberta #base_peft #minstral
    USE_CTOKENIZER= False
    OVERFIT = False # For debugging
    ADD_Y_BOS = False # testing this? - other issues with first token... try those first.
    batch_size = 2
    lr=1e-4
    max_answer_token_size = 50
    epochs = 100
    if DEBUG:
        checkpoint_rate = 1 # while debugging.... do like 500 later.
    else:
        checkpoint_rate = 500


    OUTPUT_DIR = f'../models/token_encoder/e{ENCODER_TYPE}_bs{batch_size}_lr{lr:.0E}_OnlyComp20p_38'
    print('saving model to: ',OUTPUT_DIR)


    # MODEL PARAM
    #W_ENCODER_DIR = '../models/w_encoder/new_lr1e-4_L2_2L1024_L2_fudge0.03_00'
    #W_ENCODER_DIR = '../models/w_encoder/new_lr1e-4_L2_2L1024_L2_fudge0.03_01'
    W_ENCODER_DIR = None
    peft_config = LoraConfig(
            r=16,  # dimension of the updated matrices
            lora_alpha=16,  # parameter for scaling
            #target_modules= None,
            #target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","embed_tokens","lm_head"],
            #target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.1,  # dropout probability for layers
            bias="none",
            task_type="CAUSAL_LM",
            inference_mode = False,
            # NAN's occur when included in training.. track down why....
            #modules_to_save=['lm_head','embed_tokens'], #TODO check this out and see how I can use to just train the new tokens. 
            #modules_to_save=['embed_tokens'], # Try just the emb tokens....
            modules_to_save=[],
            # Could also try some other optimizers... maybe it is something to do with the WAdam build....
    )


    # Load target model
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    model, tokenizer = create_model_and_tokenizer(model_id,device_map=None)




  # Load target dataset.
    dataset = load_data()
    #dataloader = DataLoader(dataset['train'], batch_size=batch_size,shuffle = True)
    val_dataloader = DataLoader(dataset['val'], batch_size=batch_size,shuffle = True)
    if OVERFIT:
        dataloader = DataLoader(dataset['train'], batch_size=batch_size,shuffle = False)
        overfit_batch = next(iter(dataloader))


    if ENCODER_TYPE == 'roberta':
        print('Using Roberta Model as encoder.')
        encoder_model_id = 'xlm-roberta-base'
        encoder_model,etokenizer,ctokenizer,param = create_roberta_encoder(encoder_model_id,tokenizer,None,peft_config,device = device,verbose = True)
    elif ENCODER_TYPE == 'base_peft':
        print('Model being created is PEFT of Base')
        # Create all model objects.
        model,encoder_model,ctokenizer,param = create_encoder_objects(model,tokenizer,W_ENCODER_DIR,peft_config,device = device)
        etokenizer = tokenizer # in this case same tokenizer...
        encoder_model.base_model.print_trainable_parameters()
    elif ENCODER_TYPE == 'minstral':
        print('Model being created is PEFT of Minstral')
        # Create all model objects.
        model,encoder_model,ctokenizer,param,etokenizer = create_minstral_objects(model,tokenizer,W_ENCODER_DIR,peft_config,device = device)
        encoder_model.base_model.print_trainable_parameters()


    # Check if model exists, if so load state
    # FIX: looks like reload didnt really work!!!!
    # check and validate.
    # I bet wencoder is not working correct....
    epoch_o = count_o = 0
    if os.path.isdir(OUTPUT_DIR):
        if not DEBUG:
            print('Loading existing ENCODER MODEL.')
            encoder_model,ctokenizer,epoch_o,count_o = load_existing_model(encoder_model,OUTPUT_DIR)
            model = ctokenizer.add_model(model) # Clunky, revisit...
    else:
        os.mkdir(OUTPUT_DIR)
    
    

    # Get rid of once you resolve embed problems.
    #embedding_table_o = encoder_model.get_input_embeddings().original_module
    #embedding_table =  encoder_model.base_model.get_input_embeddings().modules_to_save.default.lora_embedding_A.default
    # embedding_table_o = encoder_model.base_model.get_input_embeddings()
    embedding_table =  encoder_model.base_model.get_input_embeddings()
    #TODO: try bundling w_encoder into a peft model as well????? --> get the floats correct... Maybe?

    if DEBUG:
        for n,p in encoder_model.named_parameters():
            if p.requires_grad:
                print(n)
    
    # Very stange behaviour causing and error here:
    # 1) I want these modules to optimize but give NAN.
    # 2) When wrapping a peft module it doesnt seem to know what to include...
    for n,p in encoder_model.base_model.named_parameters():
        if n in exclude_param_list:
            print("Parameters found and excluded: ",n)
            p.requires_grad = False


    params = [
        {'params': encoder_model.parameters(), 'lr': lr, 'weight_decay': 0},
        #{'params': param, 'lr': lr, 'weight_decay': 0},
        #{'params': encoder_model.w_parameters, 'lr': lr, 'weight_decay': 0},
        #{'params': encoder_model.base_model.parameters(), 'lr': lr, 'weight_decay': 0}, #TODO ensure w_encoder params are passed.
        {'params': ctokenizer.parameters(), 'lr': lr, 'weight_decay': 0},
            ]

    # optimizer TODO look into best scheduling options (also oscilatory )
    opt = torch.optim.AdamW(params) 
    scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=100)    

    # Accelerate Components with huggingface accelerate
    # The accelerate config defines what changes are actually made here
    # encoder_model, opt, dataloader, scheduler = accelerator.prepare(
    #     encoder_model, opt, dataloader, scheduler
    # )


    # Needed if loading a preexisting run
    # accelerator.load_state(None)
    encoder_model.train()
    model.train()

    # tb writer
    train_writer = SummaryWriter(log_dir = os.path.join( OUTPUT_DIR ,'logs/train'))
    val_writer = SummaryWriter(log_dir = os.path.join( OUTPUT_DIR ,'logs/val'))



    ll = 999
    rl = 999
    response_dif = 9999
    nan_artifacts = [] # get rid of after debug.
    # c_o = torch.mean(torch.square(ctokenizer.embed))
    # e_o = torch.mean(torch.square(embedding_table_o.weight))
    #e_o = torch.mean(torch.square(embedding_table.weight))
    #e_o = torch.mean(torch.square(embedding_table.data))
    #print(e_o)
    #print(embedding_table.data)
    #with torch.autocast(device_type='cuda'):
    count = 0 # Just have one continous count... easier...
    for epoch in range(epochs):
        

        for i,ds in enumerate(dataset['train']):
            print(f"Running {i} of {len(dataset['train'])} data sets for epoch {epoch}")
            
          
            dataloader = DataLoader(ds, batch_size=batch_size,shuffle = True)
            progress_bar = tqdm(enumerate(dataloader))
            for j,batch in progress_bar:
                torch.cuda.empty_cache()
                count += 1
                if OVERFIT:
                    batch = overfit_batch

                progress_bar.set_description(f"recon loss: {rl:.2f} length loss: {ll:.4f}, resp dif ave: {response_dif:.2f} epoch: {epoch}")

                
                opt.zero_grad()
                sample_answer = batch['answers'][0]
                sample_q = batch['questions'][0]
                


                batch = prepare_data(batch,tokenizer,ctokenizer,device,max_answer_token_size = max_answer_token_size)
                #print('MAX LENGTHS')
                #print(batch['x_max_lengths'])
                loss,artifacts = training_step(
                    batch['x'],
                    batch['y'],
                    batch['x_att'],
                    batch['y_att'],
                    batch['x_max_lengths'],
                    ctokenizer.get_mag_tensor(batch['condenser_mags'],device = device),
                    ctokenizer.get_scale_tensor(batch['condenser_mags'],device = device),
                    model,
                    encoder_model,
                    ctokenizer,
                    device,
                    generation_max=400,
                    #insert_rand_eos = ll > RAND_EOS_LIM
                    insert_rand_eos = False,
                    add_y_bos = ADD_Y_BOS,
                    use_ctokenizer=USE_CTOKENIZER,
                )

                
                if torch.isnan(loss):
                    print('\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx')
                    print(count)
                    print('NAN FOUND\n loss:')
                    print(loss)

                    ll = artifacts['length_loss']
                    rl = artifacts['recon_loss']
                    ideal_response = artifacts['ideal_response_len']
                    actual_response = artifacts['response_len']
                    response_dif = np.mean(np.absolute(actual_response - ideal_response))
                    print('length loss: ',ll)
                    print('reconstruction loss: ',rl)
                    print('ideal length: ',ideal_response)
                    print('actual length: ',actual_response)
                    print('response dif: ',response_dif)

                    print('param')
                    print('\nNAN Params START')
                    for name, param in encoder_model.named_parameters():
                        #if param.requires_grad:
                        if torch.sum(torch.isnan(param.data)) > 0 :
                            print (name, param.data)            
                                
                    print('\nNAN Params END')

                    #nan_artifacts.append((artifacts,batch))
                    x = torch.isnan(artifacts['logits'])
                    x = torch.max(torch.max(x,dim = 1).values,dim=1).values


                    print('all logits')
                    print(artifacts['logits'])
                    print('batch logits found NAN:')
                    print(x)

                    print('x emb. token input')
                    print(batch['x'])
                    print('labels')
                    print(batch['y'])



                    # should output to some log so I can check everything...
                    #exit()
                else:

                    #accelerator.backward(loss)
                    loss.backward()
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()



                    # extract follow vars
                    ll = artifacts['length_loss']
                    rl = artifacts['recon_loss']
                    ideal_response = artifacts['ideal_response_len']
                    actual_response = artifacts['response_len']
                    response_dif = np.mean(np.absolute(actual_response - ideal_response))
                    #print(response_dif)
                    # print("IDEAL",ideal_response)
                    # print("ACTUAL",actual_response)



                    #print(ll,rl)                    
                    # write loss terms to logger
                    #log_counter = count + count_o + len(dataloader)*(epoch + epoch_o) # o's added for restarts
                    log_counter = count + count_o
                    train_writer.add_scalar('Loss/loss', loss.cpu().detach().numpy(), log_counter)
                    train_writer.add_scalar('Loss/length',ll, log_counter)
                    train_writer.add_scalar('Loss/recon', rl, log_counter)

                    # STILL haven't confirmed that embeddings are optimizing....
                    # print(encoder_model.base_model.get_input_embeddings().weight[-5:,:5])
                    # print('embed scalar: ',torch.mean(torch.square(encoder_model.base_model.get_input_embeddings().weight)))
                    # print(encoder_model.base_model.get_input_embeddings().weight.shape)

                    

                if count % checkpoint_rate == 0:
                    



                    # print progress
                    #try:
                    # TODO - FIX TO MAKE GREEDY.
                    
                    print("QUESTION: \n",sample_q)
                    ans = tokenizer.decode(tokenizer(sample_answer)['input_ids'][1:max_answer_token_size +2])

                    print("LABEL   : \n",ans)
                    pred = artifacts['logits'][:,artifacts['max_x_length']:-1] 
                    a,o = decode_logits(pred,tokenizer)

                    print("\nPRED: \n",a[0])
                    #except:
                    #    print("EXCEPTION: Problem predicting tokens, model probably defunc.")
                        
                    # Save checkpoints
                    meta_data = {
                            'epoch':epoch,
                            'count':count,
                            'w_encoder_path':W_ENCODER_DIR
                    }                
                    checkpoint_model_objects(encoder_model,ctokenizer,meta_data,OUTPUT_DIR)

                    print('OTHER INTERESTING READOUTS')
                    print('ideal length: ',ideal_response)
                    print('actual length: ',actual_response)
                    print('response dif: ',response_dif)



                    # write out random sampling of logits.
                    print("Rand logits")
                    xx = pred.cpu().detach().numpy()
                    rand_logits = xx[np.random.choice(xx.shape[0],50)]
                    
                    print(rand_logits.flatten())
                    print(rand_logits.shape)
                    # write out some pred logits.
                    print("Pred Logits")
                    ans_logits = np.take(xx,o[0].cpu().detach().numpy())
                    print(ans_logits)

                    print("rand / ans logit mean")
                    print(np.mean(rand_logits),np.mean(ans_logits))

                    #print('Emb table dif from original.')
                    #print('embed_p dif: ',e_o - torch.mean(torch.square(encoder_model.base_model.get_input_embeddings().weight)))
                    #print(embedding_table.weight[0:10,0:10])
                    #embedding_table.data

                    #print("Lora projection on Embed:", torch.mean(torch.square(embedding_table.data)))
                    #print(embedding_table.data)

                    # print model name to follow during training.
                    print()
                    print(OUTPUT_DIR)
                    # print('ctoken  dif: ',c_o - torch.mean(torch.square(ctokenizer.embed)))
                    # print('embed_o dif: ',e_o - torch.mean(torch.square(embedding_table_o.weight)))
                    
                    # Run Val
                    with torch.no_grad():
                        batch = next(iter(val_dataloader))
                        batch = prepare_data(batch,tokenizer,ctokenizer,device,max_answer_token_size = max_answer_token_size)
                        loss,artifacts = training_step(
                            batch['x'],
                            batch['y'],
                            batch['x_att'],
                            batch['y_att'],
                            batch['x_max_lengths'],
                            ctokenizer.get_mag_tensor(batch['condenser_mags'],device = device),
                            ctokenizer.get_scale_tensor(batch['condenser_mags'],device = device),
                            model,
                            encoder_model,
                            ctokenizer,
                            device,
                            generation_max=400,
                            insert_rand_eos = False,
                            add_y_bos = ADD_Y_BOS,
                            use_ctokenizer=USE_CTOKENIZER,

                        )
                        val_writer.add_scalar('Loss/loss', loss.cpu().detach().numpy(), log_counter)
                        val_writer.add_scalar('Loss/length',artifacts['length_loss'], log_counter)
                        val_writer.add_scalar('Loss/recon', artifacts['recon_loss'], log_counter)


        
        # # Save PEFT model/final accelerator state
        #accelerator.save_state(os.path.join(OUTPUT_DIR,"final_checkpoint"))
        #encoder_model = accelerator.unwrap_model(encoder_model)
        encoder_model.save(OUTPUT_DIR)
        print('done')



if __name__ == "__main__":
    main()