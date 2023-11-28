#accelerate launch --config_file ../config/ds_config.yaml train.py
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
from peft import LoraConfig, TaskType, get_peft_model

from trl import SFTTrainer

from torch.utils.data import DataLoader

from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter





#local
from utils import (
    create_model_and_tokenizer, 
    maxpool_sentence_tokens,run_encoding_generation,
    training_step,
    run_encoding_generation,
    prepare_data,
    print_gpu_utilization,
    load_data,
    decode_logits

)

from condenser_tokens import CondenserTokenizer
from architectures import (
    WordEmbeddingAligner,
    TokenCondenserWrapper,
    NeuralNetwork, # remove once I train a new WordEmbeddingAligner
)

    
# device = "cuda:0" if torch.cuda.is_available() else "cpu"



def main():
    # Setup Huggingface Accelerate Object
    accelerator = Accelerator()
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = accelerator.device
    print(device)


    OVERFIT = False # For debugging

    batch_size = 4
    lr=1e-4
    max_answer_token_size = 50
    epochs = 100
    checkpoint_rate = 100
    RAND_EOS_LIM = 0.5

    # Define Paramters 
    # FIX: ctokens still not getting optimized!!!!!!!!!!!!!!!!!!
    #   - They can now be directly optimized, but for some reason no signal is getting back to them during training.
    # - TODO: into config
    # - copy into save
    # - make name be based on params.
    # - combine encoder/w_encoder for easy load save.
    OUTPUT_DIR = f'../models/token_encoder/EOSembLoss_gradEnc_bs{batch_size}_lr{lr:.0E}_EOSinj05lim{RAND_EOS_LIM:.2f}_06'
    print(OUTPUT_DIR)
    if not os.path.isdir(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)


    # Load w_encoder - preconditioned to align output embedding space to word embedding space.
    W_ENCODER_DIR = '../models/w_encoder/lr1e-5_L2_2L512_Adam_fudge0.03_25_continue'
    #W_ENCODER_DIR = '../models/w_encoder/lr1e-5_L2_2L1024_L2_fudge0.03_28'


    w_encoder = torch.load(os.path.join(W_ENCODER_DIR,"checkpoint"),map_location = device)

    #lr1e-5_L2_2L512_Adam_fudge0.03_25_continue



    # Load target dataset.
    dataset = load_data()
    dataloader = DataLoader(dataset['train'], batch_size=batch_size,shuffle = True)
    if OVERFIT:
        dataloader = DataLoader(dataset['train'], batch_size=batch_size,shuffle = False)
        overfit_batch = next(iter(dataloader))


    # Load target model
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    model, tokenizer = create_model_and_tokenizer(model_id,device_map=None)

    # add condenser tokens and add dummy weights to model.
    ctokenizer = CondenserTokenizer(tokenizer)
    model = ctokenizer.add_model(model)

    # Load encoder model using Peft.
    # TODO COnfig this.
    peft_config = LoraConfig(
        r=8,  # dimension of the updated matrices
        lora_alpha=32,  # parameter for scaling
        target_modules= None,
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode = False,
        xxx modules_to_save=['lm_head','embed_tokens'], #TODO check this out and see how I can use to just train the new tokens. 
    )
    encoder_model = get_peft_model(model, peft_config)
    encoder_model.print_trainable_parameters()
    embedding_table = encoder_model.get_input_embeddings()
    #TODO: try bundling w_encoder into a peft model as well????? --> get the floats correct... Maybe?
    encoder_model = TokenCondenserWrapper(encoder_model,w_encoder)



    #embedding_table = model.get_input_embeddings()

    #define params that should be optimized
    # encoder, w_encoder and ctokenizer params! Base model only used to generate responses.
    # COMMENTARY: This is an aweful way to pass parameters to the optimizer.... come on pytorch!
    # params = [
    #     {'params': w_encoder.parameters(), 'lr': lr, 'weight_decay': 0},
    #     {'params': encoder_model.parameters(), 'lr': lr, 'weight_decay': 0},
    #     {'params': ctokenizer.parameters(), 'lr': lr, 'weight_decay': 0},
    #            ]

    # # optimizer TODO look into best scheduling options (also oscilatory )
    # opt = torch.optim.AdamW(params)
    # scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=100)    



    # TO DO: Should I use?
    # look into what this function is doing... a little obscur.
    #encoder_model.enable_input_require_grads()
    # look into more... seems like it allows for the grads to be loaded into CPU RAM?
    #encoder_model.gradient_checkpointing_enable()
    

    # Prepare base model. -> Can only accelerate 1 model with Deep Speed....
    # TODO experiment on how to get this to work effectively.
    # model = accelerator.prepare_model(model)


    # Needed if loading a preexisting run
    # accelerator.load_state(None)
    encoder_model.train()
    model.train()


    params = [
        #{'params': w_encoder.parameters(), 'lr': lr, 'weight_decay': 0},
        {'params': encoder_model.parameters(), 'lr': lr, 'weight_decay': 0},
        {'params': ctokenizer.parameters(), 'lr': lr, 'weight_decay': 0},
               ]

    # optimizer TODO look into best scheduling options (also oscilatory )
    opt = torch.optim.AdamW(params) 
    scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=100)    
   
    # Accelerate Components with huggingface accelerate
    # The accelerate config defines what changes are actually made here
    encoder_model, opt, dataloader, scheduler = accelerator.prepare(
        encoder_model, opt, dataloader, scheduler
    )

    # for i in range(100):
    #     opt.zero_grad()
    #     loss = torch.mean(torch.square(ctokenizer.embed.half()))
    #     #loss.requires_grad = True
    #     print(loss)
    #     loss.backward()
    #     opt.step()
    #     scheduler.step()
    #     for p in ctokenizer.parameters():
    #         print(p)
    # exit()

    # tb writer
    train_writer = SummaryWriter(log_dir = os.path.join( OUTPUT_DIR ,'logs/train'))
    #val_writer = SummaryWriter(log_dir = os.path.join( OUTPUT_DIR ,'logs/val'))


    # I DONT THINK I AM USING.. check and delete.
    ignore_tokens = tokenizer([ # This defines all the tokens to ignore during loss calculations.
        tokenizer.eos_token+
        tokenizer.pad_token+
        tokenizer.unk_token
    ],return_tensors='pt', padding=True)['input_ids'].squeeze().to(device)


    ll = 999
    rl = 999
    response_dif = 9999
    nan_artifacts = [] # get rid of after debug.
    c_o = torch.mean(torch.square(ctokenizer.embed))
    #with torch.autocast(device_type='cuda'):
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader))
        for count,batch in progress_bar:
            if OVERFIT:
                batch = overfit_batch

            progress_bar.set_description(f"recon loss: {rl:.2f} length loss: {ll:.2f}, resp dif ave: {response_dif:.2f}")

            
            opt.zero_grad()
            sample_answer = batch['answers'][0]
            sample_q = batch['questions'][0]
            


            batch = prepare_data(batch,tokenizer,ctokenizer,device,max_answer_token_size = max_answer_token_size)
            loss,artifacts = training_step(
                batch['x'],
                batch['y'],
                batch['x_att'],
                batch['y_att'],
                batch['x_max_lengths'],
                ctokenizer.get_mag_tensor(batch['condenser_mags'],device = device),
                model,
                encoder_model,
                #w_encoder,
                embedding_table,
                ctokenizer,
                ignore_tokens,
                device,
                generation_max=400,
                insert_rand_eos = ll > RAND_EOS_LIM
            )
            
            if torch.isnan(loss):
                print('NAN FOUND')
                print(loss)
                nan_artifacts.append((artifacts,batch))
                x = torch.isnan(artifacts['logits'])
                x = torch.max(torch.max(x,dim = 1).values,dim=1).values

                
                print(x)
                print(batch['x'])
                print(batch['y'])
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

                #print(ll,rl)                    
                # write loss terms to logger
                log_counter = count + len(dataloader)*epoch
                train_writer.add_scalar('Loss/loss', loss.cpu().detach().numpy(), log_counter)
                train_writer.add_scalar('Loss/length',ll, log_counter)
                train_writer.add_scalar('Loss/recon', rl, log_counter)
                

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
                accelerator.save_state(os.path.join(OUTPUT_DIR,"acc/checkpoint_interim"))


                # Print out c emb drift.
                # print("C_TOKEN DRIFT: --> should be more than nothing")
                # print(torch.sum(c_tok_init - c_tokens.get_embedding_weights(embedding_table,tokenizer)))



                # print model name to follow during training.
                print()
                print(OUTPUT_DIR)
                print('ctoken dif: ',c_o - torch.mean(torch.square(ctokenizer.embed)))

        
        # # Save PEFT model/final accelerator state
        accelerator.save_state(os.path.join(OUTPUT_DIR,"final_checkpoint"))
        encoder_model = accelerator.unwrap_model(encoder_model)
        encoder_model.save(OUTPUT_DIR)
        print('done')



if __name__ == "__main__":
    main()