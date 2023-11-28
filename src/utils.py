import json
import re
from pprint import pprint

import pandas as pd
from huggingface_hub import notebook_login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    top_k_top_p_filtering,
)
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.embeddings import HuggingFaceBgeEmbeddings

from torch import cuda, bfloat16
import torch.nn.functional as F
import transformers
from langchain.llms import HuggingFacePipeline
import torch
from torch import nn
from datasets import Dataset,DatasetDict



from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import pickle


from loss import compute_loss,compute_length_penalty,length_penalty_by_eos

access_token = "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"
#device = "cuda:0" if torch.cuda.is_available() else "cpu"






















def load_data():
    """ Bad code fix """
    with open("../data/validation_01.pkl", "rb") as input_file:
        val = pickle.load(input_file)

    with open("../data/train_01.pkl", "rb") as input_file:
        train = pickle.load(input_file)

    with open("../data/test_01.pkl", "rb") as input_file:
        test = pickle.load(input_file)

    return DatasetDict({
        "train":Dataset.from_dict(train),
        "test":Dataset.from_dict(test),
        "val":Dataset.from_dict(val),
        })





def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()



def prepare_data(batch,tokenizer,ctokenizer,device, max_answer_token_size = 100):
    """
    Set max_answer_token_size to -1 if you want no limit.
    """

    # Set truncation strategy based on max length instructions.
    if max_answer_token_size <= 0:
        padding = True
        truncation = False
        max_length = None
    else:
        padding = 'max_length'
        truncation = True
        max_length = max_answer_token_size

    inputs = ctokenizer(batch['questions'])
    max_lengths = torch.sum(inputs['attention_mask'],dim=1) - 1  - ctokenizer.get_token_shift()# -1 is to subtract the <s> token
    outputs = tokenizer(batch['answers'],
                    return_tensors='pt', 
                    add_special_tokens = False,# No special tokens for output.
                    truncation = truncation,
                    padding=padding,
                    max_length = max_length,
                    ) 

    return {
        'x':inputs['input_ids'],
        'y':outputs['input_ids'],
        'x_att':inputs["attention_mask"],
        'y_att':outputs["attention_mask"],
        'x_max_lengths':max_lengths,
        "condenser_mags":inputs['mags'],
    }



def batch_text_to_emb(text,model,tokenizer):
    ''' Take a batch of text and convert it into the word embeddings that will be
    used for the LLM.'''
    x = tokenizer(text,return_tensors='pt', padding=True)['input_ids']
    x = model.get_input_embeddings()(x)
    return x



def add_start_token_embedding(x,att,embedding_table,start_token = 1,device = 'cuda'):
    # add start embs
    s_emb = embedding_table(torch.tensor([1],device = device)).unsqueeze(0)
    s_emb = torch.tile(s_emb,dims = (x.shape[0],1,1))
    x = torch.concat([s_emb,x],dim = 1)
    # add start att.
    s_att = torch.tile(torch.tensor([[1]],device = device),dims = (x.shape[0],1))
    att = torch.concat([s_att,att],dim = 1)
    return x,att


#@torch.autocast(device_type="cuda")
def training_step(x,y,x_att,y_att,max_lengths,mags, #I/O
                  model,encoder_model,embedding_table,ctokenizer, # Models
                  ignore_tokens,device,generation_max=1000,insert_rand_eos = True): # other
    '''
    The goal of the optimization process is to generate condensed embeddings, equivalent 
    model embedding representations that take up less token space from the encoding model. 

    This training step assumes the dataset is a set of inputs/outcomes from a single model.

    The process:
        1. generate an embedding from the encoding model. This is done by feeding the original input
        into the encoding model and limiting the output to the size of the input.
        2. Taking that embedding and concatenating it with the outcome from the original model.
        3. Capturing the next word loss on the outcome part of that model by running it through the
        original model with gradients.


    Inputs:
        x - orignal model input tokens
        y - target outcome tokens (produced before hand from the orginal model)
        x_att = input token attention mask
        y_att = target outcome attention mask
        model = original model
        encoder_model = model being trained to encode embeddings
        ignore_tokens = a list of tokens to ignore in the loss (padding tokens etc.)
        generation_max = maximum generation size.
        target_compression_scale (0-1) = used to scale the length loss penalty. 
            compression target is the ideal outcome length where:
              encoder_token_length = original_token_length*target_compression_scale


    Outputs:
        loss: loss tensor for optimization
        out: model artifacts from the train step, use for debugging and indepth logging        
    '''
    
    #Preamble. Grab any nec. values
    max_x_length = x.shape[1] # m

    # print('CHECK EMB')
    # print(embedding_table(torch.tensor([32001])))
    # print(embedding_table(torch.tensor([32000])))
    # print(embedding_table(torch.tensor([32002])))    


    # GENERATE CONDENSED EMBEDDINGS USING THE ENCODER MODEL AND THE INPUT TOKENS!
    with torch.no_grad():
        x_enc =  ctokenizer.extract_embeddings(x,embedding_table)
        x_emb,x_emb_att = run_encoding_generation(x_enc,x_att,
                                                  encoder_model,
                                                  max_lengths,
                                                  device,generation_config = model.generation_config,
                                                  gen_length_shift = ctokenizer.get_token_shift(),
                                                  insert_rand_eos = insert_rand_eos
                                                  )


    
    # RUN x_EMB through encoder to get gradients!!!!
    x_emb,x_emb_att = add_start_token_embedding(x_emb,x_emb_att,embedding_table,device = device)    
    x_emb,x_emb_att,length_logits = encoder_model.process_training_embedding(x_emb.half(),x_emb_att,device =device)

    #### PROCESS THE GENERATED EMBEDDING WITH THE DESIRED OUTCOME
    y_emb = embedding_table(y) #grab the word embeddings for y
    # prepend x emb with BOS token
    bos_tensor = model.generation_config.bos_token_id*torch.ones([x.shape[0],1],dtype = torch.int)
    bos_emb = embedding_table(bos_tensor)
    bos_att = torch.ones([x.shape[0],1],dtype = torch.int)
    # concat inputs.
    xy_input = torch.concatenate([bos_emb.to(device),x_emb.to(device),y_emb.to(device)],dim=1) # put together our desired outcome string for model processing
    xy_att = torch.concatenate([bos_att.to(device),x_emb_att.to(device),y_att.to(device)],dim=1) # same for attention mask.    out = encoder_model(inputs_embeds = xy_input,attention_mask = xy_att) # process outcomes.


    # RUN MODEL - We want these to come from the OG model. This is where the inputs should finally go!!!
    out = model(inputs_embeds = xy_input.half(),attention_mask = xy_att)

    # CALCULATE LOSS TERMS
    # window logits and labels correctly
    # Model predicts at token +1 so first token (BOS) has nothing, but is not included in max length SO
    # correct window is [max length,-1] : Which is the start of the Y string logits to the end of the Y string
    # we ignore the final token because it is the next token which we have no label for.
    logits = out.loss['logits'][:,max_x_length:-1] 
    labels = y.to(device)

    # Flatten
    # logits = logits.reshape(-1, model.config.vocab_size)
    # labels = labels.view(-1)
    recon_loss = compute_loss(logits,labels,y_att)

    # length_loss = compute_length_penalty(x_emb_att,
    #                                      max_lengths,
    #                                      mags,
    #                                      zero_at_pole = True,
    #                                      device = device)
    #print(max_x_length)
    length_loss = length_penalty_by_eos(
        length_logits,
        x_emb_att,
        max_lengths,
        mags,
        model.generation_config
        )
    


    # SUM UP ALL LOSS TERMS    
    loss = recon_loss + length_loss # Currently only recon for testing is being used. Should be able to reproduce 
    #print(recon_loss,length_loss)
    response_length = x_emb_att.sum(dim=1,keepdims=False)
    ideal_lengths = (max_lengths.unsqueeze(1).to(length_logits.device)*mags.to(length_logits.device)).int() +1
    #print(response_length,ideal_lengths.squeeze(1))


    # ADD ANY ARTIFACTS YOU WANT FOR DEBUGGING AND LOGGING
    out['recon_loss'] = recon_loss.cpu().detach().numpy()
    out['length_loss'] = length_loss.cpu().detach().numpy()
    out['max_x_length'] = max_x_length#.cpu().detach().numpy()
    out['x_emb'] = x_emb
    out['x_emb_att'] = x_emb_att
    out['xy_input'] = xy_input
    out['xy_att'] = xy_att
    out['response_len'] = response_length.cpu().detach().numpy()
    out['ideal_response_len']= ideal_lengths.squeeze(1).cpu().detach().numpy()
    xy_att
    return loss,out
    

def length_to_mask(length, device, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), int(max_len)) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


#@torch.autocast(device_type="cuda")
def run_encoding_generation(
        input_embeddings,input_att,
        model,
        max_lengths,
        device,generation_config,fill_token = 0,gen_length_shift = 0,insert_rand_eos = True,rand_eos_freq = 0.05,**model_kwargs):
    '''
    Generate a set of word embeddings that is equal to or less than the input
    string tokens. This is designed to train an encoding model that will output
    more condense input embeddings that hold the same information.

    TODO thie function has a real problem with memory efficiency and doesn't free up the cuda mem after done...
        Look into how to fix...
    TODO make sure we carry over values so we are efficient.
    TODO test to make sure this is working correctly... I am not getting what I expected out of just running it and decoding.
        - see what happens when we train with eos token...
    '''
    ### PREAMBLE - enstantiate objects and counters.
    # TODO: TRY: insert a trainable prefix token here. (maybe begin / end tokens)
    #input_embeddings = embedding_table(input_tokens) # grab embeddings for input_tokens
    batch_size = input_embeddings.shape[0]
    input_length =  input_embeddings.shape[1]
    eos_token_id_tensor = torch.tensor([generation_config.eos_token_id]).to(device) # grab EOS token as tensor.
    max_length_mask = length_to_mask(max_lengths,device, dtype = torch.long).to(device) # Create max_length mask
    
    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

    #create output tensor and counters
    generated_embeddings = torch.zeros(input_embeddings.shape).to(device) # Empty embedding tensor to be filled during generation
    generated_tokens = fill_token*torch.ones(input_embeddings.shape[:2]).to(device) # Empty token tensor to be filled during generation
    gen_stop = torch.zeros(input_embeddings.shape[:2]).to(device) 
    
    # Generate a max steps of largest input tokens
    for i in range(input_length-1-gen_length_shift): # -1 for start token and -3 for prefix tokens.
        #print(i)
        #####
        # TODO: much more efficient way of doing this. See generate code. Need to pass historical values to save time.
        # I now have same code as greedy mostly... need to test if it is efficient or I am missing some magic.

        #print("att",input_att[1])
        #print(torch.max(torch.isnan(input_embeddings[1])))
        #print(torch.max(input_embeddings))
        # print(torch.max(input_embeddings[1]))
        # print(torch.max(input_embeddings[0]))
        # print(torch.max(input_embeddings[3]))

        
        # print(torch.max(torch.isnan(input_embeddings)))
        
        #print(input_embeddings.dtype,input_att.dtype)
        # for p in model.parameters():
        #     print(p.dtype)
            
        # Generate outputs and extract embeddings for next token
        next_embedding, outputs = model(input_embeddings,input_att,device = device) # Run model using embeddings NOT tokens.
        #next_embedding = outputs.hidden_states[-1][:,-1,:].to(device) # Last embedding layer, last token.
        #next_embedding = w_encoder(next_embedding.float()).half()
        #next_embedding = F.normalize(next_embedding)

        ### CHECK FOR STOPPING CONDITIONS, if we hit one stop generating
        # Check for eos token and stop if found (greedy)
        # Could try using a threshold... if EOS is over xyz STOP.
        next_token_logits = outputs.logits[:,-1]
        next_token = torch.argmax(next_token_logits, dim=-1).to(device) # Simple greedy pred.

        ### Insert random EOS to encourage early stopping
        if insert_rand_eos:
            cond = torch.rand([batch_size]) < rand_eos_freq
            next_token[cond] = generation_config.eos_token_id

        # Check if we find eos tokens to finish sequence generation.
        # TODO: Do we want to add any other stop tokens in here??????
        # Checked. this code is working correctly. it checks to see if next token is EOS token and is used to 
        # set stop mask to 1.
        unfinished_sequences = unfinished_sequences.mul(
                     next_token.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                 )
        #print(unfinished_sequences)


        # mask out already finished sequences.
        #print(unfinished_sequences,max_length_mask[:,i],i)
        stop_mask = (unfinished_sequences*max_length_mask[:,i]).to(device) # gives [0,1] mask for embedding at position X. (Can we just apply at end?)
        next_token = next_token * stop_mask + generation_config.pad_token_id * (1 - stop_mask) # If over max length, pad the encoding.

        # tag eos tokens to finish sequence generation.
        unfinished_sequences = unfinished_sequences.mul(
                     next_token.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                 )


        ### Insert results into holder tensorts
        generated_embeddings[:,i,:] = next_embedding*stop_mask.unsqueeze(-1)        
        generated_tokens[:,i] = next_token
        gen_stop[:,i] = stop_mask
        #print(gen_stop)

        # Prepare inputs for next token prediction
        input_embeddings = torch.cat([input_embeddings, torch.unsqueeze(next_embedding,1)], dim=1)
        next_att = torch.ones([input_att.shape[0],1]).to(device)*stop_mask.unsqueeze(-1) # attention mask and stop mask are same.. I should make the same.
        input_att =  torch.cat([input_att.to(device), next_att.to(device)], dim=1)
        
        # stop when each sentence is finished
        if stop_mask.max() == 0:
            break

    #return generated_embeddings,input_embeddings,outputs
    return generated_embeddings,gen_stop



def decode_logits(logits,tokenizer):
    ''' simple greedy decoding '''
    recreate_tokens = []
    for l in range(logits.shape[1]):
        last_logits = logits[:, l, :]
        final_token = torch.argmax(last_logits, dim=-1)
        # #Filter out the top results - using greedy
        # filter = top_k_top_p_filtering(last_logits, top_k=1, top_p=1.0)
        # # Finding probabilities using softmax function
        # probabilities = nn.functional.softmax(filter, dim=-1)
        # # Applying multinomial
        # final_token = torch.multinomial(probabilities, num_samples=1)
        recreate_tokens.append(final_token.unsqueeze(dim = 1))

    #print(recreate_tokens)
    # Applying cat function and decode
    
    output = torch.cat(recreate_tokens, dim=1)
    answer = tokenizer.batch_decode(output)
    return answer, output

def num_tokens_from_string(string: str, tokenizer) -> int:
    """Returns the number of tokens in a text string."""
    # encoding = tiktoken.get_encoding(encoding_name)
    encoding =  tokenizer(string, return_tensors="pt")
    num_tokens = encoding['input_ids'].shape[1]
    return num_tokens



def define_embedder(model_name = 'BAAI/bge-large-en-v1.5',instructions_template = "Represent this sentence for searching relevant passages:"):
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    embedder = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction=instructions_template
    )
    return embedder




def maxpool_sentence_tokens(input_tokens, embedding_table,kernel = 2,stride = 2):
    # grab the embeddings from the token list
    input_embeddings = embedding_table(input_tokens)
    # transpose in for max pool, execute max pool, transpose back
    x = torch.transpose(input_embeddings,1,2)
    #m = torch.nn.MaxPool1d(window, stride=stride)
    #mp_embed = m(mp_embed)
    x = torch.nn.functional.max_pool1d(x,kernel,stride = stride)
    x = torch.transpose(x,1,2)
    return x



def create_model_and_tokenizer(model_id,bnb_config=None,device_map = 'auto'):
    
    if bnb_config is None:
        bnb_config = BitsAndBytesConfig(
            #load_in_4bit=True,
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload = True,
            #bnb_4bit_quant_type="nf4",
            #bnb_4bit_compute_dtype=torch.float16,
        )


    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map=device_map,
        token=access_token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id,token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def create_stop_criteria(tokenizer,device,stop_list = None):
    if stop_list is None:
        stop_list = ['\nHuman:', '\n```\n']

    stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]    
    stop_token_ids = [torch.LongTensor(x) for x in stop_token_ids] #.to(device)
    #print(stop_token_ids)
    
    # define custom stopping criteria object
    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False
    
    return StoppingCriteriaList([StopOnTokens()])

    

def create_Llama_LC_pipeline(model_id,device,return_all = False):
    print('loading model: ', model_id)
    model, tokenizer = create_model_and_tokenizer(model_id)


    # define custom stopping criteria object
    stopping_criteria = create_stop_criteria(tokenizer)


    print('create pipeline')
    pipeline = transformers.pipeline(
        model=model, 
        tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model rambles during chat
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    print('cast to langchain')
    llm = HuggingFacePipeline(pipeline=pipeline)

    if return_all:
        return llm, model, tokenizer, pipeline
    else:
        return llm


