import os
import torch
import pickle as pkl
from peft import get_peft_model, prepare_model_for_kbit_training

from transformers import AutoTokenizer, AutoModelForMaskedLM

from utils import create_model_and_tokenizer

from condenser_tokens import CondenserTokenizer
from architectures import (
     TokenCondenserWrapper,
     NewWordEmbeddingAligner, # change name later
)
 

# Switch location to something else... maybe a config... maybe const file.
from utils import (
    SAVE_STATE_PATH,
    SAVE_STATE_META_PATH,
    SAVE_STATE_CTOKENIZER_PATH,
    SAVE_ENCODER_PATH
)

def checkpoint_model_objects(encoder_model,ctokenizer,meta_data,path):
    encoder_model.save(os.path.join(path,SAVE_STATE_PATH))
    torch.save({
        'model_state_dict': encoder_model.state_dict(),
        **meta_data
        }, os.path.join(path,SAVE_ENCODER_PATH))
    
    # save ctokenizer
    with open(os.path.join(path,SAVE_STATE_CTOKENIZER_PATH), 'wb') as handle:
        pkl.dump(ctokenizer, handle, protocol=pkl.HIGHEST_PROTOCOL)
    

    # Get rid of later....
    # save iter info for tensorboards if we continue training at a later date.
    with open(os.path.join(path,SAVE_STATE_META_PATH), 'wb') as handle:
        pkl.dump(meta_data, handle, protocol=pkl.HIGHEST_PROTOCOL)



def load_existing_model(encoder_model,path):
    PATH = os.path.join(path,SAVE_ENCODER_PATH) # Change xx
    checkpoint = torch.load(PATH)
    encoder_model.load_state_dict(checkpoint['model_state_dict'])

    # change to get from checkpoint instead of pickle
    # with open( os.path.join(path,SAVE_STATE_META_PATH), 'rb') as f:
    #     meta_data = pkl.load(f)

    with open(os.path.join(path,SAVE_STATE_CTOKENIZER_PATH), 'rb') as handle:
        ctokenizer = pkl.load(handle)        

    return encoder_model,ctokenizer,checkpoint['epoch'],checkpoint['count']



def load_w_encoder(path,model_args,device = 'auto'):
    if path is None:
        w_encoder = NewWordEmbeddingAligner(
            device,
            in_dim = model_args['in_dim'],
            out_dim = model_args['out_dim'],

            normalize = False,
            fudge_factor=0.03,
            n_layers=model_args['n_layers'],
            h_dim = model_args['h_dim'],
            use_feedback_layer = model_args['use_feedback_layer']
            )
    else:
        w_encoder = torch.load(os.path.join(path,"checkpoint"),map_location = device)
    return w_encoder


def create_roberta_encoder(model_id,tokenizer,w_encoder_path,peft_config,device = 'auto',w_encoder_args = None,verbose = False):
    #torch.Size([1, 13, 768]) - w_encoder has to go from this to 4028


    #### Load or Create w encoder
    # put all the configs into files....
    # should automate in/out dim sizes.
    if w_encoder_args is None:
        w_encoder_args = {
            'in_dim':768,
            'out_dim':4096,
            'h_dim':1024,
            'n_layers':2,
            'use_feedback_layer':True
        }
    if w_encoder_path is None:
        print('No pre-existing w_encoder. Creating new.')
        w_encoder = load_w_encoder(w_encoder_path,w_encoder_args,device = device)
    else:
        print('Loading existing w_encoder from: ',w_encoder_path)
        w_encoder = torch.load(os.path.join(w_encoder_path,"checkpoint"),map_location = device)


    
    # create encoder_model
    # test PEFTING this model? Nec?
    encoder_tokenizer = AutoTokenizer.from_pretrained(model_id)
    encoder_model = AutoModelForMaskedLM.from_pretrained(
                model_id,
                use_safetensors=True,
                #quantization_config=bnb_config,
                trust_remote_code=True,
                device_map=device,
    )    





    # add condenser tokens and add dummy weights to model.
    ctokenizer = CondenserTokenizer(encoder_tokenizer)
    encoder_model = ctokenizer.add_model(encoder_model)

    #encoder_model = prepare_model_for_kbit_training(encoder_model)
    #encoder_model = get_peft_model(encoder_model, peft_config)


    if verbose:
        for n,p in encoder_model.named_parameters():
            if p.requires_grad:
                print(n)

    param = encoder_model.parameters()
    encoder_model = TokenCondenserWrapper(encoder_model,w_encoder)    
    
    return encoder_model,encoder_tokenizer,ctokenizer,param



def create_encoder_objects(model,tokenizer,w_encoder_path,peft_config,device = 'auto',w_encoder_args = None,verbose = False):
    print(peft_config)



    if w_encoder_args is None:
        w_encoder_args = {
            'in_dim':4096,
            'out_dim':4096,
            'h_dim':1024,
            'n_layers':2,
            'use_feedback_layer':False
        }
    if w_encoder_path is None:
        print('No pre-existing w_encoder. Creating new.')
        w_encoder = load_w_encoder(w_encoder_path,w_encoder_args,device = device)
    else:
        print('Loading existing w_encoder from: ',w_encoder_path)
        w_encoder = torch.load(os.path.join(w_encoder_path,"checkpoint"),map_location = device)


    # add condenser tokens and add dummy weights to model.
    ctokenizer = CondenserTokenizer(tokenizer)
    #model = ctokenizer.add_model(model)


    # load w_encoder
    #w_encoder = torch.load(os.path.join(w_encoder_path,"checkpoint"),map_location = device)
    # create encoder_model
    # Load target model
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    encoder_model, etokenizer = create_model_and_tokenizer(model_id,device_map=None)
    encoder_model = ctokenizer.add_model(encoder_model)
    #encoder_model = prepare_model_for_kbit_training(encoder_model)
    encoder_model = get_peft_model(encoder_model, peft_config)
    # encoder_model = ctokenizer.add_model(encoder_model)

    if verbose:
        for n,p in encoder_model.named_parameters():
            if p.requires_grad:
                print(n)

    param = encoder_model.parameters()
    encoder_model = TokenCondenserWrapper(encoder_model,w_encoder)    
    encoder_model.base_model.config = encoder_model.base_model.generation_config
    return model,encoder_model,ctokenizer,param





def create_minstral_objects(model,tokenizer,w_encoder_path,peft_config,device = 'auto',w_encoder_args = None,verbose = False):

    #print('adding target modules to Minstral Peft Config.')
    #peft_config['target_modules']=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]



    if w_encoder_args is None:
        w_encoder_args = {
            'in_dim':4096,
            'out_dim':4096,
            'h_dim':1024,
            'n_layers':2,
            'use_feedback_layer':False
        }
    if w_encoder_path is None:
        print('No pre-existing w_encoder. Creating new.')
        w_encoder = load_w_encoder(w_encoder_path,w_encoder_args,device = device)
    else:
        print('Loading existing w_encoder from: ',w_encoder_path)
        w_encoder = torch.load(os.path.join(w_encoder_path,"checkpoint"),map_location = device)


    # add condenser tokens and add dummy weights to model.
    ctokenizer = CondenserTokenizer(tokenizer)
    #model = ctokenizer.add_model(model)


    # load w_encoder
    #w_encoder = torch.load(os.path.join(w_encoder_path,"checkpoint"),map_location = device)
    # create encoder_model
    # Load target model
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    encoder_model, etokenizer = create_model_and_tokenizer(model_id,device_map=None)
    encoder_model = ctokenizer.add_model(encoder_model)
    encoder_model = prepare_model_for_kbit_training(encoder_model)
    encoder_model = get_peft_model(encoder_model, peft_config)
    # encoder_model = ctokenizer.add_model(encoder_model)

    if verbose:
        for n,p in encoder_model.named_parameters():
            if p.requires_grad:
                print(n)

    param = encoder_model.parameters()
    encoder_model = TokenCondenserWrapper(encoder_model,w_encoder)    
    #encoder_model.base_model.config = encoder_model.base_model.generation_config
    encoder_model.base_model.config.pad_token_id = encoder_model.base_model.config.eos_token_id
    return model,encoder_model,ctokenizer,param,tokenizer
