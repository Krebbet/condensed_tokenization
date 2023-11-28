#accelerate launch --config_file ds_config.yaml dtc_train.py

print('xxxx')
import torch
from datasets import load_dataset


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


#local
from utils import (
    create_model_and_tokenizer, 
    maxpool_sentence_tokens,run_encoding_generation,
    training_step,
    run_encoding_generation,
    prepare_data,
    print_gpu_utilization,

)

    
# device = "cuda:0" if torch.cuda.is_available() else "cpu"


# Enoder Name
OUTPUT_DIR = 'dtc_encoder/test_00/'



def main():

    print_gpu_utilization()
    # Setup Huggingface Accelerate Object
    accelerator = Accelerator()
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = accelerator.device
    print('xxxxxxx')
    print(device)
    print('yyyyyyy')


    #MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    #MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
    model, tokenizer = create_model_and_tokenizer(model_id,device_map=None)

    peft_config = LoraConfig(
        r=8,  # dimension of the updated matrices
        lora_alpha=32,  # parameter for scaling
        target_modules= None,
        # [
        # "q_proj",
        # "up_proj",
        # "o_proj",
        # "k_proj",
        # "down_proj",
        # "gate_proj",
        # "v_proj"],
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode = False,
    )




    encoder_model = get_peft_model(model, peft_config)
    print('done')
    encoder_model.print_trainable_parameters()

    # look into what this function is doing... a little obscur.
    #encoder_model.enable_input_require_grads()
    # look into more... seems like it allows for the grads to be loaded into CPU RAM?
    #encoder_model.gradient_checkpointing_enable()



    # load dataset
    dataset = load_dataset("THUDM/webglm-qa")
    dataloader = DataLoader(dataset['train'], batch_size=1)

    # optimizer
    lr=1e-5
    opt = torch.optim.AdamW(encoder_model.parameters(), lr=lr)
    scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=100)


    # Prepare base model.
    #model = accelerator.prepare_model(model)


    # Accelerate Components with huggingface accelerate
    # The accelerate config defines what changes are actually made here
    encoder_model, opt, dataloader, scheduler = accelerator.prepare(
        encoder_model, opt, dataloader, scheduler
    )

    # Needed if loading a preexisting run
    # accelerator.load_state(None)
    encoder_model.train()
    model.train()

    # Counters and constants
    count = 0 #simple counter
    training_losses = []
    ignore_tokens = tokenizer([ # This defines all the tokens to ignore during loss calculations.
        tokenizer.eos_token+
        tokenizer.pad_token+
        tokenizer.unk_token
    ],return_tensors='pt', padding=True)['input_ids'].squeeze().to(device)


    ll = 0
    rl = 0 
    # this prevents type errors for fp16 training and fp32 code
    with torch.autocast(device_type='cuda'):
        for batch in tqdm(dataloader,desc = f"recon loss: {rl} length loss: {ll}"):
            opt.zero_grad()
            
            batch = prepare_data(batch['question'],tokenizer,device)
            loss,artifacts,max_x_length,y_tokens = training_step(
                batch,
                model,
                encoder_model,
                ignore_tokens,
                device,generation_max=300
                )
            
            # extract follow vars
            ll = artifacts['length_loss']
            rl = artifacts['recon_loss']
        
            accelerator.backward(loss)
            opt.step()
            scheduler.step()

            count += 1
            training_losses.append(float(loss))
            if count % 20 == 0:
                # Save checkpoints
                accelerator.save_state(OUTPUT_DIR+"/acc/checkpoint_interim")

        # # Save PEFT model/final accelerator state
        accelerator.save_state("final_checkpoint")
        encoder_model = accelerator.unwrap_model(encoder_model)
        encoder_model.save_pretrained(OUTPUT_DIR+'/model/encoder')
        print('done')



main()