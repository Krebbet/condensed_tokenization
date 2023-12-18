
import torch
import torch.nn.functional as F





# decreased from 0.05 to 0.01... maybe should do opposite
# try boosting first 5 positions maybe....
def position_weighted_cross_entropy(logits,labels,att_mask = None,k_pos = 0.05):
    """
    Compute the cross-entropy loss between predictions and targets.
    Args:
    - predictions: Tensor of shape (batch_size, num_classes) containing the model's predictions.
    - targets: Tensor of shape (batch_size) containing the true class labels.

    Returns:
    - loss: Scalar tensor representing the cross-entropy loss.
    """
    # Apply log_softmax to the predictions


    log_probs = F.log_softmax(logits, dim=2)

    # Gather the log probabilities corresponding to the target classes
    # This is equivalent to selecting the log probability of the correct class for each sample
    x = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)
    # weight by position
    pos = torch.arange(start=1, end=x.shape[1]+1,device = x.device).unsqueeze(0)
    x = x*torch.exp(-k_pos*pos)

    # Mask out irrelelavant loss contributions with attention mask
    if not att_mask is None:
        x = x*att_mask.to(x.device)

    # Compute the negative log likelihood loss
    loss = -torch.sum(x) * (k_pos/logits.size(0))  # Normalize by batch size
    return loss


def compute_loss(logits, labels,att_mask):
    # # Apply the mask to the loss
    # loss = F.cross_entropy(flat_logits[att_mask.view(-1)], flat_target[att_mask.view(-1)])
    loss = position_weighted_cross_entropy(logits,labels,att_mask)
    if not loss.requires_grad:
        loss.requires_grad = True

    return loss



def compute_length_penalty(x_emb_att,max_lengths,target_compression_rates,zero_at_pole = True,device = 'auto'):

    # Test range of this error
    response_length = x_emb_att.sum(dim=1,keepdims=False)
    norm = 1.0/(1.0 - target_compression_rates) 
    #print(response_length,max_lengths)
    #print(response_length.shape,max_lengths.shape)
    x = (response_length.to(device)/max_lengths.float().to(x_emb_att.device)) - target_compression_rates
    x = norm*x
    #denom =  max_lengths.to(device)
    #length_loss = (1.0-num/denom) - target_compression_rates
    if zero_at_pole:
        x = torch.clamp(x, min = 0.0)
    #length_loss = torch.maximum((x,torch.zeros_like(max_lengths).to(device))# FIX: Use clip
    length_loss = torch.sqrt(torch.mean(x**2)) #requires_grad=True. # Shoot for exactly that compression rate!
    if not length_loss.requires_grad:
        length_loss.requires_grad=True
    return length_loss    



def length_penalty_by_eos(logits,att_mask,max_lengths,target_compression_rates,loss_scale,
                          generation_config,k_eos = 0.1,k_res = 10.0,verbose = False,eps = 1e-5,len_eos_entry_weighting=5.0):
    # EOS prob on the response side has to be verrrry small!!!!!
    # if not it will still over power other tokens. 
    # Do some tests here.
    # print()
    # print(logits.shape)
    # print(att_mask.shape)
    # print(max_lengths.shape)
    # print(target_compression_rates.shape)
    # print(loss_scale.shape)



    # grab max length
    max_len = logits.shape[1]
    
    # Grab the EOS log probs, we want them to be large when above ideal length and small (negative) when we havent hit that point yet.
    x = F.softmax(logits, dim=2)
    # artificially create bin cross on EOS token to ensure EOS is not over predicted during response
    # we need a counter weight to our EOS forcing.
    eos_probs = x[:,:,generation_config.eos_token_id] # Gather all EOS logits.
    eos_probs = eos_probs*0.1
    x_eos = torch.log(eos_probs + eps)
    x_response = torch.log(1.0 - eos_probs + eps)
    
    # Calculate "Ideal Lengths" This should be a certain compression based on the special compression token used.
    ideal_lengths = (max_lengths.unsqueeze(1).to(x.device)*target_compression_rates.to(x.device)).int() + 1
    ideal_lengths = ideal_lengths.squeeze(1).to(x.device)

    # create binary masks for both the eos signal (everything AFTER ideal length) and response signal (everything BEFORE ideal length)
    len_mask = torch.arange(max_len,device = x.device).expand(len(ideal_lengths), max_len) < ideal_lengths.unsqueeze(1)
    len_eos_mask = torch.logical_not(len_mask).int()
    len_res_mask = len_mask.int()

  

    # Mask out irrelelavant loss contributions with attention mask
    if not (att_mask is None): # This will eliminate loss caused by additional irrelevant tokens so yeah!
        len_eos_mask = len_eos_mask*att_mask.to(x.device)
        
        # Is this masking out the bad EOS errors? 
        #len_res_mask = len_res_mask*att_mask.to(x.device) # add a negative loss in for the response length to ensure everything doesnt go to zero.

        # weight eos mask by number of EOS prediciton failures.
        #len_eos_mask = len_eos_mask*(1+ torch.sum(len_eos_mask,dim =1).unsqueeze(1)/len_eos_entry_weighting)
        len_eos_mask = len_eos_mask

    # Apply masks
    x_eos = x_eos*len_eos_mask.to(x.device)
    eos_len_scale = 1.0 + torch.sqrt(torch.sum(len_eos_mask,dim =1).unsqueeze(1))/len_eos_entry_weighting
    eos_n = torch.sum(len_eos_mask,dim =1) +1 
    x_response = x_response*len_res_mask.to(x.device) # add a negative loss in for the response length to ensure everything doesnt go to zero.

    # Weight by distance from ideal loss (may not need... test)
    pass


    # EXPERIMENTAL: Apply SCALING for Continous Loss here. <C_CONT>
    # We are trying to weight the EOS loss LESS to see if we can let 
    # the model choose the embedding length dynamically.
    x_eos = loss_scale*eos_len_scale*x_eos
    x_response = loss_scale*x_response





    # sum loss terms
    if verbose:
        #print(att_mask)
        print('eos_loss terms')
        print(x_eos[0])
        print('response_loss terms')
        print(x_response[0])
        print('rand log prob entry')

        print('eos mask')
        print(len_eos_mask[0])
        print('response mask')
        print(len_res_mask[0])
        print('ideal lengths')
        print(ideal_lengths)


        print()
        print(torch.sum(x_eos))
        print(torch.sum(x_response,dim = 0))
        print(torch.mean(x_response))

    # k_res hyper parameter, Probably okay at 1.
    #response_loss = -k_res*torch.mean(x_response) # Mean the loss - All preds equil
    response_loss = -k_res*torch.sum(x_response) # try sum... EOS only has to come up once so makes sense.
    # We want some of the length of the EOS tokens to come through
    # SO we SUM the weight, norm by batch and then insert human determine scale factor.
    eos_loss = -k_eos*torch.mean((torch.sum(x_eos,dim=1)/eos_n))
    # print(torch.sum(x_eos,dim=1))
    # print(eos_n)
    # print((torch.sum(x_eos,dim=1)/eos_n))
    # print('\nresponse | eos')
    # print('\n',response_loss.cpu().detach().numpy(),eos_loss.cpu().detach().numpy())
    # print('MAX LENS:',max_lengths)

    loss = eos_loss + response_loss
    return loss

    bos_tensor = generation_config.eos_token_id*torch.ones([x.shape[0],1],dtype = torch.int)
    bos_emb = embedding_table(bos_tensor)
    bos_att = torch.ones([x.shape[0],1],dtype = torch.int)