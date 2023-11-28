
import torch
import torch.nn.functional as F






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



def length_penalty_by_eos(logits,att_mask,max_lengths,target_compression_rates,generation_config,k_res = 0.5):
    # print(logits.shape)
    # print(generation_config.eos_token_id)
    
    
    log_probs = F.log_softmax(logits, dim=2)
    x = log_probs[:,:,generation_config.eos_token_id]

    #print('LEN')
    # Mask out everything before Ideal length
    ideal_lengths = (max_lengths.unsqueeze(1).to(x.device)*target_compression_rates.to(x.device)).int() + 1
    ideal_lengths = ideal_lengths.squeeze(1).to(x.device)

    max_len = x.shape[1]
    len_mask = torch.arange(max_len,device = x.device).expand(len(ideal_lengths), max_len) < ideal_lengths.unsqueeze(1)
    len_eos_mask = torch.logical_not(len_mask).int()
    len_res_mask = len_mask.int()
    #print(len_mask)
    x = x*len_eos_mask.to(x.device)
    x_response = x*len_res_mask.to(x.device) # add a negative loss in for the response length to ensure everything doesnt go to zero.


    # Weight by distance from ideal loss (may not need... test)
    pass


    # Mask out irrelelavant loss contributions with attention mask
    if not att_mask is None: # This will eliminate loss caused by additional irrelevant tokens so yeah!
        x = x*att_mask.to(x.device)
        x_response = x_response*att_mask.to(x.device) # add a negative loss in for the response length to ensure everything doesnt go to zero.


    #print(x)
    # Compute the negative log likelihood loss
    #x = 1.0-x
    #loss = -torch.mean(x)
    loss = -torch.sum(x) + k_res*torch.sum(x_response)
    return loss

    bos_tensor = generation_config.eos_token_id*torch.ones([x.shape[0],1],dtype = torch.int)
    bos_emb = embedding_table(bos_tensor)
    bos_att = torch.ones([x.shape[0],1],dtype = torch.int)