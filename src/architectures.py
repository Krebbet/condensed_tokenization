import os
import torch



class TokenCondenserWrapper(torch.nn.Module):
    # try smaller hidden layer 128/256/1024
    # try 4/8 layers.
    def __init__(self, base_model, w_encoder):
        super(TokenCondenserWrapper, self).__init__()
        self.base_model = base_model
        self.w_encoder = w_encoder   


    def save(self,path,base_name = 'saved_model',w_encoder_name = 'aligner_checkpoint.cpt'):
        self.base_model.save_pretrained(os.path.join(path,base_name))
        torch.save(self.w_encoder,os.path.join(path,base_name,w_encoder_name))


    def get_parameters():
        pass



    #TODO - this only works when inputting embeddings. If we want 
    # to use this more generally I will have to do a better job of 
    # writing this wrapper.
    def forward(self, x,att,device = 'cuda'):
        outputs = self.base_model(
                    inputs_embeds = x,#.to(torch.float16),
                    attention_mask = att,#.to(torch.int8()),
                    return_dict=True,
                    output_hidden_states = True,
                    output_attentions = False,
                    use_cache=True,
        )
        x = outputs.hidden_states[-1][:,-1,:].to(device)
        x = self.w_encoder(x.float()).half()
        return x, outputs
    
    def process_training_embedding(self,x,att,device = 'cuda'):
        outputs = self.base_model(
                    inputs_embeds = x,#.to(torch.float16),
                    attention_mask = att,#.to(torch.int8()),
                    return_dict=True,
                    output_hidden_states = True,
                    use_cache=True,
        )
        x = outputs.hidden_states[-1][:,:-1,:].to(device)
        x = self.w_encoder(x.float()).half()
        att = att[:,1:]
        logits = outputs.loss['logits'][:,:-1] 
        return x,att,logits

 



class WordEmbeddingAligner(torch.nn.Module):
    # try smaller hidden layer 128/256/1024
    # try 4/8 layers.
    def __init__(self, device,dim=4096,h_dim = 256,n_layers=2,normalize = True,fudge_factor = 1.0):
        super(WordEmbeddingAligner, self).__init__()
        if n_layers == 0:
            h_dim = dim
        
        self.in_layer = torch.nn.Linear(dim, h_dim).to(device)
        self.layers = [torch.nn.Linear(h_dim, h_dim).to(device) for n in range(n_layers -1)]
        #self.layer1 = torch.nn.Linear(dim, dim).to(device)
        self.out_layer = torch.nn.Linear(h_dim, dim).to(device)
        self.normalize = normalize
        self.fudge_factor = fudge_factor
        #nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
       
    def forward(self, x):
        # print(self.in_layer.weight.dtype,x.dtype)
        x = torch.nn.functional.relu(self.in_layer(x))
        for layer in self.layers:
            #x = torch.nn.functional.relu(layer(x))
            # print(layer.weight.dtype,x.dtype)
            x = layer(x)
            # print('xxxxx')
            # print('l',layer.weight[0,:10])
            # print('b',layer.bias[:10])
            # print(x[0,:10])
        #     print(x.shape,layer)
        x = self.out_layer(x)*self.fudge_factor
        # print('xf')
        # print('l',self.out_layer.weight[0,:10])
        # print(x[0,:10])
        if self.normalize:
            # print('norm',x.shape)
            x = torch.nn.functional.normalize(x)
        return x
 




### Deps, get rid of once I train a new encoder.
class NeuralNetwork(torch.nn.Module):
    # try smaller hidden layer 128/256/1024
    # try 4/8 layers.
    def __init__(self, device,dim=4096,h_dim = 256,n_layers=2,normalize = True,fudge_factor = 1.0):
        super(WordEmbeddingAligner, self).__init__()
        if n_layers == 0:
            h_dim = dim
        
        self.in_layer = torch.nn.Linear(dim, h_dim).to(device)
        self.layers = [torch.nn.Linear(h_dim, h_dim).to(device) for n in range(n_layers -1)]
        #self.layer1 = torch.nn.Linear(dim, dim).to(device)
        self.out_layer = torch.nn.Linear(h_dim, dim).to(device)
        self.normalize = normalize
        self.fudge_factor = fudge_factor
        #nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
       
    def forward(self, x):
        # print(self.in_layer.weight.dtype,x.dtype)
        x = torch.nn.functional.relu(self.in_layer(x))
        for layer in self.layers:
            #x = torch.nn.functional.relu(layer(x))
            # print(layer.weight.dtype,x.dtype)
            x = layer(x)
            # print('xxxxx')
            # print('l',layer.weight[0,:10])
            # print('b',layer.bias[:10])
            # print(x[0,:10])
        #     print(x.shape,layer)
        x = self.out_layer(x)*self.fudge_factor
        # print('xf')
        # print('l',self.out_layer.weight[0,:10])
        # print(x[0,:10])
        if self.normalize:
            # print('norm',x.shape)
            x = torch.nn.functional.normalize(x)
        return x
 
