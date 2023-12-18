import os
import torch
from collections import OrderedDict




class TokenCondenserWrapper(torch.nn.Module):
    # try smaller hidden layer 128/256/1024
    # try 4/8 layers.
    def __init__(self, base_model, w_encoder):
        super(TokenCondenserWrapper, self).__init__()
        # when wrapped PEFT defs are messed up.
        self.base_parameters = base_model.parameters()
        self.w_parameters = w_encoder.parameters()

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
                    #use_cache=True,
        )
        x = outputs.hidden_states[-1][:,-1,:].to(device)
        x,x_fb = self.w_encoder(x.float(),get_feedback= True)
        x = x.half()
        x_fb = x_fb.half()
        return x,x_fb, outputs
    
    def process_training_embedding(self,x,att,device = 'cuda'):
        outputs = self.base_model(
                    inputs_embeds = x,#.to(torch.float16),
                    attention_mask = att,#.to(torch.int8()),
                    return_dict=True,
                    output_hidden_states = True,
                    #use_cache=True,
        )
        x = outputs.hidden_states[-1][:,:-1,:].to(device)
        x = self.w_encoder(x.float()).half()
        att = att[:,1:]
        #logits = outputs.loss['logits'][:,:-1] 
        logits = outputs.logits[:,:-1] 
        return x,att,logits

 



class WordEmbeddingAligner(torch.nn.Module):
    # try smaller hidden layer 128/256/1024
    # try 4/8 layers.
    def __init__(self, device,dim=4096,h_dim = 256,n_layers=2,normalize = True,fudge_factor = 1.0):
        super(WordEmbeddingAligner, self).__init__()
        if n_layers == 0:
            h_dim = dim
        
        self.in_layer = torch.nn.Linear(dim, h_dim).to(device)

        layers = []
        for i,l in enumerate(range(n_layers)):
            layers.append((f'linear_{i}',torch.nn.Linear(h_dim, h_dim)))
            layers.append((f'relu_{i}',torch.nn.ReLU()))

        layers = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layers).to(device)

        self.out_layer = torch.nn.Linear(h_dim, dim).to(device)
        self.normalize = normalize
        self.fudge_factor = fudge_factor
        #nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
       
    def forward(self, x):
        # print(self.in_layer.weight.dtype,x.dtype)
        x = torch.nn.functional.relu(self.in_layer(x))
        x = self.layers(x)
        x = self.out_layer(x)*self.fudge_factor
        # print('xf')
        # print('l',self.out_layer.weight[0,:10])
        # print(x[0,:10])
        if self.normalize:
            # print('norm',x.shape)
            x = torch.nn.functional.normalize(x)
        return x
    

class NewWordEmbeddingAligner(torch.nn.Module):
    '''FIX: Need 2 heads.
    1. for feeding back into encoder
    2. for producing the encodings into the base model.
    '''
    # try smaller hidden layer 128/256/1024
    # try 4/8 layers.
    def __init__(self, device,in_dim=4096,h_dim = 256,out_dim=4096,n_layers=2,normalize = True,use_feedback_layer = False,fudge_factor = 1.0):
        super(NewWordEmbeddingAligner, self).__init__()
        if n_layers == 0:
            h_dim = dim
        
        self.in_layer = torch.nn.Linear(in_dim, h_dim).to(device)

        layers = []
        for i,l in enumerate(range(n_layers)):
            layers.append((f'linear_{i}',torch.nn.Linear(h_dim, h_dim)))
            layers.append((f'relu_{i}',torch.nn.ReLU()))

        layers = OrderedDict(layers)
        self.layers = torch.nn.Sequential(layers).to(device)

        self.out_layer = torch.nn.Linear(h_dim, out_dim).to(device)
        self.use_feedback_layer = use_feedback_layer
        if use_feedback_layer:
            self.feedback_layer = torch.nn.Linear(h_dim, in_dim).to(device)
        # else:
        # else:
        #     self.feedback_layer = self.out_layer

        self.normalize = normalize
        self.fudge_factor = fudge_factor
        #nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
       
    def forward(self, x,get_feedback = False):
        # print(self.in_layer.weight.dtype,x.dtype)
        x = torch.nn.functional.relu(self.in_layer(x))
        x = self.layers(x)
        y = self.out_layer(x)*self.fudge_factor
        # print('xf')
        # print('l',self.out_layer.weight[0,:10])
        # print(x[0,:10])
        if self.normalize:
            # print('norm',x.shape)
            y = torch.nn.functional.normalize(y)

        if get_feedback:
            if self.use_feedback_layer:
                y_f = self.feedback_layer(x)
            else:
                y_f = y

            return y,y_f
        else:
            return y
            

 




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
 
