import torch
from torch import nn
import torch.nn.functional as F


class Classification_RNN(nn.Module):
    '''
    nn.Linear(in_features, out_features): in_features is the number of inputs each neuron gets. out_features is the number of total neurons in that layer
    '''
    def __init__(self, input_size, hidden_size, output_size):
        '''
        input_size: Size of the vocab
        hidden_size: Randomly chosen size (256)
        output_size: No of classes (no of languages (18))
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_layer = nn.Linear(in_features = input_size + hidden_size , out_features = hidden_size)
        self.output_layer = nn.Linear(in_features = input_size + hidden_size, out_features = output_size)
        
    def forward(self, input, hidden_state):
        combined_input = torch.cat((input, hidden_state), 1)  # concat(59 + 256) -> 315-D Vector.
        hidden_state= torch.sigmoid(self.hidden_layer(combined_input)) #output -> hidden_size dim vector
        output = self.output_layer(combined_input) # output -> output_size
        return output, hidden_state
        
    def init_hidden(self):
        '''
        Returns a initialized hidden state for the first iteration
        '''
        return nn.init.kaiming_uniform_(torch.empty( (1, self.hidden_size) )) # output: (1,256)-D Vector
    
    
    
        
        


