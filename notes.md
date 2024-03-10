# Project: To predict the nationality of a person give the name of the person. 

* Total nationalities (classes): 18
* Total Size of vocab: 59 (This includes all the alphabets(lower and uppercase) of the english language + few symbols.)
* Hidden state size: 256 (The size of the randomly initialized tensor which carries the memory)


## RNN architecture:
##### Refer to the 'Classification_RNN.jpg' picture. 

* The input xt is a single character (one-hot tensor of 'vocab' size (59). There is the digit 1 at the index where xt is in the vocab). 

* The hidden state (256-dim tensor) basically is the memory which is carried to the next iteration and concatenated(at axis =1) with the next input character.

* One word is represented by a (no of characters, vocab size) tensor. 

* In the architecture, The input tensor is concatenated with the hidden state. This new combined input goes through two different layers. The output of the first layer(hidden layer)
is calculated by applying the sigmoid and is called the hidden state. This hidden state goes on to the next time-step and is concatenated with the next input character. 
The output of the 2nd layer is outputted normally. this output is overwritten and is only collected at the end of the WORD's processing (because the final layer will contain the optimized label).

* The loss is calculated at the end of every WORD's processing.

**Remember**: In nn.Linear(in_features, out_features), in_features is how many inputs will a single neuron take. out_features is how many such neurons in that layer.

## Kaiming Weight Initialization
* In the function init_hidden(), which generates the initial hidden state vector for every word, kaiming initialization is used to initiate the initial
weights. More reading on kaiming: <br>
Reference: https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

## Loss Calculation between a 18-dim tensor and a 1-dim tensor ('output' and 'label' respectively)
* In the rnn.ipynb, training loop -> the loss is calculated between 'output' variable and 'label' variable. 
* The output variable is a 18-dim tensor (no of classes) and the 'label' variable is a 1-dim tensor (the index (integer) of the true class. The way cross entropy calculates the loss internally is by initializing a one-hot vector for the 'label' variable and storing the value of 1 at the index of the true class. 
* Now, both 'output' and 'label' vector have same length. Now, the cross entropy loss is calculated by same way, which
will end up with the true label at the specified index being multiplied with the predicted probability in the 'output' vector at the same specified index.