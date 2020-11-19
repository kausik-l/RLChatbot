import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

import numpy as np

import . from utils

"""
Dimension of hidden state on expected i/p and RNN o/p.
Count of neurons in RNN.
"""
HIDDEN_STATE_SIZE = 512
"""
We won't be using pre-trained embeddings like GLoVE.
We want to train it alongside the model.
Embedding are vectors which represent tokens in our dictionary.
"""
EMBEDDING_DIM = 50

class PhraseModel(nn.module):
    def __init__(self,emb_size,dict_size,hid_size):
        super(PhraseModel,self).__init__()
        #Convert words to embeddings.
        self.emb = nn.Embedding(num_embeddings=dict_size,embedding_dim=emb_size)
        """
        One layer for encoder and another for decoder.
        If batch_first is True, the batch will be provided as the 1st dimension.
        input_size is the no. of expected features from input.
        hidden_size is no. of features in hidden state.
        """
        self.encoder = nn.LSTM(input_size=emb_size,hidden_size=hid_size,batch_first=True)
        self.decoder = nn.LSTM(input_size=emb_size,hidden_size=hid_size,batch_first=True)
        #Gives the probability distribution at the output.
        self.output = nn.Sequential(nn.Linear(hid_size,dict_size))

    def encode(self,x):
        """
        The output of that encoder has output,(hn,cn).
        output comprises all hidden states in the last layer.
        Tuple (hn,cn) is hidden and cell state of overall architecture.
        """
        _,hidden = self.encoder(x)
        return hidden

    def get_encoded_item(self,encoded,index):
        """
        In encode(), we are encoding whole batch of sequence.Here, we extract hidden state of
        nth(index) element of batch as decoding is performed for batches.The encoded i/p is
        a tuple of (hn,cn).
        By calling .contiguous(), we are setting the memory layout to be in such a way that
        even if we transpose it,the cells we skip in memory(check stride) to move to next position in cell
        will be constant (https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107).
        """
        return encoded[0][:, index:index+1].contiguous(), encoded[1][:, index:index+1].contiguous()

    def decode_teacher(self,hid,input_seq):
        """
        Teacher-forcing mode is being used in which input for every step is known.
        At each step, we feed a token from correct sentence and ask RNN to produce
        correct token next.

        """
        out , _ = self.decoder(input_seq, hid)
        #.data is used just to make sure we get underlying tensor from the variable.
        out = self.output(out.data)
        return out

    def decode_one(self,hid,input_x):
        """
        Performs one single decoding when there is a single input.
        Output is the raw scores(Logits) of every token instead of Prob. distribution
        and the new hidden state.

        """
        #unsqueeze(0) adds a dimension of size one inserted at specified position(0).
        out, new_hid = self.decoder(input_x.unsqueeze(0), hid)
        out = self.output(out)
        #squeeze(0) removes dimension of size 1 from input.
        return out.squeeze(dim=0) , new_hid

    def decode_chain_argmax(self,hid,begin_emb,seq_len,stop_token=None):
        """
        We feed the predicted token to the network again
        by acting greedily. Logits are unnormalized probabilities for every word
        in dictionary.
        Inputs : hid is hidden_state by encoder,begin_emb is #BEG in our case,
                 seq_len is max length of decoded sequence. If we don't stop the
                 decoding at #END, it will start repeating itself, apparently.
                 stop_token is the #END token in our case.
        Outputs :Tensor with resulting logits(used for training to calculate the loss)
                 and list of Token IDs(to pass to BLEU score calculation).
        """
        out_logits = []
        out_tokens = []
        curr_emb = begin_emb

        for i in range(seq_len):
            logits,hid = self.decode_one(hid,curr_emb)
            #torch.max()[1] has indices.nump()[0] as 2D array is returned([[x,y,..]])
            token = torch.max(logits,dim=1)[1].data.cpu().numpy()[0]#Max along rows.
            curr_emb = self.emb(torch.max(logits,dim=1)[1])

            out_logits.append(logits)
            out_tokens.append(token)

            if stop_token is not None and token == stop_token:
                break
        return torch.cat(out_logits),out_tokens

    def decode_chain_sampling(self,hid,begin_emb,seq_len,stop_token=None):
        """
        Here we act based on probabilities. Predicted token is fed to the network again.
        Inputs and Outputs are similar to the above function. The difference is we will
        be using Softmax here and it performs random sampling from returned probability
        distribution.
        """
        out_logits = []
        out_actions = []
        curr_emb = begin_emb

        for i in range(seq_len):
            logits,hid = self.decode_one(hid,curr_emb)
            token_probs = F.softmax(logits, dim=1).data.cpu().numpy()[0]
            """
            Argument p defines the probability associated with each entry.
            we reshape it to [0] as the input has to be a 1D array.
            """
            action = int(np.random.choice(token_probs.shape[0], p=token_probs))
            take_action = torch.LongTensor([action]).to(begin_emb.device)
            curr_emb = self.emb(take_action)

            out_logits.append(logits)
            out_actions.append(token)

            if stop_token is not None and action == stop_token:
                break
        return torch.cat(out_logits),out_actions

"""
Following functions can be used to process the input that will be given to the
model. It has to be in the form of PyTorch Tensor.

We are only padding the input phrases but not response phrases.
"""
"""

Inputs: batch is list of tuples(phrase,reply) (They are in the form of token IDs).
        embeddings(dictionary) is used to convert token ids to embeddings.
        (Unique token ids for each token will be converted into
        a vector of real numbers(embeddings) for sentences).

Outputs : packed sequence to be given to encoder,
          list of lists of integer ids of phrases and replies.
"""

def pack_batch_no_out(batch, embeddings, device="cpu"):
    assert isinstance(batch,list) #check if batch is a list.
    """
    Sorting them in descending(when reverse='True') order for CuDNN(CUDA backend).
    We also specify the sorting criteria with the key(lambda function specifying length of phrase).
    """
    batch.sort(key=lambda s: len(s[0]), reverse=True)
    #Now, [(p1,r1),(p2,r2),...] will become [(p1,p2,..),(r1,r2,...)]
    phr, rep = zip(*batch)
    """
    We create a matrix of dimension [batch,max_input_phrase] by padding zeros to make
    all the variable length sequences to constant length.
    """
    lens = list(map(len,phr)) #list of len of phrases
    #As phrases are in descending order,lens[0] will have highest phrase length.
    input_matrix = np.zeros((len(batch),lens[0]), dtype=np.int64)
    #enumerate of [a,b,c] -> [(0,a),(1,b),(2,c)]
    for idx,phrase in enumerate(phr):
        #For every row, upto len(phrase) column fill it with phrase(Remaining were already padded with zeros).
        input_matrix[idx, :len(phrase)] = phrase
    input_tensor = torch.tensor(input_matrix).to(device)
    """
    This is an in-built function.
    Inputs : The tensor with phrases, list of length of phrases,batch_first is true,
             batch will be provided as 1st dimension.
    Process : The phrases will be arranged columnwise.
    Reference : https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
                (line 118 of the code).
    Output : PackedSequence ready to be given to LSTM.
    """
    input_seq = rnn_utils.pack_padded_sequence(input_tensor,lens,batch_first=True)
    #Convert to word embeddings.
    r = embeddings(input_seq.data)
    #Convert the embeddings to PackedSequence.
    emb_input_seq = rnn_utils.PackedSequence(r,input_seq.batch_sizes)
    return emb_input_seq, phr,rep

def pack_input(input_data,embeddings,device="cpu"):
    """
    Used to convert encoded phrase (List of token ids) into packed sequence
    suitable to pass to RNN. In previous function, we had whole batches.
    """
    input_v = torch.LongTensor([input_data]).to(device)
    r = embeddings(input_v)
    return rnn_utils.pack_padded_sequence(r,[len(input_data)], batch_first=True)

def pack_batch(batch, embeddings, device="cpu"):
    """
    Similar to pack_batch_no_out but this function also converts output replies
    into list of packed sequences to be used in teacher-forcing mode of training.
    It also removes '#END' token from sentences.
    """
    emb_input_seq,phr,rep = pack_batch_no_out(batch,embeddings,device)
    emb_output_seq_list = []
    for each in rep:
        #Remove the '#END' token.
        emb_output_seq_list.append(pack_input(each[:-1],embeddings,device))
    return emb_input_seq,emb_output_seq_list,phr,rep

def seq_bleu(model_out,ref_seq):
    """
    We give it model output and reference sequence.
    It will take indices of tensor produced with logits
    from the decoder in teacher-forcing method.
    Takes max of those and uses BLEU func from utils.py
    by comparing it with reference sequence.
    """
    model_seq = torch.max(model_out.data, dim=1)[1]
    model_seq = model_seq.cpu().numpy()
    return utils.calc_bleu(model_seq,ref_seq)
