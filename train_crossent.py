"""
Cross-entropy method is one of the methods to train the model. During the training,
we randomly switch between teacher-forcing and argmax chain decoding(curriculum learning).
Deciding between these 2 will be random with a limiting probability of 50%.
(Faster convergence from teacher-forcing and stable decoding from curriculum learning).
"""
import os
import random
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.optim as optim
import torch.nn.functional as F

import high_level_cornell,model,utils

SAVES_DIR = "saves"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100

log = logging.getLogger("train")

TEACH_CURR_PROB = 0.5

def run_test(test_data, net, end_token, device="cpu"):
    """
    We calculate mean bleu score for every epoch for hold-out test dataset.
    Inputs: small test dataset, our LSTM network, end_token.
    """
    bleu_sum = 0.0
    bleu_count = 0

    for p1,p2 in test_data:
        #Encoded phrase to packed sequence.
        input_seq = model.pack_input(p1, net.emb, device)
        #Encode function is used to get hidden state.
        enc = net.encode(input_seq)
        #Passing hidden state, start_token, seq_len and stop_token to get list of Token IDs.
        _ , tokens = net.decode_chain_argmax(enc, input_seq.data[0:1],seq_len=high_level_cornell.MAX_TOKENS,stop_token=end_token)
        #Pass the above tokens along with reference tokens(replies without #BEGIN token).
        bleu_sum += utils.calc_bleu(tokens,p2[1:])
        bleu_count += 1
    return bleu_sum / bleu_count

#Execute the following only when invoked directly.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="%(asctime)-15s %(levelname)s %(message)s")
    #To pass the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Genre you want to train it on.Empty to train on full dataset.")
    #store_true action stores the argument as true.
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable CUDA.")
    #To save checkpoints and also to use in TensorBoard.
    parser.add_argument("-n","--name",required=True,help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    saves_path = os.path.join(SAVES_DIR, args.name)
    os.makedirs(saves_path, exist_ok=True)
    #Outputs (phrase,phrase) list and dictionary with unique IDs for each word.
    phrase_pairs, emb_dict = high_level_cornell.load_data(genre_filter=args.data)
    log.info("Obtained %d phrase pairs with %d unique words", len(phrase_pairs),len(emb_dict))
    #We save the embedding dictionary.
    high_level_cornell.save_emb_dict(saves_path,emb_dict)
    #Access the end token(#END)
    end_token = emb_dict[high_level_cornell.END_TOKEN]
    """
    (phrase,phrase) list becomes [(list of tokens IDs of p11,list of token IDs of p21),
                                  (list of tokens IDs of p12,list of token IDs of p22)]
    """
    train_data = high_level_cornell.encode_phrase_pairs(phrase_pairs, emb_dict)
    #Shuffle with seed as we have to shuffle it same way while using RL.
    rand = np.random.RandomState(high_level_cornell.SHUFFLE_SEED)
    rand.shuffle(train_data)
    log.info("Training data consisting of %d samples was shuffled", len(train_data))
    #Splitting into training and testing dataset.
    train_data,test_data = high_level_cornell.split_train_test(train_data)
    log.info("Training set has %d samples and test set has %d samples",len(train_data),len(test_data))
    #Our LSTM network.
    net = model.PhraseModel(emb_size=model.EMBEDDING_DIM, dict_size=len(emb_dict),
                            hid_size=model.HIDDEN_STATE_SIZE).to(device)
    log.info("Model : %s", net)

    writer = SummaryWriter(comment="-" + args.name)

    optimiser = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    best_bleu = None
    for epoch in range(MAX_EPOCHS):
        losses = []
        bleu_sum = 0.0
        bleu_count = 0
        for batch in high_level_cornell.iterate_batches(train_data,BATCH_SIZE):
            optimiser.zero_grad()
            #pack_batch returns packed input and output seq along with input and output token ids indices.
            input_seq, out_seq_list, input_idx, out_idx = model.pack_batch(batch, net.emb, device)
            #Returns hidden state from RNN.
            enc = net.encode(input_seq)

            net_results = []
            net_targets = []
            for idx,out_seq in enumerate(out_seq_list):
                ref_seqs = out_idx[idx][1:]
                #Gets the encoded input seq of that particular index.
                enc_item = net.get_encoded_item(enc,idx)
                """
                Randomize betwwen teacher-forcing and curriculum learning.
                Difference comes in calculation of bleu score.
                """
                if random.random() < TEACH_CURR_PROB:
                    #We give actual output token and ask it to produce output token(r).
                    r = net.decode_teacher(enc_item,out_seq)
                    bleu_sum += model.seq_bleu(r, ref_seqs)
                else:
                    #resulting logits and list of output token ids.
                    r,seq = net.decode_chain_argmax(enc_item,out_seq.data[0:1],len(ref_seqs))
                    bleu_sum += utils.calc_bleu(seq,ref_seqs)
                net_results.append(r)
                net_targets.extend(ref_seqs)
                bleu_count += 1
            #Concatenation of logits.
            results_v = torch.cat(net_results)
            #Tensor of reference token indices.
            targets_v = torch.LongTensor(net_targets).to(device)
            #Calculating cross entropy loss.
            loss_v = F.cross_entropy(results_v,targets_v)
            loss_v.backward()
            optimiser.step()
            losses.append(loss_v.item())
        #Mean bleu score.
        bleu = bleu_sum / bleu_count
        #We calculate bleu for hold-out set to assess output metrics.
        bleu_test = run_test(test_data,net,end_token,device)
        log.info("Epoch %d: mean loss %.3f, Mean BLEU %.3f, test BLEU %.3f ",
                  epoch,np.mean(losses),bleu,bleu_test)
        #Add to our SummaryWriter.
        writer.add_scalar("loss",np.mean(losses),epoch)
        writer.add_scalar("BLEU",bleu,epoch)
        writer.add_scalar("BLEU_Test",bleu_test,epoch)
        #Saving models scores with the best test BLEU seen so far for fine-tuning.
        if best_bleu is None or best_bleu < bleu_test:
            if best_bleu is not None:
                out_name = os.path.join(saves_path,"pre_bleu_%.3f_%02d.dat" % (bleu_test,epoch))
                #State_dict() maps each layer to its parameters.
                torch.save(net.state_dict(),out_name)
                log.info("Best BLEU updated %.3f",bleu_test)
            #If best_bleu is less than test_bleu.
            best_bleu = bleu_test
        #Save checkpoint for every 10 epochs.
        if epoch % 10 == 0:
            out_name = os.path.join(saves_path,"epoch_%03d_%.3f_%.3f.dat" % (epoch,bleu,bleu_test))
            torch.save(net.state_dict(), out_name)

    writer.close()
