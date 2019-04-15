import os, sys
sys.path.append('/home/liuxg/workspace/SAparaphrase/')
sys.path.append('/home/liuxg/workspace/SAparaphrase/bert')
sys.path.append('/home/liuxg/workspace/SAparaphrase/bert/pytorch_pretrained_bert')
import argparse
import time,random
import torch
import torch.nn as nn
import numpy as np
import data
from experiment import Experiment
from models import  *
from utils import get_corpus_bleu_scores, savetexts
from simulateAnnealing import  metropolisHasting, simulatedAnnealing

class Option(object):
    def __init__(self, d):
        self.__dict__ = d
    def save(self):
        with open(os.path.join(self.this_expsdir, "option.txt"), "w") as f:
            for key, value in sorted(self.__dict__.items(), key=lambda x: x[0]):
                f.write("%s, %s\n" % (key, str(value)))

def main():
    parser = argparse.ArgumentParser(description="Experiment setup")
    # misc
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--gpu', default="", type=str)
    parser.add_argument('--no_train', default=False, action="store_true")
    parser.add_argument('--from_model_ckpt', default=None, type=str)
    parser.add_argument('--no_rules', default=False, action="store_true")
    parser.add_argument('--rule_thr', default=1e-2, type=float)    
    parser.add_argument('--no_preds', default=False, action="store_true")
    parser.add_argument('--get_vocab_embed', default=False, action="store_true")
    parser.add_argument('--exps_dir', default=None, type=str)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--load', default=None, type=str)
    # data property
    parser.add_argument('--data_path', default='data/quora/quora.txt', type=str)
    parser.add_argument('--dict_path', default='data/quora/dict.pkl', type=str)
    parser.add_argument('--dict_size', default=30000, type=int)
    parser.add_argument('--vocab_size', default=30003, type=int)
    parser.add_argument('--backward', default=False, action="store_true")
    parser.add_argument('--keyword_pos', default=True, action="store_false")
    parser.add_argument('--type_check', default=False, action="store_true")
    parser.add_argument('--domain_size', default=128, type=int)
    parser.add_argument('--no_extra_facts', default=False, action="store_true")
    parser.add_argument('--query_is_language', default=False, action="store_true")
    parser.add_argument('--vocab_embed_size', default=128, type=int)
    # model architecture
    parser.add_argument('--num_steps', default=15, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--emb_size', default=256, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--model', default=0, type=int)
    # optimization
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--print_per_batch', default=3, type=int)
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--min_epoch', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--clip_norm', default=0.00, type=float)
    parser.add_argument('--no_norm', default=False, action="store_true")
    parser.add_argument('--no_cuda', default=False, action="store_true")
    parser.add_argument('--local', default=False, action="store_true")
    parser.add_argument('--thr', default=1e-20, type=float)

    # evaluation
    parser.add_argument('--sim', default='word_max', type=str)
    parser.add_argument('--mode', default='sa', type=str)
    parser.add_argument('--get_attentions', default=False, action="store_true")
    parser.add_argument('--adv_rank', default=False, action="store_true")
    parser.add_argument('--rand_break', default=False, action="store_true")
    parser.add_argument('--accuracy', default=False, action="store_true")
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--accumulate_step', default=1, type=int)
    parser.add_argument('--backward_path', default=None, type=str)
    parser.add_argument('--forward_path', default=None, type=str)

    # sampling
    parser.add_argument('--use_data_path', default='data/input/input.txt', type=str)
    parser.add_argument('--reference_path', default=None, type=str)
    parser.add_argument('--pos_path', default='POS/english-models', type=str)
    parser.add_argument('--emb_path', default='data/quora/emb.pkl', type=str)
    parser.add_argument('--max_key', default=3, type=float)
    parser.add_argument('--max_key_rate', default=0.5, type=float)
    parser.add_argument('--rare_since', default=300000, type=int)
    parser.add_argument('--sample_time', default=100, type=int)
    parser.add_argument('--search_size', default=100, type=int)
    parser.add_argument('--action_prob', default=[0.4,0.3,0.3], type=list)
    parser.add_argument('--threshold', default=0.1, type=float)
    parser.add_argument('--just_acc_rate', default=0.0, type=float)
    parser.add_argument('--sim_mode', default='keyword', type=str)
    parser.add_argument('--save_path', default='temp.txt', type=str)
    
    d = vars(parser.parse_args())
    option = Option(d)

    random.seed(option.seed)
    np.random.seed(option.seed)
    torch.manual_seed(option.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu


    if option.exp_name is None:
      option.tag = time.strftime("%y-%m-%d-%H-%M")
    else:
      option.tag = option.exp_name  
    if option.accuracy:
      assert option.top_k == 1
    

    dataclass = data.Data(option)       
    print("Data prepared.")

    option.this_expsdir = os.path.join(option.exps_dir, option.tag)
    if not os.path.exists(option.this_expsdir):
        os.makedirs(option.this_expsdir)
    option.ckpt_dir = os.path.join(option.this_expsdir, "ckpt")
    if not os.path.exists(option.ckpt_dir):
        os.makedirs(option.ckpt_dir)
    option.model_path = os.path.join(option.ckpt_dir, "model")
    
    option.save()
    print("Option saved.")

    device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    if option.model == 0:
        learner = RNNModel(option)

    learner.to(device)
    # learner = nn.DataParallel(learner, [0,1,2])
    if option.load is  not None: 
        with open(option.load, 'rb') as f:
            learner.load_state_dict(torch.load(f))

    experiment = Experiment(option, learner=learner, data=dataclass)
    print("Experiment created.")

    if not option.no_train:
        print("Start training...")
        experiment.train()
    
    if option.mode == 'sa':
        forwardmodel = RNNModel(option).cuda()
        backwardmodel = RNNModel(option).cuda()
        if option.forward_path is  not None: 
            with open(option.forward_path, 'rb') as f:
                forwardmodel.load_state_dict(torch.load(f))

        if option.backward_path is  not None: 
            with open(option.backward_path, 'rb') as f:
                backwardmodel.load_state_dict(torch.load(f))
        forwardmodel.eval()
        backwardmodel.eval()
        generated_word_lists = simulatedAnnealing(option, dataclass, forwardmodel, backwardmodel,\
                sim_mode = option.sim_mode)
        savetexts(generated_word_lists,option.save_path)
        # Evaluate model scores
        if option.reference_path is not None:
            actual_word_lists = []
            with open(option.reference_path) as f:
                for line in f:
                    actual_word_lists.append([line.strip().split()])

            bleu_scores = get_corpus_bleu_scores(actual_word_lists, generated_word_lists)
            print('bleu scores:', bleu_scores)


    elif option.mode == 'mh':
        forwardmodel = RNNModel(option).cuda()
        backwardmodel = RNNModel(option).cuda()
        if option.forward_path is  not None: 
            with open(option.forward_path, 'rb') as f:
                forwardmodel.load_state_dict(torch.load(f))

        if option.backward_path is  not None: 
            with open(option.backward_path, 'rb') as f:
                backwardmodel.load_state_dict(torch.load(f))
        forwardmodel.eval()
        backwardmodel.eval()
        metropolisHasting(option, dataclass, forwardmodel, backwardmodel)

    print("="*36 + "Finish" + "="*36)

if __name__ == "__main__":
    main()

