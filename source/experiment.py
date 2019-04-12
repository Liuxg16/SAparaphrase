import sys
import os
import time
import pickle
from collections import Counter
import numpy as np
import torch
import torch
import torch.nn as nn

class Experiment():
    """
    This class handles all experiments related activties, 
    including training, testing, early stop, and visualize
    results, such as get attentions and get rules. 

    Args:
        sess: a TensorFlow session 
        saver: a TensorFlow saver
        option: an Option object that contains hyper parameters
        learner: an inductive learner that can  
                 update its parameters and perform inference.
        data: a Data object that can be used to obtain 
              num_batch_train/valid/test,
              next_train/valid/test,
              and a parser for get rules.
    """
    
    def __init__(self, option, learner=None, data=None):
        self.option = option
        self.learner = learner
        self.data = data
        # helpers
        self.msg_with_time = lambda msg: \
                "%s Time elapsed %0.2f hrs (%0.1f mins)" \
                % (msg, (time.time() - self.start) / 3600., 
                        (time.time() - self.start) / 60.)

        self.start = time.time()
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.best_valid_in_top = 0.
        self.train_stats = []
        self.valid_stats = []
        self.test_stats = []
        self.early_stopped = False
        # self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "w")

        self.write_log_file("-----------has built a model---------\n")

        param_optimizer = list(self.learner.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        # t_total = -1 # self.data.num_batch_train
        # self.optimizer = BertAdam(optimizer_grouped_parameters,
        #                  lr= self.option.learning_rate,
        #                  warmup= 0.01,
        #                  t_total=t_total)
        self.optimizer = torch.optim.Adam(self.learner.parameters(), self.option.learning_rate,\
                weight_decay = self.option.weight_decay)

        self.device = torch.device("cuda" if torch.cuda.is_available() and not option.no_cuda else "cpu")

    def one_epoch(self, mode, num_batch, next_fn):
        epoch_loss = [0]
        epoch_in_top = [0]
        self.optimizer.zero_grad()
        for batch in xrange(num_batch):
            if (batch+1) % max(1, (num_batch / self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, num_batch))
                sys.stdout.flush()
            
            data, lengths, target= next_fn() # query(relation), head(target), tails
            data = data.to(self.device)
            target = target.to(self.device)

            if mode == "train":
                loss, acc, outputs = self.learner(data, target)

                loss.backward()
                if self.option.clip_norm>0: 
                    torch.nn.utils.clip_grad_norm(self.learner.parameters(),self.option.clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                loss, acc, outputs = self.learner(data, target)
            epoch_loss += [loss.item()]
            epoch_in_top += [acc.item()]
            # msg = 'intop:{}'.format(np.mean(epoch_loss))
            # print msg
            # self.write_log_file(msg)


        msg = self.msg_with_time(
                "Epoch %d mode %s Loss %0.4f In top %0.4f." 
                % (self.epoch+1, mode, np.mean(epoch_loss), np.mean(epoch_in_top)))
        print(msg)
        self.write_log_file(msg)
        # self.log_file.write(msg + "\n")
        return epoch_loss, epoch_in_top

    def one_epoch_train(self):
        self.learner.train()
        loss, in_top = self.one_epoch("train", 
                                      self.data.train_data.length/self.option.batch_size, 
                                      self.data.next_train)

        
        self.train_stats.append([loss, in_top])
        
    def one_epoch_valid(self):
        self.learner.eval()
        loss, in_top = self.one_epoch("valid", 
                                      self.data.valid_data.length/self.option.batch_size, 
                                      self.data.next_valid)
        self.valid_stats.append([loss, in_top])
        self.best_valid_loss = min(self.best_valid_loss, np.mean(loss))
        self.best_valid_in_top = max(self.best_valid_in_top, np.mean(in_top))

    def one_epoch_test(self):
        self.learner.eval()
        loss, in_top = self.one_epoch("test", 
                                      self.data.test_data.length/self.option.batch_size,
                                      self.data.next_test)
        self.test_stats.append([loss, in_top])
    
    def early_stop(self):
        loss_improve = self.best_valid_loss == np.mean(self.valid_stats[-1][0])
        in_top_improve = self.best_valid_in_top == np.mean(self.valid_stats[-1][1])


        # if in_top_improve:
        if loss_improve:
            print '----------------'
            with open(self.option.model_path+'-best.pkl', 'wb') as f:
                torch.save(self.learner.state_dict(), f)
            return False

        else:
            if self.epoch > self.option.min_epoch:
                return True
            else:
                return False

    def train(self):
        while (self.epoch < self.option.max_epoch and not self.early_stopped):
            self.one_epoch_train()
            self.one_epoch_valid()
            self.one_epoch_test()
            self.epoch += 1
            # model_path = self.saver.save(self.sess, 
            #                              self.option.model_path,
            #                              global_step=self.epoch)
            # print("Model saved at %s" % model_path)
            # 

            if self.early_stop():
                self.early_stopped = True
                print("Early stopped at epoch %d" % (self.epoch))
        
        all_test_in_top = [np.mean(x[1]) for x in self.test_stats]
        best_test_epoch = np.argmax(all_test_in_top)
        best_test = all_test_in_top[best_test_epoch]
        
        msg = "Best test in top: %0.4f at epoch %d." % (best_test, best_test_epoch + 1)       
        print(msg)
        self.write_log_file(msg + "\n")
        pickle.dump([self.train_stats, self.valid_stats, self.test_stats],
                    open(os.path.join(self.option.this_expsdir, "results.pckl"), "w"))

    def get_predictions(self):
        self.learner.eval()
        if self.option.query_is_language:
            all_accu = []
            all_num_preds = []
            all_num_preds_no_mistake = []

        f = open(os.path.join(self.option.this_expsdir, "test_predictions.txt"), "w")
        if self.option.get_phead:
            f_p = open(os.path.join(self.option.this_expsdir, "test_preds_and_probs.txt"), "w")
        all_in_top = []
        for batch in xrange(self.data.num_batch_test):
            if (batch+1) % max(1, (self.data.num_batch_test / self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, self.data.num_batch_test))
                sys.stdout.flush()
            (qq, hh, tt), mdb, bacov, maxrow = self.data.next_test()
            in_top, predictions_this_dd \
                    = self.learner.get_predictions_given_queries(qq, hh, tt, mdb)
            localp = predictions_this_dd.reshape(2,-1)
            predictions_this_batch = np.zeros((2,maxrow))
            for i in range(2):
                for j in range(localp.shape[1]):
                    predictions_this_batch[i,bacov[j]] = localp[i,j]
            assert len(qq)==2
            all_in_top += list(in_top)

            hh = [bacov[x] for x in hh]
            tt = [bacov[x] for x in tt]

            for i, (q, h, t) in enumerate(zip(qq, hh, tt)):
                p_head = predictions_this_batch[i, h]
                if self.option.adv_rank:
                    eval_fn = lambda (j, p): p >= p_head and (j != h)
                elif self.option.rand_break:
                    eval_fn = lambda (j, p): (p > p_head) or ((p == p_head) and (j != h) and (np.random.uniform() < 0.5))
                else:
                    eval_fn = lambda (j, p): (p > p_head)
                this_predictions = filter(eval_fn, enumerate(predictions_this_batch[i, :]))
                this_predictions = sorted(this_predictions, key=lambda x: x[1], reverse=True)
                if self.option.query_is_language:
                    all_num_preds.append(len(this_predictions))
                    mistake = False
                    for k, _ in this_predictions:
                        assert(k != h)
                        if not self.data.is_true(q, k, t):
                            mistake = True
                            break
                    all_accu.append(not mistake)
                    if not mistake:
                        all_num_preds_no_mistake.append(len(this_predictions))
                else:
                    this_predictions.append((h, p_head))
                    this_predictions = [self.data.number_to_entity[j] for j, _ in this_predictions]
                    q_string = self.data.parser["query"][q]
                    h_string = self.data.number_to_entity[h]
                    t_string = self.data.number_to_entity[t]
                    to_write = [q_string, h_string, t_string] + this_predictions
                    # num += len(to_write)
                    f.write(",".join(to_write) + "\n")
                    if self.option.get_phead:
                        f_p.write(",".join(to_write + [str(p_head)]) + "\n")
        f.close()
        if self.option.get_phead:
            f_p.close()
        
        if self.option.query_is_language:
            print("Averaged num of preds", np.mean(all_num_preds))
            print("Averaged num of preds for no mistake", np.mean(all_num_preds_no_mistake))
            msg = "Accuracy %0.4f" % np.mean(all_accu)
            print(msg)
            self.log_file.write(msg + "\n")

        msg = "Test in top %0.4f" % np.mean(all_in_top)
        msg += self.msg_with_time("\nTest predictions written.")
        print(msg)
        self.write_log_file(msg + "\n")
        
    def get_performance(self):
        self.learner.eval()
        if self.option.query_is_language:
            all_accu = []
            all_num_preds = []
            all_num_preds_no_mistake = []

        f = open(os.path.join(self.option.this_expsdir, "test_predictions.txt"), "w")
        if self.option.get_phead:
            f_p = open(os.path.join(self.option.this_expsdir, "test_preds_and_probs.txt"), "w")
        all_in_top = []
        lasttimes = [] 
        for batch in xrange(self.data.num_batch_test):
            if (batch+1) % max(1, (self.data.num_batch_test / self.option.print_per_batch)) == 0:
                sys.stdout.write("%d/%d\t" % (batch+1, self.data.num_batch_test))
                sys.stdout.flush()
            (qq, hh, tt), mdb, bacov, maxrow = self.data.next_test()
            in_top, predictions_this_dd, lasttime \
                    = self.learner.get_predictions(qq, hh, tt, mdb)
            lasttimes+=lasttime
            print np.mean(lasttimes)
            localp = predictions_this_dd.reshape(2,-1)
            predictions_this_batch = np.zeros((2,maxrow))
            for i in range(2):
                for j in range(localp.shape[1]):
                    predictions_this_batch[i,bacov[j]] = localp[i,j]
            assert len(qq)==2
            all_in_top += list(in_top)

            hh = [bacov[x] for x in hh]
            tt = [bacov[x] for x in tt]

            for i, (q, h, t) in enumerate(zip(qq, hh, tt)):
                p_head = predictions_this_batch[i, h]
                if self.option.adv_rank:
                    eval_fn = lambda (j, p): p >= p_head and (j != h)
                elif self.option.rand_break:
                    eval_fn = lambda (j, p): (p > p_head) or ((p == p_head) and (j != h) and (np.random.uniform() < 0.5))
                else:
                    eval_fn = lambda (j, p): (p > p_head)
                this_predictions = filter(eval_fn, enumerate(predictions_this_batch[i, :]))
                this_predictions = sorted(this_predictions, key=lambda x: x[1], reverse=True)
                if self.option.query_is_language:
                    all_num_preds.append(len(this_predictions))
                    mistake = False
                    for k, _ in this_predictions:
                        assert(k != h)
                        if not self.data.is_true(q, k, t):
                            mistake = True
                            break
                    all_accu.append(not mistake)
                    if not mistake:
                        all_num_preds_no_mistake.append(len(this_predictions))
                else:
                    this_predictions.append((h, p_head))
                    this_predictions = [self.data.number_to_entity[j] for j, _ in this_predictions]
                    q_string = self.data.parser["query"][q]
                    h_string = self.data.number_to_entity[h]
                    t_string = self.data.number_to_entity[t]
                    to_write = [q_string, h_string, t_string] + this_predictions
                    # num += len(to_write)
                    f.write(",".join(to_write) + "\n")
                    if self.option.get_phead:
                        f_p.write(",".join(to_write + [str(p_head)]) + "\n")
        f.close()
        if self.option.get_phead:
            f_p.close()
        
        if self.option.query_is_language:
            print("Averaged num of preds", np.mean(all_num_preds))
            print("Averaged num of preds for no mistake", np.mean(all_num_preds_no_mistake))
            msg = "Accuracy %0.4f" % np.mean(all_accu)
            print(msg)
            self.log_file.write(msg + "\n")

        msg = "Test in top %0.4f" % np.mean(all_in_top)
        msg += self.msg_with_time("\nTest predictions written.")
        print(msg)
        self.write_log_file(msg + "\n")

    def get_attentions(self, threshold = 0.001):
        number2relation = {v:k for k,v in self.data.relation_to_number.items()}

        def getrel(number2relation,id):
            i = id/2
            j = id%2
            relation = number2relation[i]
            if j==1:
                relation = 'inv_'+relation
            return relation


        for name, param in self.learner.named_parameters():
            if name=='edge_attentions':
                alpha = nn.Softmax(1)(param)
                values,index_topk = torch.topk(alpha,3, sorted=True)
                print 'relations:{}, matrix combinations:{}, attentionsize:{}'.format(\
                        self.data.num_operator,self.data.num_operator**2 ,index_topk.size())
                values = values.cpu().data.tolist()
                index_topk = index_topk.cpu().data.tolist()
                for id, (vs, ins) in enumerate(zip(values, index_topk)):
                    relation = getrel(number2relation,id)
                    print 'relation,',relation
                    for v,i in zip(vs,ins):
                        if v>threshold:
                            print (getrel(number2relation,i%self.data.num_operator),\
                                    getrel(number2relation,i/self.data.num_operator)),

                    print

    def get_rules(self):
        all_attention_operators, all_attention_memories, queries = self.get_attentions()

        all_listed_rules = {}
        all_printed_rules = []
        for i, q in enumerate(queries):
            if not self.option.query_is_language:
                if (i+1) % max(1, (len(queries) / 5)) == 0:
                    sys.stdout.write("%d/%d\t" % (i, len(queries)))
                    sys.stdout.flush()
            else: 
                # Tuple-ize in order to be used as dict keys
                q = tuple(q)
            all_listed_rules[q] = list_rules(all_attention_operators[q], 
                                             all_attention_memories[q],
                                             self.option.rule_thr,)
            all_printed_rules += print_rules(q, 
                                             all_listed_rules[q], 
                                             self.data.parser,
                                             self.option.query_is_language)

        pickle.dump(all_listed_rules, 
                    open(os.path.join(self.option.this_expsdir, "rules.pckl"), "w"))
        with open(os.path.join(self.option.this_expsdir, "rules.txt"), "w") as f:
            for line in all_printed_rules:
                f.write(line + "\n")
        msg = self.msg_with_time("\nRules listed and printed.")
        print(msg)
        self.log_file.write(msg + "\n")

    def get_vocab_embedding(self):
        vocab_embedding = self.learner.get_vocab_embedding(self.sess)
        msg = self.msg_with_time("Vocabulary embedding retrieved.")
        print(msg)
        self.log_file.write(msg + "\n")
        
        vocab_embed_file = os.path.join(self.option.this_expsdir, "vocab_embed.pckl")
        pickle.dump({"embedding": vocab_embedding, "labels": self.data.query_vocab_to_number}, open(vocab_embed_file, "w"))
        msg = self.msg_with_time("Vocabulary embedding stored.")
        print(msg)
        self.log_file.write(msg + "\n")

    def close_log_file(self):
        self.log_file.close()

    def write_log_file(self, string):
        self.log_file = open(os.path.join(self.option.this_expsdir, "log.txt"), "a+")
        self.log_file.write(string+ "\n")
        self.log_file.close()

