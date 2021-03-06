import torch, pickle
import torch.nn as nn
import numpy as np
import time, random
from utils import *


class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, option):
		super(RNNModel, self).__init__()
		rnn_type = 'LSTM'
		self.option = option
		dropout = option.dropout
		ntoken = option.vocab_size
		self.vocab_size = option.vocab_size
		ninp = option.emb_size
		nhid = option.hidden_size
		self.nlayers = option.num_layers
		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(ntoken, ninp)
		self.rnn = nn.LSTM(ninp, nhid, self.nlayers, dropout = dropout ,batch_first=True)
		self.decoder = nn.Linear(nhid, ntoken)
		self.init_weights()
		self.rnn_type = rnn_type
		self.nhid = nhid
		self.ntoken = ntoken
		self.criterion = nn.CrossEntropyLoss()
		self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, target):
		'''
		bs,15; bs,15
		'''
		batch_size = input.size(0)
		length = input.size(1)
		target = target.view(-1)
		emb = self.drop(self.encoder(input))
		c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		output, hidden = self.rnn(emb, (c0,h0))
		output = self.drop(output).contiguous().view(batch_size*length,-1)
		decoded = self.decoder(output)
		loss = self.criterion(decoded, target)
		v,idx = torch.max(decoded,1)
		acc = torch.mean(torch.eq(idx,target).float())
		return loss,acc, decoded.view(batch_size, length, self.ntoken)

	def predict1(self, input):
		'''
		bs,15; bs,15
		'''
		batch_size = input.size(0)
		length = input.size(1)

		emb = self.encoder(input)
		c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		output, hidden = self.rnn(emb, (c0,h0))
		decoded = nn.Softmax(2)(self.decoder(output)).view(batch_size*length,self.ntoken) # bs,l,vocab
 		# bs,l,
		target = torch.cat([input[:,1:],\
            torch.ones(batch_size,1).long().cuda()*(self.option.dict_size+1)],1).view(batch_size*length,-1) # bs,l
		output = torch.gather(decoded, 1, target).view(batch_size,length) # bs,l
		return output

	def predict(self, input):
		'''
		bs,15; bs,15
		'''
		batch_size = input.size(0)
		length = input.size(1)

		emb = self.encoder(input)
		c0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		h0 = torch.zeros(self.nlayers, batch_size, self.nhid).to(self.device)
		output, hidden = self.rnn(emb, (c0,h0))
		decoded = nn.Softmax(2)(self.decoder(output))
		return decoded



	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)


class PredictingModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, option):
        super(PredictingModel, self).__init__()
        rnn_type = 'LSTM'
        self.option = option
        dropout = option.dropout
        ntoken = option.vocab_size+1
        ninp = option.emb_size
        nhid = option.hidden_size
        self.nlayers = option.num_layers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, self.nlayers, dropout = dropout ,batch_first=True,\
                bidirectional=True )
        self.decoder = nn.Linear(nhid*2, ntoken)
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.ntoken = ntoken
        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, target):
        '''
        bs,15; bs,15
        '''
        batch_size = input.size(0)
        length = input.size(1)

        ind = int(random.random()*length)

        target = input[:,ind].clone()
        input[:,ind] = self.ntoken-1

        emb = self.drop(self.encoder(input))
        c0 = torch.zeros(2*self.nlayers, batch_size, self.nhid).to(self.device)
        h0 = torch.zeros(2*self.nlayers, batch_size, self.nhid).to(self.device)
        output, hidden = self.rnn(emb, (c0,h0)) # batch, length,2h
        pooled = nn.MaxPool1d(length)(output.permute(0,2,1)) #batch,2h
        decoded = self.decoder(pooled.squeeze(2))
        loss = self.criterion(decoded, target)
        v,idx = torch.max(decoded,1)
        acc = torch.mean(torch.eq(idx,target).float())
        return loss,acc, decoded.view(batch_size, self.ntoken)

    def predict(self, input):
        '''
        bs,15; bs,15
        '''
        batch_size = input.size(0)
        length = input.size(1)

        emb = self.encoder(input)
        c0 = torch.zeros(2*self.nlayers, batch_size, self.nhid).to(self.device)
        h0 = torch.zeros(2*self.nlayers, batch_size, self.nhid).to(self.device)
        output, hidden = self.rnn(emb, (c0,h0)) # batch, length,2h
        pooled = nn.MaxPool1d(length)(output.permute(0,2,1)) #batch,2h
        decoded = nn.Softmax(1)(self.decoder(pooled.squeeze(2)))
        return decoded



    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
 
class UPRL(nn.Module):
    def __init__(self, option):
        super(UPRL, self).__init__()
        rnn_type = 'LSTM'
        self.option = option
        dropout = option.dropout
        ntoken = option.vocab_size
        self.vocab_size = option.vocab_size
        ninp = option.emb_size
        nhid = option.hidden_size
        self.nlayers = option.num_layers
        self.drop = nn.Dropout(dropout)
        self.embedding= nn.Embedding(ntoken, ninp)

        self.rnn = nn.LSTM(ninp*2+2, nhid, self.nlayers, dropout = dropout ,batch_first=True,\
                bidirectional=True )

        self.repeat_size = 10
        self.decoder = nn.Linear(2*nhid, ntoken+1)
        self.n_token = ntoken
        self.nhid = nhid
        self.ntoken = ntoken
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.option.no_cuda else "cpu")

    def forward(self, input, key_pos, id2sen, emb_word):
        '''
        bs,15; bs,15
        '''

        print('====================')
        print(' '.join(id2sen(input[0])))
        print(key_pos)

        length = 15
        input = torch.tensor(input).long().view(1,length).repeat(self.repeat_size,1)
        key_pos_ = [key_pos for i in range(self.repeat_size)]
        key_pos = torch.tensor(key_pos_).float().view(self.repeat_size,length,1)
        N_step = 10
        st = input
        s0 = input
        N_step = 10
        pis = torch.zeros(self.repeat_size,N_step)
        actions = torch.zeros(self.repeat_size,N_step)
        rewards = torch.zeros(self.repeat_size,N_step)
        batch_size = self.repeat_size
        c0 = torch.zeros(2*self.nlayers, batch_size, self.nhid)
        h0 = torch.zeros(2*self.nlayers, batch_size, self.nhid)

        
        scores = self.f(st,s0, key_pos_,id2sen, emb_word)
        for i in range(N_step): 
            pos = i% length
            st,pi = self.step(st,s0, key_pos,pos, length,c0,h0)
            score_new = self.f(st,s0, key_pos_,id2sen, emb_word)
            reward  = score_new-scores
            
            #print(' '.join(id2sen(st[0].tolist())))
            #print(score_new, reward)

            pis[:,i:i+1] = pi
            rewards[:,i:i+1] = reward
            scores = score_new
            
        total_r = torch.sum(rewards,1)
        inc_flag = torch.gt(total_r, torch.mean(total_r)).float()
        rlloss =  -torch.log(pis.clamp(1e-6,1)) *rewards
        self.loss = torch.mean(rlloss,1)*inc_flag
        print(total_r, inc_flag)
        avg_rewards = torch.mean(rewards)
        return torch.mean(self.loss), avg_rewards
     

    def step(self, s_t_1, s0, key_pos, pos, length,c0,h0):
        # bs,L
        batch_size = key_pos.size(0)
        #print(s_t_1.size(), s0.size(), key_pos.size())
        pos_tensor = torch.zeros(self.repeat_size,length,1)
        pos_tensor[:,pos,:] = 1
        embt = self.embedding(s_t_1)
        emb0 = self.embedding(s0)
        emb = torch.cat([embt,emb0,pos_tensor,key_pos],2)
        output, hidden = self.rnn(emb, (c0,h0)) # batch, length,2h
        pooled = nn.MaxPool1d(length)(output.permute(0,2,1)) #batch,2h
        decoded = nn.Softmax(1)(self.decoder(pooled.squeeze(2))) # bs,V
        if self.training:
            action = decoded.multinomial(1)
        else: 
            values,action = torch.max(decoded,1)
        pi = torch.gather(decoded, 1, action) # (b_s,1), \pi for i,j
        
        replaceflag = torch.lt(action,self.n_token).long()
        st = torch.clone(s_t_1)
        st[:,pos:pos+1] = action*replaceflag +  (1-replaceflag)*s_t_1[:,pos:pos+1]
        return st, pi
    
    def f(self, st,s0, key_pos, id2sen, emb_word):
        xt = st.tolist()
        x0 = s0.tolist()
        sims =  similarity_batch(xt, x0, key_pos, id2sen, emb_word, self.option)
        return torch.tensor(sims,dtype=torch.float)

    def init_hidden(self, bsz):
    	weight = next(self.parameters())
    	if self.rnn_type == 'LSTM':
    		return (weight.new_zeros(self.nlayers, bsz, self.nhid),
    				weight.new_zeros(self.nlayers, bsz, self.nhid))
    	else:
    		return weight.new_zeros(self.nlayers, bsz, self.nhid)
    
