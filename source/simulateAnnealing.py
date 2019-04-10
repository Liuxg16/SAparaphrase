import numpy as np
import RAKE, math
from zpar import ZPar
from data import array_data
import torch
import pickle as pkl

def output_p(sent, model):
    # list
    sent = torch.tensor(sent, dtype=torch.long).cuda()
    output = model.predict(sent) # 1,15,300003
    return output.squeeze(0).cpu().detach().numpy()

def keyword_pos2sta_vec(option,keyword, pos):
    key_ind=[]
    pos=pos[:option.num_steps-1]
    for i in range(len(pos)):
        if pos[i]=='NNP':
            key_ind.append(i)
        elif pos[i] in ['NN', 'NNS'] and keyword[i]==1:
            key_ind.append(i)
        elif pos[i] in ['VBZ'] and keyword[i]==1:
            key_ind.append(i)
        elif keyword[i]==1:
            key_ind.append(i)
        elif pos[i] in ['NN', 'NNS','VBZ']:
            key_ind.append(i)
    key_ind=key_ind[:max(int(option.max_key_rate*len(pos)), option.max_key)]
    sta_vec=[]
    for i in range(len(keyword)):
        if i in key_ind:
            sta_vec.append(1)
        else:
            sta_vec.append(0)
    return sta_vec

def read_data_use(option,  sen2id):

    file_name = option.use_data_path
    max_length = option.num_steps
    dict_size = option.dict_size
    Rake = RAKE.Rake(RAKE.SmartStopList())
    z=ZPar(option.pos_path)
    tagger = z.get_tagger()
    with open(file_name) as f:
        data=[]
        vector=[]
        sta_vec_list=[]
        j=0
        for line in f:
            print('sentence:'+line)
            sta_vec=list(np.zeros([option.num_steps-1]))
            keyword=Rake.run(line.strip())
            pos_list=tagger.tag_sentence(line.strip()).split()
            pos=zip(*[x.split('/') for x in pos_list])[0]
            print('sentence pos:',pos)
            print('keyword,',keyword)
            if keyword!=[]:
                keyword=list(zip(*keyword)[0])
                keyword_new=[]
                for item in keyword:
                    tem1=[line.strip().split().index(x) for x in item.split() if x in line.strip().split()]
                    keyword_new.extend(tem1)
                for i in range(len(keyword_new)):
                    ind=keyword_new[i]
                    if ind<option.num_steps-2:
                        sta_vec[ind]=1
            if option.keyword_pos==True:
                sta_vec_list.append(keyword_pos2sta_vec(option,sta_vec,pos))
            else:
                sta_vec_list.append(list(np.zeros([option.num_steps-1])))
            data.append(sen2id(line.strip().lower().split()))
    print('data', data)
    data_new=array_data(data, max_length, dict_size)
    print('------------------------')
    return data_new, sta_vec_list # sentence, keyvector

def choose_action(c):
    r=np.random.random()
    c=np.array(c)
    for i in range(1, len(c)):
        c[i]=c[i]+c[i-1]
    for i in range(len(c)):
        if c[i]>=r:
            return i

def sigma_word(x):
    if x>0.7:
        return x
    elif x>0.65:
        return (x-0.65)*14
    else:
        return 0
    #return max(0, 1-((x-1))**2)
    #return (((np.abs(x)+x)*0.5-0.6)/0.4)**2
def sen2mat(s, id2sen, emb_word, option):
    mat=[]
    for item in s:
        if item==option.dict_size+2:
            continue
        if item==option.dict_size+1:
            break
        word=id2sen([item])[0]
        if  word in emb_word:
            mat.append(np.array(emb_word[word]))
        else:
            mat.append(np.random.random([option.hidden_size]))
    return np.array(mat)

def similarity(s1,s2, sta_vec, id2sen, emb_word, option):
    e=1e-5
    emb1=sen2mat(s1, id2sen, emb_word, option)
    #wei2=normalize( np.array([-np.log(id2freq[x]) for x in s2 if x<=config.dict_size]))
    emb2=sen2mat(s2, id2sen, emb_word, option)
    wei2=np.array(sta_vec[:len(emb2)]).astype(np.float32)
    #wei2=normalize(wei2)
    
    emb_mat=np.dot(emb2,emb1.T)
    norm1=np.diag(1/(np.linalg.norm(emb1,2,axis=1)+e))
    norm2=np.diag(1/(np.linalg.norm(emb2,2,axis=1)+e))
    sim_mat=np.dot(norm2,emb_mat).dot(norm1)
    sim_vec=sim_mat.max(axis=1)
    #sim=(sim_vec*wei2).sum()
    # print('sss',sim_vec*wei2)
    sim=min([x for x in list(sim_vec*wei2) if x>0]+[1])
    # sim=(sim_vec).mean()
    return sigma_word(sim)

    def similarity_batch_word(s1, s2, sta_vec, option):
        return np.array([ similarity_word(x,s2,sta_vec, option) for x in s1 ])




def metropolisHasting(option, dataclass,forwardmodel, backwardmodel):

    emb_word,emb_id=pkl.load(open(option.emb_path))
    sim=option.sim
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    
    for sen_id in range(use_data.length):
        #generate for each sentence
        sta_vec=sta_vec_list[sen_id%len(sta_vec)]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        for i in range(1,option.num_steps):
          if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
            sta_vec[i-1]=1
        pos=0
        print(' '.join(id2sen(input[0])))
        print(sta_vec)

        for iter in range(option.sample_time):
            #ind is the index of the selected word, regardless of the beginning token.
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            print(' '.join(id2sen(input[0])))

            # word replacement (action: 0)
            if action==0: 
                prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                  tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                  similarity_old=similarity(input[0], input_original, sta_vec, id2sen, emb_word, option)
                  prob_old_prob*=similarity_old
                else:
                  similarity_old=-1
                print math.log(prob_old_prob)
