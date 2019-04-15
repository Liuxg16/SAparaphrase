import numpy as np
import RAKE, math
from zpar import ZPar
from data import array_data
import torch, sys,os
import pickle as pkl
from copy import copy
from bert.bertinterface import BertEncoding
from utils import get_corpus_bleu_scores

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
            # pos=zip(*[x.split('/') for x in pos_list])[0]
            pos=list(zip(*[x.split('/') for x in pos_list]))[0]
            print('sentence pos:',pos)
            print('keyword,',keyword)
            if keyword!=[]:
                keyword=list(list(zip(*keyword))[0])
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

def similarity_semantic_bleu(s1,s2, sta_vec, id2sen, emb_word, option, model):
    sourcesent = [' '.join(id2sen(s1))]
    rep1 = model.get_encoding(sourcesent)
    sourcesent2 = [' '.join(id2sen(s2))]
    rep2 = model.get_encoding(sourcesent2)
    norm1 = rep1.norm().item()
    norm2 = rep2.norm().item()
    semantic = torch.sum(rep1*rep2)/(rep1.norm()*rep2.norm())
    semantic = semantic*(1- (abs(norm1-norm2)/max(norm1,norm2)))
    actual_word_lists = [[id2sen(s2)]]
    generated_word_lists = [id2sen(s1)]
    bleu_scores = get_corpus_bleu_scores(actual_word_lists, generated_word_lists)[1]
    sim = semantic.item() * np.power((1-bleu_scores), 1.0/(len(id2sen(s1))))
    return sim

    def similarity_batch_word(s1, s2, sta_vec, option):
        return np.array([ similarity_word(x,s2,sta_vec, option) for x in s1 ])

def similarity_semantic(s1_list,s2, sta_vec, id2sen, emb_word, option, model):
    K = 4
    sourcesent = [' '.join(id2sen(s1)) for s1 in s1_list]
    rep1 = model.get_encoding(sourcesent)
    sourcesent2 = [' '.join(id2sen(s2))]
    rep2 = model.get_encoding(sourcesent2)
    summation = torch.sum(rep1*rep2,1).cpu().tolist()
    norm1 = rep1.norm(2,1).cpu().tolist()
    norm2 = rep2.norm().item()
    semantics = []
    for n, s in zip(norm1, summation):
        semantic = s/(n*norm2)
        semantic = semantic*(1- (abs(n-norm2)/max(n,norm2)))
        semantics.append(semantic)
    res = np.array(semantics)
    res = np.power(semantics,K)
    return res

def similarity_semantic_keyword(s1_list,s2, sta_vec, id2sen, emb_word, option, model):
    C1 = 0.5
    K = 4
    sourcesent = [' '.join(id2sen(s1)) for s1 in s1_list]
    rep1 = model.get_encoding(sourcesent)
    sourcesent2 = [' '.join(id2sen(s2))]
    rep2 = model.get_encoding(sourcesent2)
    summation = torch.sum(rep1*rep2,1).cpu().tolist()
    norm1 = rep1.norm(2,1).cpu().tolist()
    norm2 = rep2.norm().item()
    semantics = []
    for n, s, s1 in zip(norm1, summation, s1_list):
        semantic = s/(n*norm2)
        semantic = semantic*(1- (abs(n-norm2)/max(n,norm2)))
        tem = 1
        for i,x in zip(sta_vec,s2):
            if i==1 and x not in s1:
                tem *= C1
        semantic *= tem
        semantics.append(semantic)
    res = np.array(semantics)
    res = np.power(semantics,K)
    return res

def similarity_keyword(s1_list, s2, sta_vec, id2sen, emb_word, option, model = None):
    e=1e-5
    sims=  []
    for s1 in s1_list:
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
        # print('sss',sim_vec)
        # print wei2
        # sim=min([x for x in list(sim_vec*wei2) if x>0]+[1])
        sim=min([x for x,y in zip(list(sim_vec*wei2),list(wei2)) if y>0]+[1])
        sim = sigma_word(sim)
        sims.append(sim)
    res = np.array(sims)
    return res


    def similarity_batch_word(s1, s2, sta_vec, option):
        return np.array([ similarity_word(x,s2,sta_vec, option) for x in s1 ])

def cut_from_point(input, sequence_length, ind,option, mode=0):
    batch_size=input.shape[0]
    num_steps=input.shape[1]
    input_forward=np.zeros([batch_size, num_steps])+option.dict_size+1
    input_backward=np.zeros([batch_size, num_steps])+option.dict_size+1
    sequence_length_forward=np.zeros([batch_size])
    sequence_length_backward=np.zeros([batch_size])
    for i in range(batch_size):
        input_forward[i][0]=option.dict_size+2
        input_backward[i][0]=option.dict_size+2
        length=sequence_length[i]-1

        for j in range(ind):
            input_forward[i][j+1]=input[i][j+1]
        sequence_length_forward[i]=ind+1
        if mode==0:
            for j in range(length-ind-1):
                input_backward[i][j+1]=input[i][length-j]
            sequence_length_backward[i]=length-ind
        elif mode==1:
            for j in range(length-ind):
                input_backward[i][j+1]=input[i][length-j]
            sequence_length_backward[i]=length-ind+1
    return input_forward.astype(np.int32), input_backward.astype(np.int32), sequence_length_forward.astype(np.int32), sequence_length_backward.astype(np.int32)
   
def generate_candidate_input(input, sequence_length, ind, prob, search_size, option, mode=0):
    input_new=np.array([input[0]]*search_size)
    sequence_length_new=np.array([sequence_length[0]]*search_size)
    length=sequence_length[0]-1
    if mode!=2:
        ind_token=np.argsort(prob[: option.dict_size])[-search_size:]
        # print ind_token
    
    if mode==2:
        for i in range(sequence_length[0]-ind-2):
            input_new[: , ind+i+1]=input_new[: , ind+i+2]
        for i in range(sequence_length[0]-1, option.num_steps-1):
            input_new[: , i]=input_new[: , i]*0+option.dict_size+1
        sequence_length_new=sequence_length_new-1
        return input_new[:1], sequence_length_new[:1]
    if mode==1:
        for i in range(0, sequence_length_new[0]-1-ind):
            input_new[: , sequence_length_new[0]-i]=input_new[: ,  sequence_length_new[0]-1-i]
        sequence_length_new=sequence_length_new+1
    for i in range(search_size):
        input_new[i][ind+1]=ind_token[i]
    return input_new.astype(np.int32), sequence_length_new.astype(np.int32)

def normalize(x, e=0.05):
    tem = copy(x)
    if max(tem)==0:
        tem+=e
    return tem/tem.sum()

def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))

def just_acc(option):
    r=np.random.random()
    if r<option.just_acc_rate:
        return 0
    else:
        return 1

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

            if action==0: # word replacement (action: 0)
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

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)
                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(option.search_size):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]
                if input_candidate[prob_candidate_ind][ind+1]<option.dict_size and\
                        (prob_candidate_prob>prob_old_prob*option.threshold or just_acc(option)==0):
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        input= input1
                        print(' '.join(id2sen(input[0])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)

                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003

                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input[0], input_original, sta_vec, id2sen, emb_word, option)
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                #alpha is acceptance ratio of current proposal
                alpha=min(1, prob_candidate_prob*option.action_prob[2]/(prob_old_prob*option.action_prob[1]*prob_candidate_norm[prob_candidate_ind]))
            
                if choose_action([alpha, 1-alpha])==0 and \
                        input_candidate[prob_candidate_ind][ind]<option.dict_size and \
                        (prob_candidate_prob>prob_old_prob* option.threshold or just_acc(option)==0):
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1
                    pos+=1
                    sta_vec.insert(ind, 0.0)
                    del(sta_vec[-1])
                    print(' '.join(id2sen(input[0]))) 

            elif action==2: # word delete
                if sequence_length[0]<=2:
                    pos += 1
                    break

                prob_old = output_p(input, forwardmodel)
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input[0], input_original, sta_vec, id2sen, emb_word, option)
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, None, option.search_size, option, mode=action)

                # delete sentence
                prob_new = output_p(input_candidate, forwardmodel)
                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]
                prob_new_prob=tem
                if sim!=None:
                    similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option)
                    prob_new_prob=prob_new_prob*similarity_candidate
                
                # original sentence
                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=0)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=0)
                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003

                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
            
                if sim!=None:
                    similarity_candidate=similarity_batch(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option)
                    prob_candidate=prob_candidate*similarity_candidate
            
                prob_candidate_norm=normalize(prob_candidate)
                #alpha is acceptance ratio of current proposal
                if input[0] in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    alpha=min(prob_candidate_norm[candidate_ind]*prob_new_prob*option.action_prob[1]/(option.action_prob[2]*prob_old_prob), 1)
                else:
                    alpha=0
             
                if choose_action([alpha, 1-alpha])==0 and (prob_new_prob> prob_old_prob*option.threshold or just_acc(option)==0):
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    del(sta_vec[ind])
                    sta_vec.append(0)

                    pos -= 1
                    print(' '.join(id2sen(input[0])))

            pos += 1


def  simulatedAnnealing(option, dataclass,forwardmodel, backwardmodel, sim_mode = 'keyword'):
    sim=option.sim
    similaritymodel = None
    if sim_mode == 'keyword':
        similarity = similarity_keyword
    elif sim_mode =='semantic':
        similaritymodel =  BertEncoding()
        similarity = similarity_semantic
    elif sim_mode =='semantic-bleu':
        similaritymodel =  BertEncoding()
        similarity = similarity_semantic_bleu
    elif sim_mode =='semantic-keyword':
        similaritymodel =  BertEncoding()
        similarity = similarity_semantic_keyword


    generated_sentence = []
    fileemb = open(option.emb_path,'rb')
    emb_word,emb_id=pkl.load(fileemb, encoding = 'latin1')
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    C = 1 # 0.2
    
    for sen_id in range(use_data.length):
        #generate for each sentence
        sta_vec=sta_vec_list[sen_id%len(sta_vec)]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        for i in range(1,option.num_steps):
          if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
            sta_vec[i-1]=1
        pos=0

        print('Origin Sentence:')
        print(' '.join(id2sen(input[0])))
        print(sta_vec)
        print('Paraphrase:')

        for iter in range(option.sample_time):
            #ind is the index of the selected word, regardless of the beginning token.
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            steps = float(iter/(sequence_length[0]-1))
            temperature = C/(math.log(steps+2))

            if action==0: # word replacement (action: 0)
                prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                  similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                  prob_old_prob*=similarity_old
                else:
                  similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)
                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(option.search_size):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                sim_new = similarity_candidate[prob_candidate_ind]
                sim_old =similarity_old
                V_new = math.log(max(prob_candidate_prob,1e-200))
                V_old = math.log(max(prob_old_prob,1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,10)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        input= input1
                        # print('vold, vnew,simold, simnew',V_old, V_new,sim_old, sim_new)
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)

                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003

                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    tem = np.power(tem,(sequence_length[0]*1.0)/(sequence_length_candidate[0]))
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,\
                            option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                #alpha is acceptance ratio of current proposal
                sim_new = similarity_candidate[prob_candidate_ind]
                sim_old =similarity_old

                V_new = math.log(max(prob_candidate_prob, 1e-200))
                V_old = math.log(max(prob_old_prob,1e-200))
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size and (prob_candidate_prob>prob_old_prob* option.threshold):
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1
                    pos+=1
                    sta_vec.insert(ind, 0.0)
                    del(sta_vec[-1])
                    # print('vold, vnew,simold, simnew',V_old, V_new,sim_old, sim_new)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))
 

            elif action==2: # word delete
                if sequence_length[0]<=2:
                    pos += 1
                    break

                prob_old = output_p(input, forwardmodel)
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word, \
                            option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, None, option.search_size, option, mode=action)

                # delete sentence
                prob_new = output_p(input_candidate, forwardmodel)
                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]

                tem = np.power(tem,sequence_length[0]*1.0/(sequence_length_candidate[0]))
                prob_new_prob=tem
                if sim!=None:
                    similarity_new=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_new_prob=prob_new_prob*similarity_new
                
                sim_new = similarity_new[0]
                sim_old =similarity_old
                V_new = math.log(max(prob_new_prob,1e-300))
                V_old = math.log(max(prob_old_prob,1e-300))
                
                alphat = min(1,math.exp((V_new-V_old)/temperature))

                if sim_new<=sim_old:
                    alphat=0
                      
                if choose_action([alphat, 1-alphat])==0:
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    pos-=1
                    del(sta_vec[ind])
                    sta_vec.append(0)
                    
                    # print('vold, vnew,simold, simnew',V_old, V_new,sim_old, sim_new)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

            pos += 1
        generated_sentence.append(id2sen(input[0]))
    return generated_sentence

def  simulatedAnnealing_std(option, dataclass,forwardmodel, backwardmodel, sim_mode = 'keyword'):
    sim=option.sim
    similaritymodel = None
    if sim_mode == 'keyword':
        similarity = similarity_keyword
    elif sim_mode =='semantic':
        similaritymodel =  BertEncoding()
        similarity = similarity_semantic
    elif sim_mode =='semantic-bleu':
        similaritymodel =  BertEncoding()
        similarity = similarity_semantic_bleu
    elif sim_mode =='semantic-keyword':
        similaritymodel =  BertEncoding()
        similarity = similarity_semantic_keyword


    generated_sentence = []
    fileemb = open(option.emb_path,'rb')
    emb_word,emb_id=pkl.load(fileemb, encoding = 'latin1')
    sta_vec=list(np.zeros([option.num_steps-1]))

    use_data, sta_vec_list = read_data_use(option, dataclass.sen2id)
    id2sen = dataclass.id2sen
    C = 5 # 0.2
    
    for sen_id in range(use_data.length):
        #generate for each sentence
        sta_vec=sta_vec_list[sen_id%len(sta_vec)]
        input, sequence_length, _=use_data(1, sen_id)
        input_original=input[0]
        for i in range(1,option.num_steps):
          if input[0][i]>option.rare_since and  input[0][i]<option.dict_size:
            sta_vec[i-1]=1
        pos=0

        print('Origin Sentence:')
        print(' '.join(id2sen(input[0])))
        print(sta_vec)
        print('Paraphrase:')

        for iter in range(option.sample_time):
            #ind is the index of the selected word, regardless of the beginning token.
            print(iter)
            ind=pos%(sequence_length[0]-1)
            action=choose_action(option.action_prob)
            steps = float(iter/(sequence_length[0]-1))
            temperature = C/(math.log(steps+2))

            if action==0: # word replacement (action: 0)
                prob_old= output_p(input, forwardmodel) #15,K
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                  similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,
                          option, similaritymodel)[0]
                  prob_old_prob*=similarity_old
                else:
                  similarity_old=-1

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)
                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003
                prob_candidate=[]
                for i in range(option.search_size):
                  tem=1
                  for j in range(sequence_length[0]-1):
                    tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                  tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                  prob_candidate.append(tem)
          
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                sim_new = similarity_candidate[prob_candidate_ind]
                sim_old =similarity_old
                V_new = math.log(max(prob_candidate_prob,1e-200))
                V_old = math.log(max(prob_old_prob, 1e-200))
                alphat = min(1,math.exp((V_new-V_old)/temperature))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size:
                    input1=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    if np.sum(input1[0])==np.sum(input[0]):
                        pass
                    else:
                        input= input1
                        print('vold, vnew,simold, simnew',V_old, V_new,sim_old, sim_new)
                        print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

            elif action==1: # word insert
                if sequence_length[0]>=option.num_steps:
                    pos += 1
                    break

                input_forward, input_backward, sequence_length_forward, sequence_length_backward =\
                        cut_from_point(input, sequence_length, ind, option, mode=action)
                prob_forward = output_p(input_forward, forwardmodel)[ind%(sequence_length[0]-1),:]
                prob_backward = output_p(input_backward,backwardmodel)[
                        sequence_length[0]-1-ind%(sequence_length[0]-1),:]
                prob_mul=(prob_forward*prob_backward)

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, prob_mul, option.search_size, option, mode=action)

                prob_candidate_pre = output_p(input_candidate, forwardmodel) # 100,15,300003

                prob_candidate=[]
                for i in range(option.search_size):
                    tem=1
                    for j in range(sequence_length_candidate[0]-1):
                        tem*=prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem*=prob_candidate_pre[i][j+1][option.dict_size+1]
                    tem = np.power(tem,(sequence_length[0]*1.0)/(sequence_length_candidate[0]))
                    prob_candidate.append(tem)
                prob_candidate=np.array(prob_candidate)
                if sim!=None:
                    similarity_candidate=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_candidate=prob_candidate*similarity_candidate
                prob_candidate_norm=normalize(prob_candidate)
                prob_candidate_ind=sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob=prob_candidate[prob_candidate_ind]

                prob_old = output_p(input, forwardmodel) # 100,15,300003

                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word,\
                            option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1
                #alpha is acceptance ratio of current proposal
                sim_new = similarity_candidate[prob_candidate_ind]
                sim_old =similarity_old

                V_new = math.log(prob_candidate_prob)
                V_old = math.log(prob_old_prob)
                alphat = min(1,math.exp(min((V_new-V_old)/temperature,200)))
                if choose_action([alphat, 1-alphat])==0 and input_candidate[prob_candidate_ind][ind]<option.dict_size and (prob_candidate_prob>prob_old_prob* option.threshold):
                    input=input_candidate[prob_candidate_ind:prob_candidate_ind+1]
                    sequence_length+=1
                    pos+=1
                    sta_vec.insert(ind, 0.0)
                    del(sta_vec[-1])
                    print('vold, vnew,simold, simnew',V_old, V_new,sim_old, sim_new)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))
 

            elif action==2: # word delete
                if sequence_length[0]<=2:
                    pos += 1
                    break

                prob_old = output_p(input, forwardmodel)
                tem=1
                for j in range(sequence_length[0]-1):
                    tem*=prob_old[j][input[0][j+1]]
                tem*=prob_old[j+1][option.dict_size+1]
                prob_old_prob=tem
                if sim!=None:
                    similarity_old=similarity(input, input_original, sta_vec, id2sen, emb_word, \
                            option, similaritymodel)[0]
                    prob_old_prob=prob_old_prob*similarity_old
                else:
                    similarity_old=-1

                input_candidate, sequence_length_candidate=generate_candidate_input(input,\
                        sequence_length, ind, None, option.search_size, option, mode=action)

                # delete sentence
                prob_new = output_p(input_candidate, forwardmodel)
                tem=1
                for j in range(sequence_length_candidate[0]-1):
                    tem*=prob_new[j][input_candidate[0][j+1]]
                tem*=prob_new[j+1][option.dict_size+1]

                tem = np.power(tem,sequence_length[0]*1.0/(sequence_length_candidate[0]))
                prob_new_prob=tem
                if sim!=None:
                    similarity_new=similarity(input_candidate, input_original,sta_vec,\
                            id2sen, emb_word, option, similaritymodel)
                    prob_new_prob=prob_new_prob*similarity_new
                
                sim_new = similarity_new[0]
                sim_old =similarity_old
                V_new = math.log(max(prob_new_prob,1e-300))
                V_old = math.log(prob_old_prob)
                
                alphat = min(1,math.exp((V_new-V_old)/temperature))
                      
                if choose_action([alphat, 1-alphat])==0:
                    input=np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1]*0+option.dict_size+1], axis=1)
                    sequence_length-=1
                    pos-=1
                    del(sta_vec[ind])
                    sta_vec.append(0)
                    
                    print('vold, vnew,simold, simnew',V_old, V_new,sim_old, sim_new)
                    print('Temperature:{:3.3f}:   '.format(temperature)+' '.join(id2sen(input[0])))

            pos += 1
        generated_sentence.append(id2sen(input[0]))
    return generated_sentence

