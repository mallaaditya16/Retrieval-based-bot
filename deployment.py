import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torchvision import transforms
from torch.nn import init
import pickle
import onnxruntime as ort

#loading all the necessary files dumped during model training
file=open("words_dict",'rb')
words_dict=pickle.load(file)
file.close()
file1=open("allresponses.pkl",'rb')
answers=pickle.load(file1)
file1.close()
file2=open("answer_embeds(1).pkl",'rb')
ls_utt=pickle.load(file2)
file2.close()
ls_utt=np.array(ls_utt)
ort_session=ort.InferenceSession("model1.onnx")

#function returns the best possible answer from the training corpus responses for a given input string
def bestanswer(questionmatrix,mask_ques_matrix):
    output=ort_session.run(None , {'data1':questionmatrix,'data2':mask_ques_matrix})
    return np.argmax(np.matmul(output,ls_utt.T)[0])

#function to preprocess the input string
def preprocess(s, words_dict):
    mask = [[]]
    ls = [[]]
    ls[0] = s.split()
    i = 0
    n = len(ls[0])
    while i < n:
        if ls[0][i] in words_dict:
            ls[0][i] = words_dict[ls[0][i]]
            mask[0].append(1)
            i += 1
        else:
            ls[0].pop(i)
            n -= 1
    ls = np.array(ls,dtype='int64')
    mask = np.array(mask,dtype='int64')
    return ls, mask

def test(input_string,answers):
    ques_matrix,mask_ques_matrix=preprocess(input_string,words_dict)
    if len(ques_matrix[0])==0:
        return "Hi I do not understand the question."
    id_best=bestanswer(ques_matrix,mask_ques_matrix)
    return answers[id_best]


