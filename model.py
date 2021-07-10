import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torch.nn import init

#function input_data to return list of all questions answers and respective labels
# separately given path to file as input
def input_data(path):
    file=open(path,'r',encoding="utf-8")
    train=file.readlines()
    file.close()
    context=[]
    utter=[]
    label=[]
    for i in train:
        c,u,l=i.split('\t')
        context.append(c.lower().strip().split())
        utter.append(u.lower().strip().split())
        label.append(int(l.strip()))
    return context,utter,label

context_train,utter_train,label_train=input_data("WikiQA-train.txt")
context_test,utter_test,label_test=input_data("WikiQA-test.txt")
context_valid,utter_valid,label_valid=input_data("WikiQA-dev.txt")

file=open("WikiQA-train.txt",'r',encoding="utf-8")
train=file.readlines()
file.close()
all_answers=[]
for i in train:
    all_answers.append(i.split('\t')[1])
file1=open("allresponses.pkl",'ab')
pickle.dump(all_answers,file1)
file1.close()

label_train=np.array(label_train,dtype='float32')
label_test=np.array(label_test,dtype='float32')
label_valid=np.array(label_valid,dtype='float32')

#function to convert every word in the whole corpus into its repective index
def word_index(sentence,utter,sv,uv,st,ut):
    wordindex={}
    k=0
    train_words=sentence+utter+sv+uv+st+ut
    for sent in tqdm(train_words):
        for t in sent:
            if t not in wordindex.keys():
                wordindex[t]=k
                k+=1
            else:
                continue
    print(len(wordindex.keys()))
    return wordindex

words_dict=word_index(context_train,utter_train,context_test,context_valid,utter_test,utter_valid)
#dumping the dictionary to pkl file for reusing
file=open("words_dict",'ab')
pickle.dump(words_dict,file)
file.close()

# function returns maximum sentence length of all the sentences in a given list of sentences
def max_utt_len(sentence):
    maxi=0
    for i in sentence:
        maxi=max(len(i),maxi)
    return maxi

#converting to a fixed maximum length
for i in range(len(utter_train)):
    if len(utter_train[i])>150:
        utter_train[i]=utter_train[i][:150]

#function converts all the words in the sentence to their respective index
def wordtoindex(sentences,d):
    for sent in tqdm(sentences):
        for i in range(len(sent)):
            sent[i]=d[sent[i]]

wordtoindex(context_train,words_dict)
wordtoindex(utter_train,words_dict)
wordtoindex(context_test,words_dict)
wordtoindex(utter_test,words_dict)
wordtoindex(context_valid,words_dict)
wordtoindex(utter_valid,words_dict)

#function to convert all sentences to a fixed max length and return a mask for each sentence
#mask consists of 0's and 1's , 1 if word is present in a sentence and 0 if it isnt
def matrix(sentence,max_len):
    mask=[]
    for i in tqdm(range(len(sentence))):
        mask.append([1 for i in range(len(sentence[i]))])
        for j in range(max_len-len(sentence[i])):
            sentence[i].append(0)
            mask[i].append(0)
    return mask

context_train=np.array(context_train)

utter_test=np.array(utter_test)

context_test=np.array(context_test)

utter_valid=np.array(utter_valid)

context_valid=np.array(context_valid)

mask_utter_train=np.array(mask_utter_train)

mask_sentence_train=np.array(mask_sentence_train)

mask_utter_test=np.array(mask_utter_test)

mask_sentence_test=np.array(mask_sentence_test)

mask_utter_valid=np.array(mask_utter_valid)

mask_sentence_valid=np.array(mask_sentence_valid)

utter_train=np.array(utter_train)
#finally converted the given data into array of size (total sentences in corpus,max length of sentence in corpus)

#converting the pretrained glove vectors into a dictionary
file=open("glove.6B.50d.txt",'r',encoding="utf-8")
wordembeds=file.readlines()
file.close()
glove_embeds={}
for i in tqdm(wordembeds):
    ls=i.strip().split()
    for i in range(1,len(ls)):
        ls[i]=float(ls[i])
    glove_embeds[ls[0]]=ls[1:]

#function to create a dictionary mapping id of the words to their respective glove embeddings if the glove embedding is not
#present for a certain word then the embedding is to be randomly initiated
def id_to_glove(id_dict,glove_embeds):
    id_to_glove={}
    for word,embed in glove_embeds.items():
        if word in id_dict:
            id_to_glove[id_dict[word]]=np.array(embed,dtype='float32')
    for word,ind in id_dict.items():
        if ind not in id_to_glove:
            vec=np.zeros(50,dtype='float32')
            vec[:]=np.random.randn(50)*0.01
            id_to_glove[ind]=vec
    return id_to_glove

words_id_glove=id_to_glove(words_dict,glove_embeds)

#customdataloader class to return the input data as a batch size of neural network trainable data
class customdataloader(torch.utils.data.Dataset):
    def __init__(self,sent,utter,lab,sent_mask,utter_mask):
        self.sent=sent
        self.utter=utter
        self.lab=lab
        self.sent_mask=sent_mask
        self.utter_mask=utter_mask
    def __len__(self):
        return len(self.lab)
    def __getitem__(self,idx):
        return (self.sent[idx],self.utter[idx],self.lab[idx],self.sent_mask[idx],self.utter_mask[idx])


class Net(nn.Module):
    def __init__(self, Dictionary, word_embedding_length=50):
        super(Net, self).__init__()
        self.Dictionary = Dictionary
        self.lenDictionary = len(Dictionary)
        self.word_embedding_length = word_embedding_length
        self.embedding = nn.Embedding(self.lenDictionary, self.word_embedding_length)
        self.lstmBlock = nn.LSTM(self.word_embedding_length, self.word_embedding_length)
        self.dropout = nn.Dropout(0.5)
        self.init_weights()

    #function to get trained weights for words without embeddings
    def init_weights(self):
        # init.uniform(self.lstmBlock.weight_ih_l0,a=-0.01,b=0.01)
        # init.orthogonal(self.lstmBlock.weight_hh_l0)
        # self.lstmBlock.weight_ih_l0.requires_grad=True
        # self.lstmBlock.weight_hh_l0.requires_grad=True

        embedding_weights = torch.FloatTensor(self.lenDictionary, self.word_embedding_length)
        for idx, glove in self.Dictionary.items():
            embedding_weights[idx] = torch.FloatTensor(list(glove))
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=True)
        self.embedding = nn.Embedding.from_pretrained(self.embedding.weight)

    def forward(self, sent, masksent):
        out_sent = self.forwardLSTM(sent, masksent)
        return out_sent

    def forwardLSTM(self, utt, mask):
        output = torch.zeros([utt.shape[0], self.word_embedding_length]).to(utt.device)#output shape is (batchsize,50)
        for no, (utti, maski) in enumerate(zip(utt, mask)):
            utti_embed = self.embedding(utti)
            numutt = torch.sum(maski)
            utti_embed = utti_embed[:numutt].unsqueeze(1)
            _, (last_hidden, _) = self.lstmBlock(utti_embed)
            last_hidden = self.dropout(last_hidden[0][0])
            output[no] = last_hidden
        return output

def train(model,train_loader,optimizer,epoch):
    model.train()
    for batchid,(sent,utt,lab,masksent,maskutt) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        output_sent=model(sent,masksent)
        output_utt=model(utt,maskutt)
        #loss is mean squared loss between the dot product of resulting vector of question and answer
        # and its corresponding label
        loss=0
        for i in range(len(output_sent)):
            loss+=(output_sent[i].dot(output_utt[i])-lab[i])**2
        loss/=len(output_sent)
        loss.backward()
        optimizer.step()
        if batchid % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batchid * len(sent), len(train_loader.dataset),
            100. * batchid / len(train_loader), loss.item()))

def validation(model,valid_loader):
    model.eval()
    validloss=0
    correct=0
    with torch.no_grad():
        for batchid,(sent,utt,lab,masksent,maskutt) in enumerate(valid_loader):
            output_utt=model(utt,maskutt)
            output_sent=model(sent,masksent)
            for i in range(len(output_sent)):
                validloss+=(output_sent[i].dot(output_utt[i])-lab[i])**2
                if int(output_sent[i].dot(output_utt[i])+0.5)==lab[i]:
                    correct+=1
            validloss/=len(output_sent)
            print(validloss,100*correct/len(valid_loader.dataset))

def seed(seed_value):
    torch.cuda.manual_seed_all(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def main():
    seed(0)
    train_data=customdataloader(context_train,utter_train,label_train,mask_sentence_train,mask_utter_train)
    test_data=customdataloader(context_test,utter_test,label_test,mask_sentence_test,mask_utter_test)
    valid_data=customdataloader(context_valid,utter_valid,label_valid,mask_sentence_valid,mask_utter_valid)
    train_loader=DataLoader(train_data,num_workers=4,batch_size=40,shuffle=False)
    test_loader=DataLoader(test_data,num_workers=4,batch_size=1000,shuffle=False)
    valid_loader=DataLoader(valid_data,num_workers=4,batch_size=1000,shuffle=False)
    model=Net(words_id_glove)
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    answer_embeddings=[]
    for epoch in range(1,101):
        train(model,train_loader,optimizer,epoch)
        validation(model,valid_loader)
    validation(model,test_loader)

    for (sent, utt, lab, masksent, maskutt) in train_loader:
        output=model(utt)
        answer_embeddings.apppend(torch.tolist(output))
    file2 = open("answer_embeds.pkl", 'ab')
    pickle.dump(answer_embeddings,file2)
    file2.close()

    #saving the model and loading it
    torch.save(model.state_dict(), "wikiqafinal.pt")

if __name__=="__main__":
    main()

model=Net(words_id_glove)
model.load_state_dict(torch.load("wikiqafinal.pt"))

#saving the model in onnx format
dummy1=torch.randint(0,2000,(1,len(context_train[0])))
dummy2=torch.randint(0,2000,(1,len(context_train[0])))
torch.onnx.export(
    model,args=(dummy1,dummy2),f="model1.onnx",verbose=True,opset_version=11,
    input_names=['data1','data2'],output_names=['output1'],dynamic_axes={'data1':[1],'data2':[1]},
    )

