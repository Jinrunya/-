import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MyDataset(Dataset):

    def __init__(self,guids,texts,imgs,labels):
            self.guids=guids
            self.texts=texts
            self.imgs=imgs
            self.labels=labels

    def __getitem__(self, index): 
        return self.guids[index],self.texts[index],self.imgs[index],self.labels[index]

    def __len__(self):
        return(len(self.guids))

    def mask_and_pad(self,batch):
        guids = [b[0] for b in batch]

        texts = [torch.LongTensor(b[1]) for b in batch]
        texts_mask = [torch.ones_like(text) for text in texts]

        imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch]) 

        labels = torch.LongTensor([b[3] for b in batch])

        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)
        return guids, paded_texts, paded_texts_mask, imgs, labels
        