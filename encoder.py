from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
import numpy as np

def encode(data,labelvocab,config):

    labelvocab.add_label('positive')
    labelvocab.add_label('negative')
    labelvocab.add_label('neutral')
    labelvocab.add_label('null') 

    tokenizer=AutoTokenizer.from_pretrained(config.bert_name)

    def img_re(image_size):
        for i in range(16):
            if np.power(2,i)>=image_size:
                return np.power(2,i)
        return image_size
    
    guids=[]
    encoded_texts=[]
    encoded_imgs=[]
    encoded_labels=[]

    img_changed=transforms.Compose([
        transforms.Resize(int(img_re(config.image_size))),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4965252,0.45888674,0.43725428],std=[0.25523951,0.24872537,0.2451266 ])
    ])
    #print(1)
    for item in tqdm(data,desc='it is encoding!'):
        guid,text,img,label=item
        #print(guid,text,img,label)
        guids.append(guid)
        
        text.replace('#','')
        tokens=tokenizer.tokenize('[CLS]'+text+'[SEP]')
        encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))

        encoded_imgs.append(img_changed(img))

        encoded_labels.append(labelvocab.label_to_id(label))
        #print("here!!!")
    return guids, encoded_texts, encoded_imgs, encoded_labels
