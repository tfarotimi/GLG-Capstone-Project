from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


#create a dataset object for use in BERT NER model
class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, label_dict, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.max_len = max_len
      
      

  def __getitem__(self, index):
        # step 1: get the sentence and word labels 
        sentence = self.data.sentence.to_numpy()[index]  
        sentence_tags = self.data.sentence_tags.iloc[index].split(",") 

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                             return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [self.label_dict[label] for label in sentence_tags] 

        #labels = labels.to_numpy()
        
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            encoded_labels[idx] = labels[i]
            i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

  def __len__(self):
        return self.len

  