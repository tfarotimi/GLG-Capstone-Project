from transformers import BertTokenizerFast 
import torch
import numpy as np



def perform_NER(model, text, label_dict):

  #check if gpu is available
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  #set max length of input sequence
  MAX_LEN = 128



  tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

  
  NER_model = model
  sentence = text

  '''

  '''

  inputs = tokenizer(sentence.split(),
                      return_offsets_mapping=True, 
                      padding='max_length', 
                      truncation=True, 
                      max_length=MAX_LEN,
                      return_tensors="pt")

  # move to gpu
  ids = inputs["input_ids"].to(device)
  mask = inputs["attention_mask"].to(device)
  # forward pass
  outputs = NER_model(ids, attention_mask=mask)
  logits = outputs[0]

  active_logits = logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
  flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

  #import pdb 
  #pdb.set_trace()

  tokens = tokenizer.convert_ids_to_tokens(ids.reshape(-1).tolist())
  token_predictions = [label_dict[i] for i in flattened_predictions.cpu().numpy()]
  wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

  prediction = []

  for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].view(len(wp_preds),2).tolist()):

    
    #only predictions on first word pieces are important
    if mapping[0] == 0 and mapping[1] != 0:
      prediction.append(token_pred[1])
    else:
      continue

  print(sentence.split())
  print(prediction)