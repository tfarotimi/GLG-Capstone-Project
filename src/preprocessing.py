import pandas as pd 
from torch import cuda 

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

def process_annotated_data(path, filename):
    '''
    This function takes in the path to the annotated data and the filename and returns a dataframe
    '''
    #read in data
    ner_dataset = pd.read_csv(path+filename,encoding='unicode-escape',on_bad_lines='skip')
    
    frequencies = ner_dataset['Tag'].value_counts()
    frequencies

    #count entities 
    tags = {}
    for tag, count in zip(frequencies.index, frequencies):
        if tag != 'O':
            if tag[2:5] not in tags.keys():
                tags[tag[2:5]] = count
            else:
                tags[tag[2:5]] += count


    ner_dataset = data = ner_dataset[~ner_dataset.Tag.isin(['B-nat','B-eve', 'B-art','I-nat','I-eve', 'I-art'])]

    #create dictionary for tags 
    labels_to_ids = {k:v for v, k in enumerate(ner_dataset.Tag.unique())}
    id_to_labels = {v: k for v, k in enumerate(data.Tag.unique())}


    #replace NaNs
    ner_dataset = ner_dataset.fillna(method='ffill')

    #group by sentence
    data = ner_dataset.groupby(ner_dataset['Sentence #'])

    #create sentence column
    ner_dataset['sentence'] =data['Word'].transform(lambda x:' '.join(x))

    #create sentence tags column
    ner_dataset['sentence_tags'] = data['Tag'].transform(lambda x:','.join(x))

    ner_dataset = ner_dataset[["sentence", "sentence_tags"]].drop_duplicates().reset_index(drop=True)

    return ner_dataset, labels_to_ids, id_to_labels
