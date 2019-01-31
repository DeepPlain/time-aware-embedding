import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_pretrained_glove(file):
    print("Loading")
    model = {}
    with open(file, 'r') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print(len(model),"words loaded")
    return model

def get_entity_from_txt(year):
    entities = []
    with open("dataset/original/wikipedia/wikipedia/" + year + ".txt", 'r') as f:
        event_section = "".join(f.readlines()).split("Births")[0]
        for line in event_section.split():
            entities.extend([word for word in line.split() if word.find("C_") == 0 and word.find("C_D_") == -1])
    return entities

def get_time_embedding(entities, model):
    embeddings = []
    count = 0
    for entity in entities:
        untagged_entity = entity.split('_')[2].lower()
        try:
            embeddings.append(model[untagged_entity])
            count += 1
#             print(untagged_entity)
        except Exception as e:
            continue
    avg = np.average(embeddings, axis=0)
    time_embedding = [round(val, 6) for val in avg.tolist()]
    print("-> glove에 존재하는 word 개수:", count)
    return time_embedding

def write_time_embedding(year, time_embedding):
    with open('dataset/time-embedding.txt', 'a') as f:
        embed_to_txt = ' '.join(map(str, time_embedding))
        f.write(year + ' ' + embed_to_txt + '\n')
        
if __name__ == "__main__":
    start_year = 1700
    end_year = 2019
    
    model = load_pretrained_glove("dataset/original/glove.6B.300d.txt") # load glove model
    years = list(range(start_year, end_year))
    time_embeddings = []
    appended_years = []
    # get time embeddings and write file
    for year in years: 
        year = str(year)
        try: 
            entities = set(get_entity_from_txt(year))
            print(year)
            appended_years.append(year)
            print("-> entity 개수: ",len(entities))
            time_embedding = get_time_embedding(entities, model)
            time_embeddings.append(time_embedding)
            write_time_embedding(year, time_embedding)
        except Exception as e:
            continue

    # PCA & Visualization
    pca = PCA(n_components=2)
    pca.fit(np.array(time_embeddings).transpose())
    plt.figure(figsize=(20,4))
    
    for i in range(len(pca.components_[0])):
        plt.scatter(pca.components_[0][i], pca.components_[1][i])
        plt.text(pca.components_[0][i] - 0.001, pca.components_[1][i] + 0.03, appended_years[i], fontsize=8)