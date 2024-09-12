import numpy as np
import pandas as pd

def get_stat(dataset):
    # List of columns to compute statistics for
    columns = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
    
    # Dictionary to store mean and std for each column
    stats = {}
    
    for column in columns:
        stats[column] = {
            'mean': dataset[column].mean(),
            'std': dataset[column].std(ddof=0)  # Use ddof=0 for population standard deviation
        }
    
    return stats

def dataset_z_score(dataset, stats):
    columns = ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
    
    for column in columns:
        mean = stats[column]['mean']
        std = stats[column]['std']
        dataset[f'{column}_zscore'] = (dataset[column] - mean) / std
    
    return dataset

def convert_list_to_dataset(person_list, matching, questions):
    data_temp =pd.DataFrame([person_list], columns= [column for column in matching])   
    for column in data_temp.columns:
        data_temp[column] = data_temp[column] * questions[column][1]
    data_temp['extraversion'] = data_temp.iloc[:,0:10].sum(axis=1) + 20
    data_temp['neuroticism'] = data_temp.iloc[:,10:20].sum(axis=1) +38
    data_temp['agreeableness'] = data_temp.iloc[:,20:30].sum(axis=1) +14 
    data_temp['conscientiousness'] = data_temp.iloc[:,30:40].sum(axis=1) + 14
    data_temp['openness'] = data_temp.iloc[:,40:50].sum(axis=1) + 8
    
    return data_temp

def categorie_description(dataset):
    if dataset <= -2:
        text = 'The candidate is extremely '
    elif (dataset < -2) & (dataset <= -1):
        text = 'The candidate is very '
    elif (dataset > -1) & (dataset <= -0.5):
        text = 'The candidate is quite '
    elif (dataset > -0.5) & (dataset <= 0.5):
        text = 'The candidate is relatively '
    elif (dataset > 0.5) & (dataset <= 1):
        text = 'The candidate is quite'
    elif (dataset > 1) & (dataset <= 2):
        text = 'The candidate is very '
    elif dataset > 2:
        text = 'The candidate is extremely '
        
    return text



def prepare_dataset(data):
    data = data.copy()
    data.drop(data.columns[50:107], axis=1, inplace=True)
    data.drop(data.columns[50:], axis=1, inplace=True) # here 50 to remove the country
    data.dropna(inplace=True)

    # Groups and Questions modify version
    # (1) extraversion, (2) neuroticism, (3) agreeableness, (4)conscientiousness , and (5) openness
    ext_questions = {'EXT1' : ['they are the life of the party',1],
                     'EXT2' : ['they dont talk a lot',-1],
                     'EXT3' : ['they feel comfortable around people',1],
                     'EXT4' : ['they keep in the background',-1],
                     'EXT5' : ['they start conversations',1],
                     'EXT6' : ['they have little to say',-1],
                     'EXT7' : ['they talk to a lot of different people at parties',1],
                     'EXT8' : ['they dont like to draw attention to themself',-1],
                     'EXT9' : ['they dont mind being the center of attention',1],
                     'EXT10': ['they are quiet around strangers',-1]}
    est_questions = {'EST1' : ['they get stressed out easily',-1],
                     'EST2' : ['they are relaxed most of the time',1],
                     'EST3' : ['they worry about things',-1],
                     'EST4' : ['they seldom feel blue',1],
                     'EST5' : ['they are easily disturbed',-1],
                     'EST6' : ['they get upset easily',-1],
                     'EST7' : ['they change their mood a lot',-1],
                     'EST8' : ['they have frequent mood swings',-1],
                     'EST9' : ['they get irritated easily',-1],
                     'EST10': ['they often feel blue',-1]}
    agr_questions = {'AGR1' : ['they feel little concern for others',-1],
                     'AGR2' : ['they interested in people',1],
                     'AGR3' : ['they insult people',-1],
                     'AGR4' : ['they sympathize with others feelings',1],
                     'AGR5' : ['they are not interested in other peoples problems',-1],
                     'AGR6' : ['they have a soft heart',1],
                     'AGR7' : ['they not really interested in others',-1],
                     'AGR8' : ['they take time out for others',1],
                     'AGR9' : ['they feel others emotions',1],
                     'AGR10': ['they make people feel at ease',1]}
    

    csn_questions = {'CSN1' : ['they are always prepared',1],
                     'CSN2' : ['they leave their belongings around',-1],
                     'CSN3' : ['they pay attention to details',1],
                     'CSN4' : ['they make a mess of things',-1],
                     'CSN5' : ['they get chores done right away',1],
                     'CSN6' : ['they often forget to put things back in their proper place',-1],
                     'CSN7' : ['they like order',1],
                     'CSN8' : ['they shirk their duties',-1],
                     'CSN9' : ['they follow a schedule',1],
                     'CSN10' : ['they are exacting in their work',1]}

    opn_questions = {'OPN1' : ['they have a rich vocabulary',1],
                     'OPN2' : ['they have difficulty understanding abstract ideas',-1],
                     'OPN3' : ['they have a vivid imagination',1],
                     'OPN4' : ['they are not interested in abstract ideas',-1],
                     'OPN5' : ['they have excellent ideas',1],
                     'OPN6' : ['they do not have a good imagination',-1],
                     'OPN7' : ['they are quick to understand things',1],
                     'OPN8' : ['they use difficult words',1],
                     'OPN9' : ['they spend time reflecting on things',1],
                     'OPN10': ['they are full of ideas',1]}
    
    questions = ext_questions | est_questions | agr_questions | csn_questions  | opn_questions

    # Group Names and Columns
    EXT = [column for column in data if column.startswith('EXT')]
    EST = [column for column in data if column.startswith('EST')]
    AGR = [column for column in data if column.startswith('AGR')]
    CSN = [column for column in data if column.startswith('CSN')]
    OPN = [column for column in data if column.startswith('OPN')]

    matching = EXT+EST+AGR+CSN+OPN

    # Here we update the dataframe by applying the new coefficient
    for column in data.columns:
        data[column] = data[column] * questions[column][1]

    # reference to scoring: https://sites.temple.edu/rtassessment/files/2018/10/Table_BFPT.pdf 
    data['extraversion'] = data.iloc[:, 0:10].sum(axis=1) + 20
    data['neuroticism'] = data.iloc[:, 10:20].sum(axis=1) +38
    data['agreeableness'] = data.iloc[:, 20:30].sum(axis=1) +14 
    data['conscientiousness'] = data.iloc[:, 30:40].sum(axis=1) + 14
    data['openness'] = data.iloc[:, 40:50].sum(axis=1) + 8
    data['name'] = data.index.to_series().apply(lambda idx: 'C_' + str(idx))

    return data, questions, matching



def get_description(candidate_data, dataset):
    
    # Upload the dataset
    data, questions, matching = prepare_dataset(dataset)
    stats = get_stat(dataset)
    data = dataset_z_score(data, stats)

    # First we want to check if the user want a certain candidate from the dataset 
    # or if the user did the test so it return a list
    if isinstance(candidate_data, list):
        data_c = convert_list_to_dataset(candidate_data, matching, questions)
        data_c = dataset_z_score(data_c, stats)
        
    
    elif isinstance(candidate_data,(int, float)):
        if candidate_data < 0:
            print('The number should be greater or equal to 0')
        else:
            data_c = pd.DataFrame([data.iloc[candidate_data]])
        
    text = []
    
    for i in range(0,5):
        
        # extraversion
        if i == 0:
            cat_0 = 'solitary and reserved. '
            cat_1 = 'outgoing and energetic. '
        
            if data_c['extraversion_zscore'].values > 0:
                text_t = categorie_description(data_c['extraversion_zscore'].values) + cat_1
                if data_c['extraversion_zscore'].values > 1:
                    index_max = data_c.iloc[0,0:10].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0]+'. '
                    text_t = text_t + text_2
            else:
                text_t = categorie_description(data_c['extraversion_zscore'].values) + cat_0
                if data_c['extraversion_zscore'].values < -1:
                    index_min = data_c.iloc[0,0:10].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0]+'. '
                    text_t = text_t + text_2
            text.append(text_t)
                
        # neuroticism
        if i == 1:
            cat_0 = 'resilient and confident. '
            cat_1 = 'sensitive and nervous. '
            
            if data_c['neuroticism_zscore'].values > 0:
                text_t = categorie_description(data_c['neuroticism_zscore'].values) + cat_1  \
                + 'The candidate tends to feel more negative emotions, anxiety. '
                if data_c['neuroticism_zscore'].values > 1:
                    index_max = data_c.iloc[0,10:20].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0]+'. '
                    text_t = text_t + text_2
                
            else:
                text_t = categorie_description(data_c['neuroticism_zscore'].values) + cat_0  \
                + 'The candidate tends to feel less negative emotions, anxiety. '
                if data_c['neuroticism_zscore'].values < -1:
                    index_min = data_c.iloc[0,10:20].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0]+'. '
                    text_t = text_t + text_2
            text.append(text_t)
            
        # agreeableness        
        if i == 2:
            cat_0 = 'critical and rational. '
            cat_1 = 'friendly and compassionate. '
            
            if data_c['agreeableness_zscore'].values > 0:
                text_t = categorie_description(data_c['agreeableness_zscore'].values) + cat_1  \
                + 'The candidate tends to be more cooperative, polite, kind and friendly. '
                if data_c['agreeableness_zscore'].values > 1:
                    index_max = data_c.iloc[0,20:30].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                    text_t = text_t + text_2

            else:
                text_t = categorie_description(data_c['agreeableness_zscore'].values) + cat_0  \
                + 'The candidate tends to be less cooperative, polite, kind and friendly. '
                if data_c['agreeableness_zscore'].values < -1:
                    index_min = data_c.iloc[0,20:30].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                    text_t = text_t + text_2
            text.append(text_t)
            
        # conscientiousness
        if i == 3:  
            cat_0 = 'extravagant and careless. '
            cat_1 = 'efficient and organized. '
            
            if data_c['conscientiousness_zscore'].values > 0:
                text_t = categorie_description(data_c['conscientiousness_zscore'].values) + cat_1  \
                + 'The candidate tends to be more careful or diligent. '
                if data_c['conscientiousness_zscore'].values > 1:
                    index_max = data_c.iloc[0,30:40].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                    text_t = text_t + text_2
            else:
                text_t = categorie_description(data_c['conscientiousness_zscore'].values) + cat_0  \
                + 'The candidate tends to be less careful or diligent. '
                if data_c['conscientiousness_zscore'].values < -1:
                    index_min = data_c.iloc[0,30:40].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                    text_t = text_t + text_2
            text.append(text_t)
        
        # openness
        if i == 4:
            cat_0 = 'consistent and cautious. '
            cat_1 = 'inventive and curious. '
            
            if data_c['openness_zscore'].values > 0:
                text_t = categorie_description(data_c['openness_zscore'].values) + cat_1  \
                + 'The candidate tends to be more open. '
                if data_c['openness_zscore'].values > 1:
                    index_max = data_c.iloc[0,40:50].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                    text_t = text_t + text_2
            else:
                text_t = categorie_description(data_c['openness_zscore'].values) + cat_0  \
                + 'The candidate tends to be less open. '
                if data_c['openness_zscore'].values < -1:
                    index_min = data_c.iloc[0,40:50].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                    text_t = text_t + text_2
            text.append(text_t)
        i+=1
        
    text = ''.join(text)
    text = text.replace(',','')
    return text
