def categorie_description(data_pz):
    if data_pz <= -2:
        text = 'The candidate is extremely '
    elif (data_pz < -2) & (data_pz <= -1):
        text = 'The candidate is very '
    elif (data_pz > -1) & (data_pz <= -0.5):
        text = 'The candidate is quite '
    elif (data_pz > -0.5) & (data_pz <= 0.5):
        text = 'The candidate is relatively '
    elif (data_pz > 0.5) & (data_pz <= 1):
        text = 'The candidate is quite'
    elif (data_pz > 1) & (data_pz <= 2):
        text = 'The candidate is very '
    elif data_pz > 2:
        text = 'The candidate is extremely '
    return text

def person_description(data_p, data, questions):
    data_pz = z_score(data,data_p) 
   
    
    text = []
    
    for i in range(0,5):
        
        # extraversion
        if i == 0:
            cat_0 = 'solitary and reserved. '
            cat_1 = 'outgoing and energetic. '
        
            if data_pz['extraversion_zscore'].values > 0:
                text_t = categorie_description(data_pz['extraversion_zscore'].values) + cat_1
                if data_pz['extraversion_zscore'].values > 1:
                    index_max = data_pz.iloc[0,0:10].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0]+'. '
                    text_t = text_t + text_2
            else:
                text_t = categorie_description(data_pz['extraversion_zscore'].values) + cat_0
                if data_pz['extraversion_zscore'].values < -1:
                    index_min = data_pz.iloc[0,0:10].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0]+'. '
                    text_t = text_t + text_2
            text.append(text_t)
                
        # neurotiscism
        if i == 1:
            cat_0 = 'resilient and confident. '
            cat_1 = 'sensitive and nervous. '
            
            if data_pz['neurotiscism_zscore'].values > 0:
                text_t = categorie_description(data_pz['neurotiscism_zscore'].values) + cat_1  \
                + 'The candidate tends to feel more negative emotions, anxiety. '
                if data_pz['neurotiscism_zscore'].values > 1:
                    index_max = data_pz.iloc[0,10:20].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0]+'. '
                    text_t = text_t + text_2
                
            else:
                text_t = categorie_description(data_pz['neurotiscism_zscore'].values) + cat_0  \
                + 'The candidate tends to feel less negative emotions, anxiety. '
                if data_pz['neurotiscism_zscore'].values < -1:
                    index_min = data_pz.iloc[0,10:20].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0]+'. '
                    text_t = text_t + text_2
            text.append(text_t)
            
        # agreeableness        
        if i == 2:
            cat_0 = 'critical and rational. '
            cat_1 = 'friendly and compassionate. '
            
            if data_pz['agreeableness_zscore'].values > 0:
                text_t = categorie_description(data_pz['agreeableness_zscore'].values) + cat_1  \
                + 'The candidate tends to be more cooperative, polite, kind and friendly. '
                if data_pz['agreeableness_zscore'].values > 1:
                    index_max = data_pz.iloc[0,20:30].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                    text_t = text_t + text_2

            else:
                text_t = categorie_description(data_pz['agreeableness_zscore'].values) + cat_0  \
                + 'The candidate tends to be less cooperative, polite, kind and friendly. '
                if data_pz['agreeableness_zscore'].values < -1:
                    index_min = data_pz.iloc[0,20:30].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                    text_t = text_t + text_2
            text.append(text_t)
            
        # conscientiousness
        if i == 3:  
            cat_0 = 'extravagant and careless. '
            cat_1 = 'efficient and organized. '
            
            if data_pz['conscientiousness_zscore'].values > 0:
                text_t = categorie_description(data_pz['conscientiousness_zscore'].values) + cat_1  \
                + 'The candidate tends to be more careful or diligent. '
                if data_pz['conscientiousness_zscore'].values > 1:
                    index_max = data_pz.iloc[0,30:40].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                    text_t = text_t + text_2
            else:
                text_t = categorie_description(data_pz['conscientiousness_zscore'].values) + cat_0  \
                + 'The candidate tends to be less careful or diligent. '
                if data_pz['conscientiousness_zscore'].values < -1:
                    index_min = data_pz.iloc[0,30:40].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                    text_t = text_t + text_2
            text.append(text_t)
        
        # openness
        if i == 4:
            cat_0 = 'consistent and cautious. '
            cat_1 = 'inventive and curious. '
            
            if data_pz['openness_zscore'].values > 0:
                text_t = categorie_description(data_pz['openness_zscore'].values) + cat_1  \
                + 'The candidate tends to be more open. '
                if data_pz['openness_zscore'].values > 1:
                    index_max = data_pz.iloc[0,40:50].idxmax()
                    text_2 = 'In particular they said that ' + questions[index_max][0] +'. '
                    text_t = text_t + text_2
            else:
                text_t = categorie_description(data_pz['openness_zscore'].values) + cat_0  \
                + 'The candidate tends to be less open. '
                if data_pz['openness_zscore'].values < -1:
                    index_min = data_pz.iloc[0,40:50].idxmin()
                    text_2 = 'In particular they said that ' + questions[index_min][0] +'. '
                    text_t = text_t + text_2
            text.append(text_t)
        i+=1
        
    
    return text