# Model card for World Value Survey Data Wordalisation and Chatbot

In a nutshell, our app is a retrieval augmented chatbot for making reports about countries based on data derived from the [World Value Survey](www.worldvaluessurvey.org) (WVS). The app is implemented within the [TwelveGPT Education framework](https://github.com/soccermatics/twelve-gpt-educational) and is intended as an illustration of the _wordalisation_ method. It is thus intended as an example to help others build similar tools. The wordalisations are created by comparing a country's score across six social factors to its relative position within the distribution of scores across all countries. The chosen social factors, as well as the WVS, are discussed in the [Dataset](#dataset) section. 

This work is a derivative of the full [Twelve GPT product](https://twelve.football). The original design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens, with modification made by Beimnet Zenebe and Amy Rouillard to adapt it to the WVS use-case.

This model card is based on the [model cards paper](https://arxiv.org/abs/1810.03993) and is adapted specifically to Wordalisation applications as detailed in [Representing data in words](https://arxiv.org/). We also provide this model card as an example of good practice for describing wordalisations.

Jump to section:

- [Intended use](#intended-use)
- [Factors](#factors)
- [Dataset](#dataset)
- [Model](#model)
- [Evaluation](#evaluation)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Intended use

The _primary use case_ of this wordalisation is educational. It shows how to convert a python pandas DataFrame of statistics about countries into a text that discusses a chosen country. The statistics relate to various social and political values, however we note that the results should be understood in the context of the WVS and the research in which the factors were derived. The purpose of the wordalisation and chat functionality is to use the capabilities of an large language model (LLM) to turn the raw statistics into text that is more digestible for a human reader. Our goal was not to investigate the validity of the WVS study or the presented social factors.

This chatbot cannot be used for insight generation purposes, i.e. data analysis, firstly because we do not guarantee the quality of the data and because functionality is limited. Data analysis is thus _out of scope_. Use of the chat for queries not relating to the WVS data at hand is also out of scope.

We would also strongly oppose the generalization or stereotyping of any group of people and emphasize that this chatbot cannot and should not be used to represent the values any country or its population.

## Factors

The World Value Survey data and derived factors, discussed in [Dataset](#dataset), relate to 66 countries that took part in the WVS "wave 7" 2017-2022 survey. We would like to state that any reports or chats about countries not included in the survey are not guaranteed to hold any merit. We also note that the participants of the "wave 7" survey constitute only a small sample of the population of each countries, see Figures 1 and 2. Therefore, the values and statements presented in the app should not be considered representative of the entire population of any given country.


![WVS coverage](https://github.com/soccermatics/twelve-gpt-educational/blob/dev/model%20cards/imgs/sample_size_map.png)

![WVS coverage](model cards/imgs/sample_size_map.png)


Figure 1: World Value Survey data collection distribution. The color-scale indicates the number of participants in the survey from each country.


![WVS coverage percentage](https://github.com/soccermatics/twelve-gpt-educational/blob/dev/model%20cards/imgs/sample_size_percentage_map.png)

![WVS coverage percentage](model cards/imgs/sample_size_percentage_map.png)


Figure 2: World Value Survey data collection distribution as a percentage of the coutries population. The populations were taken from the [United Nations Department of Economic and Social Affairs](https://population.un.org/wpp/Download/Standard/CSV/) data on world population prospects in 2017 (the first year of "wave 7"). The color-scale indicates the log of the percentage of the population of each country that participated.


## Dataset

The data used in this project was constructed from the [World Value Survey Wave 7 (2017-2022)](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp). The data consists of coded answers to a questionnaire which can be found at the same link. The WVS questionnaire was taken by participants from 62 countries and 4 regions (Hong Kong, Macao, Northern Ireland and Puerto Rico) with sample sizes varying from 447 in Northern Ireland to 4018 in Canada. It is clear from the map shown in Figure 1 that the data is not uniformly distributed across the globe, with the number of sampled African and European countries being especially low.

From the raw survey data six social _factors_ were constructed. These are summerised and described in Table 1 and were calculated according to Ingelhart (2005) [1] and Allison (2021) [2], for factors indicated by $^1$ and $^2$ respectively. These factors were derived using factor analysis, which is a statistical method used to describe variability among observed, correlated variables in terms of a potentially lower number of unobserved variables called factors. 

Table 1: Description of factors
| Factor | Description |
| --- |  --- |
|Traditional vs Secular Values<sup>1</sup>|  Traditional values emphasize religion, authority, and a nationalistic outlook, while rejecting abortion. Secular values place less emphasis on religion, authority, independence and perseverance in children and are more accepting of abortion. |
|Survival vs Self-expression Values<sup>1</sup> |  Survival values emphasize economic and physical security, with a focus on national identity and lower levels of trust and tolerance of homosexuality. Self-expression values prioritize leisure over work, tolerance of homosexuality, freedom, and political participation. Levels of trust are higher as well as ratings of happiness and life-satisfaction. |
|Neutrality <sup>2</sup>| Neutrality measures a lack of engagement in civic, political, or social organizations. High-scoring countries having little participation in consumer organisations, charitable or humanitarian organizations, professional organizations or self-help or mutual aid groups. |
|Fairness <sup>2</sup>| Fairness is measured by attitudes toward whether the actions of stealing, bribery, cheating on taxes, and violence are ever justifiable. A high score in fairness is associated with these actions never being justifiable. |
|Skepticism <sup>2</sup>| Skepticism represents distrust in government, civil services, political parties, and the justice system or courts. A high score in skepticism indicates a low confidence in these institutions.|
|Societal Tranquility<sup>2</sup> |  Societal Tranquillity measures the level of worry felt about war, civil war, terrorism, and access to good education. Low scores in societal tranquillity indicate worry about these issues. |

The social factors we present in the chat app are constructed such that they are a linear combinations of the observed variables. The weights and questions used to construct these factors are summerised in Table 2. The data was also normalised across answer codes, such that each answer is rescaled to a number between -1 and 1. This is to take into account the highly variable answer scales. Finaly, we also normalise each social factor by the sum of the weights to guarantee a ranges of -1 to 1. Our preprocessing pipeline is available as a [python notebook](https://github.com/soccermatics/twelve-gpt-educational/blob/dev/data/wvs/preprocessing/generate_data.ipynb), and the relevant data should be downloaded from the [official WVS site](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp). 

The six social factors are intended to aggregate the answers to several survey questions in order to provide general insight into a country's social values. For example, is a country more traditional or secular? Do they believe that acts like corruption or stealing are ever justified? How active is the population in civic organizations? We point out that the number of questions aggregated is limited and it is unlikely that they can capture the full complexity of a country's values. See [Ethical considerations](#ethical-considerations) for further discussion.

Table 2: Factors and their construction
| Factor | Questions (codes and short description) | Weights | 
| --- | --- | --- | 
|Traditional vs Secular Values | Q164: How important is God in your life? </br> Q8: How important is independence in children to you? </br> Q14: How important is determination perseverance in children to you? </br> Q15: How important is religious faith in children to you? </br> Q17: How important is obedience in children to you? </br> Q184: Abortion. Is it ever justifiable? </br> Q254: How proud are you to be of the nationality of this country? </br> Q45: In the future, would it be a good thing if we had greater respect for authority? | 0.70 </br> 0.61 </br> 0.61 </br> 0.61 </br>0.61 </br>0.61 </br>0.60 </br>0.51 |
|Survival vs Self-expression Values |Q3: How important is leisure time to you? </br> Q5: How important is work to you? </br> Q40: Do you think that work is a duty towards society? </br> Q41: Do you think work should always come first, even if it means no spare time? </br> Q43: In the future, would it be a good thing if less importance is placed on work. </br> Q131: Could you tell me how secure do you feel these days in your neighborhood? </br> Q142: To what degree are you worried about losing your job or not finding a job. </br> Q150: Most people consider both freedom and security to be important, which do you think is more important? </br> Q46: Do you feel happy? </br>    Q49: How satisfied are you with your life? </br>    Q22: Would you mind having people who are homosexuals as neighbors? </br>   Q182: Homosexuality. Is it ever justifiable? </br> Q209: Have you, might you or would you never, under any circumstances: Signing a petition. </br> Q218 : Have you, might you or would you never, under any circumstances: Signing an electronic (online) petition. </br> Q57: Can most people be trusted? </br>    Q58: How much do you trust your family? </br>    Q59: How much do you trust your neighborhood? </br>    Q60: How much do you trust people you know personally? </br>    Q61: How much do you trust people you meet for the first time? </br>    Q62: How much do you trust people of another religion? </br>    Q63: How much do you trust people of another nationality?| 0.59</br> 0.59 </br>0.59</br> 0.59</br> 0.59 </br>0.59 </br>0.59 </br>0.59 </br>0.59</br>0.59 </br>0.58 </br>0.58 </br>0.54 </br>0.54 </br>0.44 </br>0.44 </br>0.44 </br>0.44 </br>0.44 </br>0.44 </br>0.44|
Neutrality | Q102R: Are you a member of a consumer organization? </br> Q101R: Are you a member of a charitable or humanitarian organization? </br> Q100R: Are you a member of a professional organization? </br> Q103R: Are you a member of a self-help or mutual aid group? | 0.76</br> 0.74 </br>0.72 </br>0.76|
Fairness |Q179: Stealing property. Is it ever justifiable? </br> Q181: Someone accepting a bribe in the course of their duties. Is it ever justifiable? </br> Q180: Cheating on taxes if you have a chance. Is it ever justifiable? </br> Q191: Violence against other people. Is it ever justifiable? </br> Q189 : For a man to beat his wife. Is it ever justifiable?| 0.77</br>  0.74 </br>0.70 </br>0.63 </br>0.58 |
Skepticism |Q73: How much confidence do you have in the parliament? </br> Q71: How much confidence do you have in the government? </br> Q74: How much confidence do you have in the civil services  </br> Q72: How much confidence do you have in political parties? </br> Q70: How much confidence do you have in the justice system/courts?| 0.77 </br> 0.74 </br>0.70 </br>0.63</br> 0.58|
Societal Tranquility | Q148: To what degree are you worried about the following situations? A civil war </br> Q147: To what degree are you worried about the following situations? A terrorist attack </br> Q146: To what degree are you worried about the following situations? A war involving my country </br> Q143: To what degree are you worried about the following situations? Not being able to give one's children a good education | 0.82</br> 0.80</br> 0.80</br> 0.49|




In addition to the wordalisation of these factors we provide question and answer pairs to the chatbot. The first set of question and answer pairs were derived from the texts [1] and [2] as well as the World Value Survey website [3]. They are intended to contextualize the data and factors and can be found in the [WVS Qualities](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/describe/WVS_qualities.xlsx) spreadsheet. The descriptions provided in Table 1 are extracted from this spreadsheet. 

In addition, we provide question and answer pairs that are intend to provide good examples of how that chat bot should discuss the chosen country. These in-context-learning examples and can be found [here](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/gpt_examples/WVS_examples.xlsx). The usage of the above mentioned question-answer pairs is detailed in the [Normative model](#normative-model) section.

## Model

### Quantitative model

The model applies a mapping to the countries in the dataset along different value scales, see Table 2. For each metric, a z-score is calculated by subtracting the mean and dividing by the standard deviation over all countries' metric values in the dataset. The countries are then displayed in a distribution plot with the selected country highlighted, representing leaning on different value scales.

### Normative model

A prompt is constructed in several parts (_tell it who it is_, _tell it what it knows_, _tell it how to answer_, _tell it what data to use_) in order to provide relevant context to the LLM. The prompt to _tell it who it is_ identifies a human role for the chatbot as a "Data Analyst". The user-assistant pairs in the stage of _tell it what it knows_ describe the general context of the chat domain and how the selected values can be [interpreted](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/describe/WVS_qualities.xlsx) according to [1] and [2]. These descriptions outline the meaning of the values and take into account the question on which they are based. The penultimate prompt to _tell it how to answer_ provides [examples of expected outputs](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/gpt_examples/WVS_examples.xlsx) given data about a certain country. The final prompt is to _tell it what data to use_ and is therefore customised for the selected country.

At the stage of _tell it what data to use_, we apply a model that maps the z-scores for each metric associated with the selected country onto a text phrase, for example "above averagely neutral". The distribution of countries along the value scale represents the leaning of values towards one value or the other compared to all other countries. It is important to note that, for example, the global average for the "neutrality" factor score is positive, that is on average people across the global tend not to actively participate in civil society and community organizations. Our wordalisation only reflects whether a country is more or less neural compared to _the average_. For example, a country with a z-score of more than -2 on the "neutrality" score would be considered to "far bellow averagely neutral", even though participants may have reported on average only some limited participation in civil society and community organizations. To provide clarify to the user we also include the factor score in the visualization, where a positive or negative score points to the leaning of the country on the value scale, not compared to the global average.

We would also like to emphasis that factor scores and z-score values are devoid of any absolute meaning, and do not reflect any notion of better or worse. In other words, we do not consider that any particular result to be more or less favorable in a moral sense.


#### Implementation  

We use the following function to translate z-scores into evaluation words:

```python
def describe(thresholds, words, value):
    assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"
    i = 0
    while i < len(thresholds) and value < thresholds[i]:
        i += 1

    return words[i]
```

We then provide different sets of words and thresholds to use for each metric being discussed. For example, for "Traditional vs Secular Values" we might use the following:

```python
thresholds = [2, 1, -1, -2]
words = [
        "be far above averagely neutral",
        "be above averagely neutral",
        "be averagely neutral",
        "be below averagely neutral",
        "be far below averagely neutrality"
    ]
```

which fits into the template:

```python
text = f"{country.name.capitalize()} was found to {words} compared to other countries in the same wave. "

```

#### Additional context

In addition to the factor z-score wordalisation, we also provided information about certain specific questions in the WVS survey. First we decide whether or not to provide additional information by checking if the magnitude of z-score for a given factor is above a chosen threshold (in our case 1). If so, then we provide a sentence that reports on the average answer to a question that contributed most to the factor. For increased interpetiablity we select a question with a positive (negative) contribution of the z-score is above (bellow) average. For example, if a given country has a z-score for "Traditional vs Secular Values" greater than 1, then we might provide a sentence that reports "In response to the question 'How important is God in your life?', on average participants indicated God to be 'very important' in their life". Similarly, if a given country has a z-score for the "Societal Tranquillity" less than -1, then we might provide a sentence that reports "In response to the question 'To what degree are you worried about the following situations? A civil war', on average participants indicated that they worry 'very much'."

### Language model

Our implementation supports both GPT4o and ChatGPT and related APIs, as well as Gemini API. Since these language model's training data included text about these countries, this knowledge will likely effect both the initial response and subsequent queries within the same chat. This means that some aspects of the answers may come from data external to that in the provided dataframe. 

## Evaluation

Qualitative evaluation during development allowed us to refine the prompt to ensure that the chatbot was providing satisfactory responses. Qualitative analysis is also import for identifying possible issues and to establish the limitations and usefulness of the application. 

In addition we also performed the quantitative analysis described below to evaluate whether the model was providing factual and relevant information about the dataset. Figure 3 shows the results of this analysis. 


![Accuracy](https://github.com/soccermatics/twelve-gpt-educational/blob/dev/model%20cards/imgs/accuracy_country.png)

![Accuracy](model cards/imgs/accuracy_country.png)

Figure 3: Comparison of the class labels generated by the normative model with classes reconstructed from the wordalisations. Multiple wordalisations were generated for each data point, so that at least 10 valid reconstructions per data point were found, and the mean accuracy is taken over all wordalistations. We compare the accuracy of the model for two different prompts, one in which data in the form of synthetic texts was given (purple) and in the other the data
was omitted (red). The dashed line indicates the expected accuracy if the class labels were randomly chosen according to a uniform probability distribution and lie at an accuracy of $\frac{1}{5}$.

For each data point in our datasets we generate a Wordalisation using a [prompt template](https://github.com/soccermatics/twelve-gpt-educational/tree/dev/evaluation/prompts) almost identical to the prompt used in the application. For comparison, we also generated Wordalisations using a version of the prompt that did not contained the relevant synthetic text generated by the normative model. To discourage the LLM from declining to respond we added the sentence `If no data is provided answer anyway, using your prior statistical knowledge.' and modified one of the in-context learning examples by removing synthetic text from the user prompt while leaving the Wordalisation (response) unchanged. For consistency, the same prompt template was used both when data was and was not provided. To take into account random variations in the Wordalisations due to the stochastic nature of the LLM, we passed each prompt to the LLM multiple times to generate a set of Wordalisations for evaluation. 

In a new chat instance, we prompt the LLM to reconstruct the data from a given Wordalisation in the form of a json file. We then compare the `true' class according to the normative model with the reconstructed data class to measure how faithfully the Wordalisation represents on the given data. When the generated json file could not be parsed, the data was discarded.

This approach has some weaknesses, including relaying on the LLM to generate accurate reconstructions. In cases where the Wordalisations are more formulaic, this type of evaluation work can well, as was the case here. However, in cases where the texts are more engaging the reconstruction is more nuanced, see the football scout application.




## Ethical considerations

The World Value Survey is based on questionnaires filled out by a small sample of individuals from different countries around the world. In particular, we used data from the 7th wave of the survey which took place from 2017-2022. Samples are relatively small compared to the populations of the countries, see Figure 2. While data was collected from 66 regions, the data is not evenly distributed across the globe, with coverage in Africa and Europe being particularly sparse. Special status was also given to the regions of Hong Kong, Macao, Puerto Rico and Northern Ireland, which are not independent countries.

While some effort was made to accommodate multiple languages, questionnaires where provided in a limited number of languages for each country, possibly preventing some groups of the population from participating. The data is also based on self-reported answers to the questionnaires, which may be influenced by the social context in which the questionnaire was given.

In the context of the factors mentioned in [Dataset](#dataset), the data can only give a rough indication of the attitudes of a population, during the period 2017-2022. Therefore, the summaries generated by the [Normative model](#normative-model) may contain out of date information and should not be considered a reflection of the beliefs or attitudes of any given individual.

We also would like to note that it is an open question as to whether the derived factors summerised in Tables 1 and 2 give any meaningful insights and we would urge users to consider them within the research context in which they were derived, see [1] and [2] and the World Value Survey website [3]. 

## Caveats and recommendations

We have no further caveats and recommendations.

## References

[1] Inglehart, R., 2005. Christian Welzel Modernization, Cultural Change, and Democracy The Human Development Sequence. Cambridge: Cambridge university press.

[2] Allison, L., Wang, C. and Kaminsky, J., 2021. Religiosity, neutrality, fairness, skepticism, and societal tranquility: A data science analysis of the World Values Survey. Plos one, 16(1), p.e0245231.

[3] “WVS Database.” Accessed October 23, 2024. https://www.worldvaluessurvey.org/wvs.jsp.
