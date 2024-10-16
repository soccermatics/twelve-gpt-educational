# Model card for World Value Survey metric Wordalisation



The WVS chatbot is implemented within the [TwelveGPT Education framework](https://github.com/soccermatics/twelve-gpt-educational) and
is intended as an illustration of the methods. It is thus intended as an example to help others building wordalisations. The wordalisations are constructed using various social metrics derived from data collected in the [World Value Survey](www.worldvaluessurvey.org) (WVS). These social metrics as well as the WVS are discussed in the [Datasets] section. This work is a derivative of the full [Twelve GPT product](https://twelve.football). The original design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens, with modification made by Beimnet Girma and Amy Rouillard to adapt it to the WVS use-case.

This model card is based on the [model cards paper](https://arxiv.org/abs/1810.03993) and is adapted specifically to Wordalisation applications as detailed in [Representing data in words](publication here). We also provide this model card as an example of 
good practice for describing wordalisations.

Jump to section:

- [Intended use](#intended-use)
- [Factors](#factors)
- [Datasets](#dataset)
- [Model](#model)
- [Evaluation](#evaluation)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Intended use

The *primary use case* of this wordalisation is eductional. 
It shows how to convert a dataframe of statistics about a countries into a text about a chosen country.
This version cannot be used for insight generation purposes, i.e. data analysis, firstly because we do not guarantee the quality of the data and because the functionality is limited. 
Data analysis is thus *out of scope*. Use of the chat for queries not relating to the data at hand is also *out of scope*. 

## Factors



## Datasets

The data used in this project was constructed from the [World Value Survey Wave 7 (2017-2022)](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp).
The data consists of coded answers to a questionnaire, which was taken by participants from 66 countries with sample sizes varying from 447 in Northern Ireland to 3200 in Indonesia.
From this raw data we constructed 7 metrics or qualities:
* "Traditional vs Secular Values"$^1$ 
* "Survival vs Self-expression Values"$^1$
* "Neutrality"$^2$
* "Fairness"$^2$
* "Skepticism"$^2$
* "Societal Tranquility"$^2$

These metrics were calculated according to []()$^1$ and [Allison (2021)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0245231)$^2$.
We note that because of the coding of answers to the questionnaire some considerable *preprocessing* was necessary to construct these metrics. Our preprocessing pipeline is available as a [Github repository](https://github.com/BeimnetGirma/wvs-data). 

The above metrics are intended to aggregate the answers to several questions in order to provide a more general insight into a country's values. For example, is a country more traditional or secular? 

In addition to the metrics we also provide question and answer pairs for the chatbot. These are intended to provide insight into the data and the metrics which can be found in the [WVS_qualities](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/describe/WVS_qualities.xlsx) spreadsheet.

## Model

### Quantitative model

The model applies a ranking on the football players in the dataset based on the z-scores. For each metric a z-scoure is calculated by 
subtracting the mean and dividing by the standard deviation over all players in the dataset. The players are then displayed in a distribution
plot with the selected player highlighted. 

### Normative model

The model applies a ranking on the football players in the dataset based on the z-scores. So players further to the left are described as worse on the metric and those to the right are described as better. These are socially agreed upon (in football and sporting situations) ways of viewing players, but nevertheless applies a norm whereby players who have higher values of these metrics are ranked more highly. 

The prompt to *tell it who it is* identifies a human role for the wordalisation as a "UK based scout". The user-assistant pairs in the stage of *tell it what it knows* describe how the data metrics can be [interpretted in footballing terms](https://github.com/soccermatics/twelve-gpt-educational/blob/main/data/describe/Forward.xlsx). These descriptions outline the meaning of the metrics.

In the text which is generated at the stage of *tell it what data to use* we use the following function to translate z-scores to evaluation words:
```python
def describe_level(value):
    thresholds = [1.5, 1, 0.5, -0.5, -1]
    words = ["outstanding", "excellent", "good", "average", "below average", "poor"]
    return describe(thresholds, words, value)
```
There are three positive terms and two negative terms in the words used, with no negative equivalent of "outstanding" defined. 

In the *tell it how to answer* step, two examples were created, based on data from Pedro and Giroud. This somewhat compromises how the texts for these two players are constructed. These examples and the prompt used emphasise highlighting positive and negative aspects of the players' performance, and ignoring aspects which are average or typical.

### Language model

The wordalisation supports both GPT4o and ChatGPT and related APIs, as well as Gemini API. Since these language model's training data includes articles written about the players described here, this knowledge will seem into both the answers generated by the wordalisation and the chat. Some aspects of the answers will come from data external to that in the provided dataframe.

## Evaluation

No systematic *quantitative analyses* have been carried out on this wordalisation. Ideally, this wordalisation should be subjected to a rigorous *qualitative test* of:
- Reliability of information. Are there factual errors in the text?
- Biases. Does the WVS chatbot make bias statements based on 'knowledge' not present in the wordalisation?

## Ethical considerations

The World Value Survey is based on questionnaires filled out by individuals from different countries. The sets of individuals are relatively small compared to the populations of the countries. 
Further, the questionnaires where given in a limited number of languages for each country, potentially excluding some groups of the population.
Therefore, the data can only give a rough indication of the attitudes of a population. The summaries generated by no means reflect the beliefs and attitudes of any given individual.
Furthermore, it is debatable whether the derived metrics give any accurate insight and we would urge users to consider them within the research context in which they were derived, see [Datasets](#datasets).


## Caveats and recommendations

We have no further caveats and recommendations.

