# Model card for World Value Survey metric Wordalisation

The WVS chatbot is implemented within the [TwelveGPT Education framework](https://github.com/soccermatics/twelve-gpt-educational) and
is intended as an illustration of the *wordalisation* method. It is thus intended as an example to help others build similar tools. The wordalisations are constructed using various social metrics derived from data collected in the [World Value Survey](www.worldvaluessurvey.org) (WVS). These social metrics as well as the WVS are discussed in the [Datasets](#datasets) section. This work is a derivative of the full [Twelve GPT product](https://twelve.football). The original design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens, with modification made by Beimnet Zenebe and Amy Rouillard to adapt it to the WVS use-case.

This model card is based on the [model cards paper](https://arxiv.org/abs/1810.03993) and is adapted specifically to Wordalisation applications as detailed in [Representing data in words](publication here). We also provide this model card as an example of good practice for describing wordalisations.

Jump to section:

- [Intended use](#intended-use)
- [Factors](#factors)
- [Datasets](#dataset)
- [Model](#model)
- [Evaluation](#evaluation)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Intended use

The _primary use case_ of this wordalisation is eductional.
It shows how to convert a dataframe of statistics about countries into a text that discusses a chosen country. The statistics relate to various social and political values, however we note that the results should be understood in the context of the WVS and the research in which the metrics were derived. The purpose of the wordalisation and chat functionality is to use the capacities of an LLM to make the statistics more digestible for a human reader, rather than to investigate the validity of the WVS study. 

This version cannot be used for insight generation purposes, i.e. data analysis, firstly because we do not guarantee the quality of the data and because functionality is limited.
Data analysis is thus _out of scope_. Use of the chat for queries not relating to the data at hand is also _out of scope_.

## Factors

The World Value Survey metric Wordalisation relates to 66 countries that took part in the 2017-2022 survey. We would like to state that any conversations about countries not included in the survey are not guaranteed to hold any merit. We would also like note that the participants of these surveys are only a small sample of the population with the biggest sample from a given country being 4000. Therefore, the values presented here cannot be representative of a an entire population of a country.

## Datasets

The data used in this project was constructed from the [World Value Survey Wave 7 (2017-2022)](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp).
The data consists of coded answers to a questionnaire, which were taken by participants from 66 countries with sample sizes varying from 447 in Northern Ireland to 3200 in Indonesia.
From this raw data we constructed 7 metrics or qualities:

- "Traditional vs Secular Values"$^1$
- "Survival vs Self-expression Values"$^1$
- "Neutrality"$^2$
- "Fairness"$^2$
- "Skepticism"$^2$
- "Societal Tranquility"$^2$

These metrics were calculated according to Ingelhart (2005)[1] and Allison (2021)[2], for metrics indicated by $^1$ and $^2$ respectively.
We note that because of the coding of answers to the questionnaire some considerable _preprocessing_ was necessary to construct these metrics. Our preprocessing pipeline is available as a [Github repository](https://github.com/BeimnetGirma/wvs-data).

The above metrics are intended to aggregate the answers to several questions in order to provide a more general insight into a country's values. For example, is a country more traditional or secular?

In addition to the wordalisation of these metrics we provide question and answer pairs as to the chatbot. These are intended to contextualize the data and can be found in the [WVS_qualities](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/describe/WVS_qualities.xlsx) spreadsheet. Further, we provide question and answer pairs that are intend to be good examples of how that chat bot should discuss the given country. These are intended as in-context-learning examples and can be found [here](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/gpt_examples/WVS_examples.xlsx). The use of these question-answer pairs is detailed in the [Normative model](#normative-model) section.

## Model

### Quantitative model

The model applies a mapping on the countries in the dataset along different value scales. For each value metric, a z-score is calculated by subtracting the mean and dividing by the standard deviation over all countries' metric values in the dataset. The countries are then displayed in a distribution plot with the selected country highlighted, representing leaning on different value scales.

### Normative model


A prompt is constructed in several parts in order to provide relevant context to the LLM. The prompt to _tell it who it is_ identifies a human role for the chat bot as a "Data Analyst". The user-assistant pairs in the stage of _tell it what it knows_ describe how the selected values can be [interpretted in the social sciences](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/describe/WVS_qualities.xlsx). These descriptions outline the meaning of the values. And penultimate prompt to _tell it how to answer_ provides [examples of outputs](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/gpt_examples/WVS_examples.xlsx). The final prompt is to _tell it what data to use_ for the selected country.


At the stage of _tell it what data to use_, we apply a model that maps the values of a country onto different value scales based on the z-scores. The distribution of countries along the value scale represents the leaning of values towards one value or the other. For example, a country with a z-score of more than 3 would be considered to "have extremely secular values". The distribution (or the values associated with it) are devoid of any notion of ranking based on the value scores of countries. We use the following function to translate z-scores into evaluation words:

```python
def describe(thresholds, words, value):
    assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"
    i = 0
    while i < len(thresholds) and value < thresholds[i]:
        i += 1

    return words[i]
```

We then provide different sets of words and thresholds to use for each value being discusses. For example, for "Traditional vs Secular Values" we might use the following:

```python
thresholds = [2.5, 1.5, 0.5, -0.5, -1.5, -2.5]
words = [
        "have extremely secular values",
        "have very secular values",
        "have above averagely secular values",
        "have neither strongly traditional nor strongly secular values",
        "have above averagely traditional values",
        "have very traditional values",
        "have extremely traditional values"
    ]
```
which fits into the template: 
```python
text = f"{country.name.capitalize()} was found to {words} compared to other countries in the same survey. "

```

### Language model

Our implementation supports both GPT4o and ChatGPT and related APIs, as well as Gemini API. Since these language model's training data included text about these countries, this knowledge will likely effect both the initial response and subsequent queries within the same chat. This means that some aspects of the answers may come from data external to that in the provided dataframe.

## Evaluation 

**Under construction**

Some systematic _quantitative analyses_ has been carried out on this wordalisation. 
- Sentiment analysis. How does the prompt and wordalisation affect the sentiment of the chatbot when discussing a country?

Ideally, this wordalisation should be subjected to further rigorous _qualitative test_ of:

- Reliability of information. Are there factual errors in the text?
- Biases
  - Does the WVS chatbot make bias statements based on 'knowledge' not present in the wordalisation?
  - Does the wordalisation relay on socitial stereotypes of countries to generate the texts?
  - Does the wordalisation introduce notions of ranking to countries' values.

## Ethical considerations

The World Value Survey is based on questionnaires filled out by individuals from different countries. In particular was use data generate during wave 7 which took place from 2017 to 2022. Samples are relatively small compared to the populations of the countries.
Further, the questionnaires where given in a limited number of languages for each country, possibly preventing some groups of the population from participation. The data is also based on self-reported answers to the questionnaires, which may be influenced by the social context in which the questionnaire was given.

In the context of the factors mentioned the above, the data can only give a rough indication of the attitudes of a population, during the period 2017-2022. Therefore, the summaries generated by the [Normative model](#normative-model) may contain out of date information and should not be considered a reflection of the beliefs or attitudes of any given individual.

We also would like to note that it is an open question as to whether the derived metrics give any meaningful insights and we would urge users to consider them within the research context in which they were derived, see [Datasets](#datasets).

## Caveats and recommendations

We have no further caveats and recommendations.

## References

[1] Ingelhart, R. and Welzel, C., 2005. Modernization, cultural change, and democracy: The human development sequence.

[2] Allison, L., Wang, C. and Kaminsky, J., 2021. Religiosity, neutrality, fairness, skepticism, and societal tranquility: A data science analysis of the World Values Survey. Plos one, 16(1), p.e0245231.
