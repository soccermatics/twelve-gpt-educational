# Model card for World Value Survey metric Wordalisation

The WVS chatbot is implemented within the [TwelveGPT Education framework](https://github.com/soccermatics/twelve-gpt-educational) and
is intended as an illustration of the methods. It is thus intended as an example to help others building wordalisations. The wordalisations are constructed using various social metrics derived from data collected in the [World Value Survey](www.worldvaluessurvey.org) (WVS). These social metrics as well as the WVS are discussed in the [Datasets](#datasets) section. This work is a derivative of the full [Twelve GPT product](https://twelve.football). The original design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens, with modification made by Beimnet Zenebe and Amy Rouillard to adapt it to the WVS use-case.

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

The _primary use case_ of this wordalisation is eductional.
It shows how to convert a dataframe of statistics about a countries into a text about a chosen country.
This version cannot be used for insight generation purposes, i.e. data analysis, firstly because we do not guarantee the quality of the data and because the functionality is limited.
Data analysis is thus _out of scope_. Use of the chat for queries not relating to the data at hand is also _out of scope_.

## Factors

## Datasets

The data used in this project was constructed from the [World Value Survey Wave 7 (2017-2022)](https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp).
The data consists of coded answers to a questionnaire, which was taken by participants from 66 countries with sample sizes varying from 447 in Northern Ireland to 3200 in Indonesia.
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

In addition to the metrics we also provide question and answer pairs for the chatbot. These are intended to provide insight into the data and can be found in the [WVS_qualities](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/describe/WVS_qualities.xlsx) spreadsheet.

## Model

### Quantitative model

The model applies a mapping on the countries in the dataset along differnt value scales. For each value metric, a z-score is calculated by subtracting the mean and dividing by the standard deviation over all countries' metric values in the dataset. The countries are then displayed in a distribution plot with the selected country highlighted, representing leaning on different value scales.

### Normative model

The model applies a mapping of country values on different value scales based on the z-scores. The disribution of countries along the value scale represents the leaning of values towards one dimension or the other. The disribution (or the values associated with it) are devoid of any notion of ranking based on the value scores of countries.

The prompt to _tell it who it is_ identifies a human role for the wordalisation as a "Data Analyst". The user-assistant pairs in the stage of _tell it what it knows_ describe how the selected values can be [interpretted in the social sciences](https://github.com/soccermatics/twelve-gpt-educational/blob/wvs_chat/data/describe/WVS_qualities.xlsx). These descriptions outline the meaning of the values.

In the text which is generated at the stage of _tell it what data to use_ we use the following function to translate z-scores to evaluation words:

```python
def describe(thresholds, words, value):
    assert len(words) == len(thresholds) + 1, "Issue with thresholds and words"
    i = 0
    while i < len(thresholds) and value < thresholds[i]:
        i += 1

    return words[i]
```

We then provide different sets of words and thresholds to use for each value being discusses. An example of these values and threshold are provided below.

```python
thresholds_dict = dict((metric,[1.5,1, 0.5,-0.5,-1,])for metric in metrics)

description_dict = {
    "Traditional vs Secular Values": [
        "extremely secular",
        "very secular",
        "above averagely secular",
        "neither traditional nor secular",
        "above averagely traditional",
        "very traditional",
    ],
    "Neutrality": [
        "extremely high",
        "very high",
        "above average",
        "average",
        "below average",
        "very low",
    ],
```

There are three positive terms and two negative terms in the words used, with no negative equivalent of "extremely [value_name]" defined.

In the _tell it how to answer_ step, two examples were created, based on data from China and Iran.

### Language model

The wordalisation supports both GPT4o and ChatGPT and related APIs, as well as Gemini API. Since these language model's training data included text about these countries, this knowledge will seem into both the answers generated by the wordalisation and the chat. Some aspects of the answers will come from data external to that in the provided dataframe.

## Evaluation

No systematic _quantitative analyses_ have been carried out on this wordalisation. Ideally, this wordalisation should be subjected to a rigorous _qualitative test_ of:

- Reliability of information. Are there factual errors in the text?
- Biases
  - Does the WVS chatbot make bias statements based on 'knowledge' not present in the wordalisation?
  - Does the wordalisation relay on socitial stereotypes of countries to generate the texts?
  - Does the wordalisation introduce notions of ranking to countries' values.

## Ethical considerations

The World Value Survey is based on questionnaires filled out by individuals from different countries. The sets of individuals are relatively small compared to the populations of the countries.
Further, the questionnaires where given in a limited number of languages for each country, potentially excluding some groups of the population.
Therefore, the data can only give a rough indication of the attitudes of a population. The summaries generated by no means reflect the beliefs and attitudes of any given individual.
Furthermore, it is debatable whether the derived metrics give any accurate insight and we would urge users to consider them within the research context in which they were derived, see [Datasets](#datasets).

## Caveats and recommendations

We have no further caveats and recommendations.
<<<<<<< HEAD
=======

## References

[1] Ingelhart, R. and Welzel, C., 2005. Modernization, cultural change, and democracy: The human development sequence.

[2] Allison, L., Wang, C. and Kaminsky, J., 2021. Religiosity, neutrality, fairness, skepticism, and societal tranquility: A data science analysis of the World Values Survey. Plos one, 16(1), p.e0245231.

> > > > > > > 8de9a07ed0cab9d434b940e3194c891d7f4659fa
