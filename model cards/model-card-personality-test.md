# Model card for Personality Test Wordalisation

The Personality GPT is implemented within the [TwelveGPT Education framework](https://github.com/soccermatics/twelve-gpt-educational) and
is intended as an illustration of the methods. Here we detail the specific application to the Personality Test interpretation. It is thus intended as an example to help others building wordalisations. The wordalisations describe a person based on their answers to a personality test. Here is an example about what could be done for a job position, when the recruiter gives the personality test to a candidate. This work is a derivative of the full [Twelve GPT product](https://twelve.football). The design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens, and was adapted for the Personality Test by Amandine Caut.

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

The *primary use case* of this wordalisation is eductional. It shows how to convert a dataframe of personality test statistics about a person into a text about that person, which might be used by a recruiter. A *secondary use case* might be for users to understand more about the Big Five personality test.However, this version is not suitable for professional use, such as by a recruiter, primarily because it lacks the expertise of a licensed psychologist and also because its functionality is limited. Professional use is thus *out of scope*. Use of the chat for queires not relating to the data at hand is also *out of scope*. 

## Factors

The dataset was chosen because of avialability of a public dataset.

## Datasets

The dataframe used in this project was constructed from 1,015,342 questionnaire answers collected online by Open Psychometrics ([https://www.kaggle.com/datasets/tunguz/big-five-personality-test]). A *preprocessing* to clean the dataset is included in data_source.py.
There are 50 questions in 5 categories: Extraversion, Neuroticism, Agreeableness, Conscientiousness , Openness. The sum of the 10 questions in each category results in a score ranging from 0 to 40. Following professional guidelines, we apply a scoring adjustment at the end of each category, referencing the scoring table provided(reference to scoring: https://sites.temple.edu/rtassessment/files/2018/10/Table_BFPT.pdf).Afterward, we calculate the z-score for each category's final score.

## Model

### Quantitative model

The model applies a ranking on the persons in the dataset based on the z-scores. For each metric a z-scoure is calculated by 
subtracting the mean and dividing by the standard deviation over all persons in the dataset. The persons are then displayed in a distribution
plot with the selected persons highlighted. 

### Normative model

The model ranks individuals in the dataset based on their z-scores. Those positioned further to the left are considered lower on the metric, while those on the right are seen as higher. This ranking applies a standard where individuals with higher metric values are ranked more favorably.

The prompt to *tell it who it is* identifies a human role for the wordalisation as a "candidate". The user-assistant pairs in the stage of *tell it what it knows* describe how the data metrics can be [interpretted](https://github.com/soccermatics/twelve-gpt-educational/blob/personality-gpt/data/describe/Forward_bigfive.xlsx). These descriptions outline the meaning of the metrics.

In the text which is generated at the stage of *tell it what data to use* we use the following function to translate z-scores to evaluation words:
```python
    def categorie_description(self, value):
        if value <= -2:
            return 'The candidate is extremely '
        elif -2 < value <= -1:
            return 'The candidate is very '
        elif -1 < value <= -0.5:
            return 'The candidate is quite '
        elif -0.5 < value <= 0.5:
            return 'The candidate is relatively '
        elif 0.5 < value <= 1:
            return 'The candidate is quite '
        elif 1 < value <= 2:
            return 'The candidate is very '
        else: 
            return 'The candidate is extremely '
```
There are four positive terms and three negative terms in the words used. 

In the *tell it how to answer* step, four examples. This somewhat compromises how the texts for these four candidates are constructed. These examples and the prompt used emphasise highlighting positive and negative aspects of the person personality, and ignoring aspects which are average or typical.

### Language model

The wordalisation supports both GPT4o and ChatGPT and related APIs, as well as Gemini API. Since these language model's training data includes articles written about the Big Five personality test described here, this knowledge will seem into both the answers generated by the wordalisation and the chat. Some aspects of the answers will come from data external to that in the provided dataframe.

## Evaluation

No systematic *quantitative analyses* have been carried out on this wordalisation. Ideally, this wordalisation should be subjected to a rigorous *qualitative test* of:
- Reliability of information. Are there factual errors in the text?
- Biases. Does the wordalisation use social or ethnic features of the person to create the texts?

## Ethical considerations

Severel ethical challenges have arisen in this project.
First, there are no professional psychologists or psychiatrists involved in interpreting the results. The interpretation is done by statistic method as mention above. 
Second, the Big Five personality test has several criticisms, such as being overly simplistic and subject to cultural bias.
Third, the test can be easily manipulated by individuals seeking to present a more favorable personality.
Last but not least, we must ask ourselves whether we truly want to automate the hiring process and risk losing the human element. Without this, we may end up with companies that consistently recruit the same type of profile. When a company automates its recruitment process, it often starts by defining the type of candidate already present within the organization, which can inadvertently create a biased dataset. This leads to the selection of standardized candidates, who closely resemble existing employees. If the candidates mirror the profiles of those already in the company, how can we expect to bring in fresh perspectives and innovative ideas?
Careful consideration should be made if applying these methods to data collected on atheletes in environments where this type of scruitiny isn't the norm.

## Caveats and recommendations

We have no further caveats and recommendations.

