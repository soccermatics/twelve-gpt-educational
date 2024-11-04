# Model card for Personality Test Wordalisation

The Personality GPT is implemented within the [TwelveGPT Education framework](https://github.com/soccermatics/twelve-gpt-educational) and
is intended as an illustration of the methods. The specific application is to personality test data, creating a wordalisations thats 
describes a person based on their answers to a personality test. The design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens, and was adapted for personality tests by Amandine Caut.

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

The *primary use case* of this wordalisation is eductional. It shows how to convert a dataframe of personality test statistics about a person into a text about that person, which might be used by a recruiter. A *secondary use case* might be for users to understand more about the Big Five personality test. This version is not suitable for professional use, such as by a recruiter, because it was not built by qualified  psychologists nor has not been tested. However, the methods used here do have the potential to be adopted to a professional setting. The texts provided by the wordalisations do give a reasonable convincing description of the candidates. We thus discuss, in more detail below, the issues around potential professional applications. Use of the chat for queires not relating to the personality data at hand is *out of scope*. 

## Factors

The dataset was chosen because of avialability of a public dataset. The questionnaire is anonymous, and we did not use any information about the participants' country of origin in the development of the chat.

## Datasets
The dataset used in this project was sourced from Kaggle's open dataset repository [www.kaggle.com/datasets/tunguz/big-five-personality-test]. It consists of 1,015,342 questionnaire responses, collected online by Open Psychometrics. Respondents answered questions on a scale from 1 to 5, where: 1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree.

The dataset goes to a preprocessing cleaning which is handled in the data\_source.py script.
There are 50 questions divided into five categories: Extraversion, Neuroticism, Agreeableness, Conscientiousness, and Openness. Each question is associated with a weight of either +1 or -1, as determined by Open Psychometrics [https://openpsychometrics.org/printable/big-five-personality-test.pdf]. The answers to each question are multiplied by its respective weight, see Table 1.

Table 1: Description of the questions and their weights
| Metric | Questions | Weights |
| :---: |  :--- | ---: |
| Extraversion      | They are the life of the party <br> They dont talk a lot  They feel <br>comfortable around people  <br>They keep in the background  <br>They start conversations  <br>They have little to say  <br>They talk to a lot of different people at parties  <br>They dont like to draw attention to themself  <br>They dont mind being the center of attention  <br>They are quiet around strangers | 1 <br>-1 <br>1 <br>-1 <br>1 <br>-1 <br>1 <br>-1 <br>1 <br>-1         |
| Neuroticism       | They get stressed out easily  <br>They are relaxed most of the time  <br>They worry about things  <br>They seldom feel blue  <br>They are easily disturbed  <br>They get upset easily  <br>They change their mood a lot  <br>They have frequent mood swings  <br>They get irritated easily  <br>They often feel blue | -1  <br>1  <br>-1  <br>1  <br>-1 <br>-1  <br>-1 <br>-1 <br>-1 <br>-1 |
| Agreeableness     | They feel little concern for others  <br>They interested in people  <br>They insult people  <br>They sympathize with others feelings  <br>They are not interested in other peoples problems <br>They have a soft heart  <br>They not really interested in others  <br>They take time out for others  <br>They feel others emotions  <br>They make people feel at ease | -1 <br>1 <br>-1  <br>1 <br>-1 <br>1  <br>-1 <br>1 <br>1  <br>1       |
| Conscientiousness | They are always prepared  <br>They leave their belongings around  <br>They pay attention to details  <br>They make a mess of things  <br>They get chores done right away  <br>They often forget to put things back in their proper place  <br>They like order  <br>They shirk their duties  <br>They follow a schedule  <br>They are exacting in their work   | 1 <br>-1  <br>1 <br>-1 <br>1  <br>-1 <br>1 <br>-1 <br>1 <br>1        |
| Openness          | They have a rich vocabulary  <br>They have difficulty understanding abstract ideas <br>They have a vivid imagination  <br>They are not interested in abstract ideas <br>They have excellent ideas  <br>They do not have a good imagination  <br>They are quick to understand things <br>They use difficult words  <br>They spend time reflecting on things <br>They are full of ideas| 1 <br>-1 <br>1 <br>-1 <br>1 <br>-1 <br>1 <br>1 <br>1 <br>1           |

For each personality category, the score is calculated by summing the responses to the 10 questions in that category. An additional scoring adjustment is applied, resulting in a final score that ranges from 0 to 40 for each category. Subsequently, we compute the z-score for each category's final score, normalizing the results to allow for comparisons across individuals and categories.

## Model

### Quantitative model

The model applies a ranking on the persons in the dataset based on the z-scores. For each metric a z-score is calculated by 
subtracting the mean and dividing by the standard deviation over all persons in the dataset. The persons are then displayed in a distribution
plot with the selected persons highlighted. 

### Normative model

The model ranks individuals in the dataset based on their z-scores. Those positioned further to the left are considered lower on the metric, while those on the right are seen as higher. This ranking applies a standard where individuals with higher metric values are ranked more favorably.

The prompt to *tell it who it is* identifies a human role for the wordalisation as a "candidate". The user-assistant pairs in the stage of *tell it what it knows* describe how the data metrics can be [interpretted](https://github.com/soccermatics/twelve-gpt-educational/blob/personality-gpt/data/describe/Forward_bigfive.xlsx). These descriptions outline the meaning of the metrics.

In the text which is generated at the stage of *tell it what data to use* we use the following function to translate z-scores to evaluation words:
```python
    def category_description(self, value):
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

In the *tell it how to answer* step, we create four examples of candidate's profile. This somewhat compromises how the texts for these four candidates are constructed. These examples and the prompt used emphasise highlighting positive and negative aspects of the person personality, and ignoring aspects which are average or typical.

### Language model

The wordalisation supports both GPT4o and ChatGPT and related APIs, as well as Gemini API. Since these language model's training data includes articles written about the Big Five personality test described here, this knowledge will seem into both the answers generated by the wordalisation and the chat. Some aspects of the answers will come from data external to that in the provided dataframe.

## Evaluation

No systematic *quantitative analyses* have been carried out on this wordalisation. Ideally, this wordalisation should be subjected to a rigorous *qualitative test* of:
- Reliability of information. Are there factual errors in the text?
- Biases. Does the wordalisation use social or ethnic features of the person to create the texts?

## Ethical considerations

Several ethical challenges have arisen in this project.
First, there are no professional psychologists or psychiatrists involved in interpreting the results. The interpretation is done by statistic method as mention above.
Second, the Big Five personality test has several criticisms, such as being overly simplistic [1] and subject to cultural bias. Indeed, according to [2,3,4] the personality is developped in an environmental and cultural context. The test can not be apply universally [1,2], and should have more traits [1].
Third, the test can be easily manipulated by individuals seeking to present a more favorable personality.
Last but not least, we must ask ourselves whether we truly want to automate the hiring process and risk losing the human element. Without this, we may end up with companies that consistently recruit the same type of profile [5]. When a company automates its recruitment process, it often starts by defining the type of candidate already present within the organization, which can inadvertently create a biased dataset [6,7]. This leads to the selection of standardized candidates, who closely resemble existing employees. If the candidates mirror the profiles of those already in the company, how can we expect to bring in fresh perspectives and innovative ideas?
Careful consideration should be made if applying these methods to data collected on atheletes in environments where this type of scruitiny isn't the norm.

## Caveats and recommendations

We have no further caveats and recommendations.

## References
[1] Abood, N. (2019). Big five traits: A critical review. Gadjah Mada International Journal of Business, 21(2), 159–186. <br>
[2] De Raad B, Barelds DP, Levert E, Ostendorf F, Mlacić B, Di Blas L, Hrebícková M, Szirmák Z, Szarota P, Perugini M, Church AT, Katigbak MS. Only three factors of personality description are fully replicable across languages: a comparison of 14 trait taxonomies. J Pers Soc Psychol. 2010 Jan;98(1):160-73. doi: 10.1037/a0017184. PMID: 20053040. <br>
[3] Triandis, Harry & Suh, Eunkook. (2002). Cultural Influences on Personality. Annual review of psychology. 53. 133-60. 10.1146/annurev.psych.53.100901.135200. <br>
[4] Costa PT, Terracciano A, McCrae RR. Gender differences in personality traits across cultures: robust and surprising findings. J Pers Soc Psychol. 2001 Aug;81(2):322-331. doi: 10.1037/0022-3514.81.2.322. PMID: 11519935.<br>
[5] Stephen Bach, Managing Human Resources, fourth edition Personnel Management in Transition, BLACKWELL PUBLISHING <br>
[6] Miranda Bogen, Aaron RiekeHelp, Wanted - An Exploration of Hiring Algorithms, Equity and Bias, 2018 <br>
[7] Miranda Bogen, All the Ways Hiring Algorithms Can Introduce Bias, Analytics and data science, 2019 [https://hbr.org/2019/05/all-the-ways-hiring-algorithms-can-introduce-bias]
