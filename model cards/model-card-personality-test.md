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

The *primary use case* of this wordalisation is educational. It shows how to convert a dataframe of personality test statistics about a person into a text about that person, which might be used by a recruiter. A *secondary use case* might be for users to understand more about the Big Five personality test. This version is not suitable for professional use, such as by a recruiter, because it was not built by qualified  psychologists nor has not been tested. However, the methods used here do have the potential to be adopted to a professional setting. The texts provided by the wordalisation do give a reasonable convincing description of the candidates. We thus discuss, in more detail below, the issues around potential professional applications. Use of the chat for queries not relating to the personality data at hand is *out of scope*. 

## Factors

The dataset was chosen because of avialability of a public dataset. The questionnaire is anonymous, and we did not use any information about the participants' country of origin in the development of the chat.

## Datasets
The dataset used in this project was sourced from Kaggle's open dataset repository [www.kaggle.com/datasets/tunguz/big-five-personality-test]. It consists of 1,015,342 questionnaire responses, collected online by Open Psychometrics. Respondents answered questions on a scale from 1 to 5, where: 1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree.

The dataset goes to a preprocessing cleaning which is handled in the data\_source.py script.
There are 50 questions divided into five categories: Extraversion, Neuroticism, Agreeableness, Conscientiousness, and Openness. Each question is associated with a weight of either +1 or -1, as determined by Open Psychometrics [https://openpsychometrics.org/printable/big-five-personality-test.pdf]. The answers to each question are multiplied by its respective weight, see Table 1.
For each personality category, the score is calculated by summing the responses to the 10 questions in that category. An additional scoring adjustment is applied, resulting in a final score that ranges from 0 to 40 for each category. Subsequently, we compute the z-score for each category's final score, normalizing the results to allow for comparisons across individuals and categories.

The dataset creation is handle is the file data\_source.py. We use the dataset in the different instance as visual.py, description.py. 




Table 1: Description of the questions and their weights
| Metric | Questions | Weights |
| :---: |  :--- | ---: |
| Extraversion      | They are the life of the party <br> They dont talk a lot <br> They feel comfortable around people  <br>They keep in the background  <br>They start conversations  <br>They have little to say  <br>They talk to a lot of different people at parties  <br>They dont like to draw attention to themself  <br>They dont mind being the center of attention  <br>They are quiet around strangers | 1 <br>-1 <br>1 <br>-1 <br>1 <br>-1 <br>1 <br>-1 <br>1 <br>-1         |
| Neuroticism       | They get stressed out easily  <br>They are relaxed most of the time  <br>They worry about things  <br>They seldom feel blue  <br>They are easily disturbed  <br>They get upset easily  <br>They change their mood a lot  <br>They have frequent mood swings  <br>They get irritated easily  <br>They often feel blue | -1  <br>1  <br>-1  <br>1  <br>-1 <br>-1  <br>-1 <br>-1 <br>-1 <br>-1 |
| Agreeableness     | They feel little concern for others  <br>They interested in people  <br>They insult people  <br>They sympathize with others feelings  <br>They are not interested in other peoples problems <br>They have a soft heart  <br>They not really interested in others  <br>They take time out for others  <br>They feel others emotions  <br>They make people feel at ease | -1 <br>1 <br>-1  <br>1 <br>-1 <br>1  <br>-1 <br>1 <br>1  <br>1       |
| Conscientiousness | They are always prepared  <br>They leave their belongings around  <br>They pay attention to details  <br>They make a mess of things  <br>They get chores done right away  <br>They often forget to put things back in their proper place  <br>They like order  <br>They shirk their duties  <br>They follow a schedule  <br>They are exacting in their work   | 1 <br>-1  <br>1 <br>-1 <br>1  <br>-1 <br>1 <br>-1 <br>1 <br>1        |
| Openness          | They have a rich vocabulary  <br>They have difficulty understanding abstract ideas <br>They have a vivid imagination  <br>They are not interested in abstract ideas <br>They have excellent ideas  <br>They do not have a good imagination  <br>They are quick to understand things <br>They use difficult words  <br>They spend time reflecting on things <br>They are full of ideas| 1 <br>-1 <br>1 <br>-1 <br>1 <br>-1 <br>1 <br>1 <br>1 <br>1           |

Table 2: Additional points for each traits 
| Metric | Additional points |
| :---: |  :---: | 
| Extraversion      | 20 |
| Neuroticism       | 38 |
| Agreeableness     | 14 |
| Conscientiousness | 14 |
| Openness          | 8 |

For each personality category, the score is calculated by summing the responses to the 10 questions in that category. An additional scoring adjustment is applied, see Table 2, resulting in a final score that ranges from 0 to 40 for each category. Subsequently, we compute the z-score for each category's final score, normalizing the results to allow for comparisons across individuals and categories.

## Model

### Quantitative model

The model applies a ranking on the persons in the dataset based on the z-scores. For each metric a z-score is calculated by 
subtracting the mean and dividing by the standard deviation over all persons in the dataset. The persons are then displayed in a distribution
plot with the selected persons highlighted. 
The model ranks individuals in the dataset based on their z-scores. Those positioned further to the left are considered lower on the metric, while those on the right are seen as higher. This ranking applies a standard where individuals with higher metric values are ranked more favorably.

### Normative model
#### Tell it who it is


The prompt to *tell it who it is* identifies a human role for the wordalisation as a "candidate". The user-assistant pairs in the stage of *tell it what it knows* describe how the data metrics can be [interpretted](https://github.com/soccermatics/twelve-gpt-educational/blob/personality-gpt/data/describe/Forward_bigfive.xlsx). These descriptions outline the meaning of the metrics.
For example: 

{<br>
  "role": "system",<br>
  "content": "You are a recruiter. 
You provide succinct and to-the-point explanations about a candidate using data. 
You use the information given to you from the data and answers to earlier user/assistant pairs to give summaries of candidates."<br>
}


#### Tell is what it knows

In this example, we created question-answer pairs to represent each personality trait in the Personality Wordalisation framework. To ensure these pairs accurately reflect each trait, we used descriptions from Wikipedia to contextualize the measurements in the data. Each trait—Extraversion, Conscientiousness, Openness, Agreeableness, and Neuroticism—is represented through multiple question-answer pairs derived from Wikipedia's explanations. The goal is to identify the questions that best capture the essence of each trait through their corresponding answers. To illustrate, below is an example using the Wikipedia description of 'Openness'. From the following description: 

 _"Openness to experience is a general appreciation for art, emotion, adventure, unusual ideas, imagination, curiosity, and variety of experience. People who are open to experience are intellectually curious, open to emotion, sensitive to beauty, and willing to try new things. They tend to be, when compared to closed people, more creative and more aware of their feelings. They are also more likely to hold unconventional beliefs. Open people can be perceived as unpredictable or lacking focus, and more likely to engage in risky behaviour or drug-taking. Moreover, individuals with high openness are said to pursue self-actualisation specifically by seeking out intense, euphoric experiences. Conversely, those with low openness want to be fulfilled by persevering and are characterised as pragmatic and data-driven – sometimes even perceived to be dogmatic and closed-minded. Some disagreement remains about how to interpret and contextualise the openness factor as there is a lack of biological support for this particular trait. Openness has not shown a significant association with any brain regions as opposed to the other four traits which did when using brain imaging to detect changes in volume associated with each trait."_ [https://en.wikipedia.org/wiki/Big\_Five\_personality\_traits]

 We create an excel document with two columns one *Users*, with the questions, and one *Assistant* with the corresponding answers. We obtain the example Table 3. In the case of the personality tests, we reiterate the process for the four others traits. It enables the system to acquire insights into each personality trait. At this stage, we can equip the chat with professional expertise.

Table 3: Example of a pairs of questions and answers for the personality test chatbot construction.
| **User** | **Assistant** |
|----------|---------------|
| What is openness to experience? | Openness to experience is a general appreciation for art, emotion, adventure, unusual ideas, imagination, curiosity, and variety of experience. |
| What characteristics do people with high openness to experience typically have? | People who are open to experience are intellectually curious, open to emotion, sensitive to beauty, and willing to try new things. They tend to be, when compared to closed people, more creative and more aware of their feelings. They are also more likely to hold unconventional beliefs. |
| How might people with high openness to experience be perceived? | Open people can be perceived as unpredictable or lacking focus, and more likely to engage in risky behaviour or drug-taking. Moreover, individuals with high openness are said to pursue self-actualisation specifically by seeking out intense, euphoric experiences. |
| How might people with low openness to experience be perceived? | People with low openness want to be fulfilled by persevering and are characterised as pragmatic and data-driven – sometimes even perceived to be dogmatic and closed-minded. |
| What is the current scientific consensus on the biological basis of openness to experience? | Some disagreement remains about how to interpret and contextualise the openness factor as there is a lack of biological support for this particular trait. Openness has not shown a significant association with any brain regions as opposed to the other four traits which did when using brain imaging to detect changes in volume associated with each trait. |


#### Tell it what data to use

The creation of personality descriptions is handled by the `description.py` script. Using the z-score results, the script generates a personalized paragraph for each individual. Each paragraph contains at least two phrases reflecting each of the five personality traits. The phrasing is determined by the z-score, where one phrase is influenced by the direction of the z-score (positive or negative) and the other incorporates an adjective based on its magnitude. The positive or negative sign of the z-score dictates the tone, while the magnitude of the z-score is used to assign an adjective that characterizes the trait. This is done by the function `categorie_description` as we can see in the following table (refer to Table 3). 

The description is followed by two adjectives for each category, determined by the z-score's direction—positive or negative. 

Extraversion will be described as *outgoing and energetic* when the z-score is positive, accompanied by the phrase, "The candidate tends to be more social.". For a negative z-score, extraversion will be characterized as *solitary and reserved*, with the phrase, "The candidate tends to be less social.".
Neuroticism will be described as *sensitive and nervous* for a positive z-score, with the additional phrase, "The candidate tends to experience more negative emotions and anxiety.". For a negative z-score, it will be characterized as *resilient and confident*, and we add, "The candidate tends to experience fewer negative emotions and less anxiety.".
Agreeableness will be framed as *friendly and compassionate* when the z-score is positive, with the phrase, "The candidate tends to be more cooperative, polite, kind, and friendly.". For a negative z-score, it will be described as *critical and rational*, along with the phrase, "The candidate tends to be less cooperative, polite, kind, and friendly.".
Conscientiousness will be described as *efficient and organized* for a positive z-score, with the phrase, "The candidate tends to be more careful and diligent.". For a negative z-score, it will be characterized as *extravagant and careless*, with the phrase, "The candidate tends to be less careful and diligent.".
Openness will be described as *inventive and curious* for a positive z-score, with the phrase, "The candidate tends to be more open to new ideas and experiences.". For a negative z-score, it will be characterized as *consistent and cautious*, with the phrase, "The candidate tends to be less open to new ideas and experiences.".


Table 3: The thresholds and text descriptions are used to interpret the Z-score values, translating them into words to effectively describe the candidate.
| **Z-score value**      | **Text description**                   |
|------------------------|----------------------------------------|
| Below -2               | 'The candidate is extremely '          |
| Between -2 and -1      | 'The candidate is very '               |
| Between -1 and -0.5    | 'The candidate is quite '              |
| Between -0.5 and 0.5   | 'The candidate is relatively '         |
| Between 0.5 and 1      | 'The candidate is quite '              |
| Between 1 and 2        | 'The candidate is very '               |
| Higher than 2          | 'The candidate is extremely '          |

Highlighting it as a key characteristic of the individual. This ensures that more extreme scores highlight key characteristics by drawing attention to extreme scores, ensuring that these traits stand out in the final description. If the z-score for a particular trait exceeds 1 or falls below -1, we add a specific phrase to emphasize this trait. We first review the 10 questions within the category to find the question with the highest score for z-scores above 1. The added phrase will begin with "In particular, they said that" followed by the question text. Similarly, for z-scores below -1, we identify the question with the lowest score and apply the same phrasing to underline that trait. This ensures that significant traits are properly highlighted in the description.

#### Tell is how to answer

We have now constructed a description based on the data and the zscore. At this stage the aim is to show to the AI agent, how to answer. For this purpose, we create an excel file with one column, 'user', that correspond to the 'constructed' description example, and on column 'assistant', that will give a modified description. In this example, we produce four examples of candidate' profile description. We illustrate it by the Table 4.
This somewhat compromises how the texts for these four candidates are constructed. These examples and the prompt used emphasise highlighting positive and negative aspects of the person personality, and ignoring aspects which are average or typical.

Table 4: Example of the 'constructed description' and the 'generated description'
| **User** | **Assistant** |
|----------|---------------|
| The candidate is very outgoing and energetic. In particular, they said that they start conversations. The candidate is quite sensitive and nervous. The candidate tends to feel more negative emotions like anxiety. The candidate is quite friendly and compassionate. The candidate tends to be more cooperative, polite, kind, and friendly. The candidate is very efficient and organized. The candidate tends to be more careful or diligent. In particular, they said that they pay attention to details. The candidate is relatively consistent and cautious. The candidate tends to be less open. | The candidate is outgoing, energetic, and takes the initiative in starting conversations. They are sensitive and can feel nervous at times, often experiencing anxiety or negative emotions. Despite this, they are friendly, compassionate, and naturally inclined to be cooperative, polite, and kind. Their efficiency and organizational skills stand out, as they are diligent and pay close attention to details. While generally consistent and cautious, the candidate tends to be more reserved and less open in new or unfamiliar situations. |


### Language model

The wordalisation supports both GPT4o and ChatGPT and related APIs, as well as Gemini API. Since these language model's training data includes articles written about the Big Five personality test described here, this knowledge will seem into both the answers generated by the wordalisation and the chat. Some aspects of the answers will come from data external to that in the provided dataframe.

## Evaluation

Qualitative evaluation during development allowed us to refine the prompt to ensure that the chatbot was providing satisfactory responses. Qualitative analysis is also import for identifying possible issues and to establish the limitations and usefulness of the application. 

In addition we also performed the quantitative analysis described below to evaluate whether the model was providing factual and relevant information about the dataset. Figure 3 shows the results of this analysis. 


![Accuracy](https://github.com/soccermatics/twelve-gpt-educational/blob/dev/model%20cards/imgs/accuracy_person.png)

![Accuracy](model cards/imgs/accuracy_person.png)

Figure 3: Comparison of the class labels generated by the normative model with classes reconstructed from the wordalisations. Multiple wordalisations were generated for each data point, so that at least 10 valid reconstructions per data point were found, and the mean accuracy is taken over all wordalistations. We compare the accuracy of the model for two different prompts, one in which data in the form of synthetic texts was given (purple) and in the other the data
was omitted (red). The dashed line indicates the expected accuracy if the class labels were randomly chosen according to a uniform probability distribution and lie at an accuracy of $\frac{1}{2}$.

For each data point in our datasets we generate a Wordalisation using a [prompt template](https://github.com/soccermatics/twelve-gpt-educational/tree/dev/evaluation/prompts) almost identical to the prompt used in the application. For comparison, we also generated Wordalisations using a version of the prompt that did not contained the relevant synthetic text generated by the normative model. To discourage the LLM from declining to respond we added the sentence `If no data is provided answer anyway, using your prior statistical knowledge.' and modified one of the in-context learning examples by removing synthetic text from the user prompt while leaving the Wordalisation (response) unchanged. For consistency, the same prompt template was used both when data was and was not provided. To take into account random variations in the Wordalisations due to the stochastic nature of the LLM, we passed each prompt to the LLM multiple times to generate a set of Wordalisations for evaluation. 

In a new chat instance, we prompt the LLM to reconstruct the data from a given Wordalisation in the form of a json file. We then compare the `true' class according to the normative model with the reconstructed data class to measure how faithfully the Wordalisation represents on the given data. When the generated json file could not be parsed, the data was discarded.

This approach has some weaknesses, including relaying on the LLM to generate accurate reconstructions. In cases where the Wordalisations are more formulaic, this type of evaluation work can well, as was the case here. However, in cases where the texts are more engaging the reconstruction is more nuanced, see the football scout use-case. 

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
