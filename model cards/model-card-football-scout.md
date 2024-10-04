# Model card for Football Scout Wordalisation

The football scout is implemented within the [TwelveGPT Education framework](https://github.com/soccermatics/twelve-gpt-educational) and
is intended as an illustration of the methods. Here we detail the specific application to football scouting. It is thus intended as an example to help others building wordalisations. The wordalisations describe players playing as strikers, for at least 300 minutes, in the Premier League 2017-18 season. This work is a derivative of the full [Twelve GPT product](https://twelve.football). The design and code for this project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens. 

This model card is based on the [model cards paper](https://arxiv.org/abs/1810.03993) and is adapted specifically to Wordalisation applications as detailed in [Representing data in words](publication here). We thus also see this model card as an example of good practice for describing wordalisations.

Jump to section:

- [Intended use](#intended-use)
- [Factors](#factors)
- [Datasets](#dataset)
- [Quantitative model](#quantitative-model)
- [Normative model](#quantitative-model)
- [Evaluation](#evaluation)
- [Ethical considerations](#ethical-considerations)
- [Caveats and recommendations](#caveats-and-recommendations)

## Intended use

The *primary use case* of this wordalisation is eductional. It shows how to convert a dataframe of football statistics about a player into a text about that player, which might be used by a scout or of interest to a football fan. A *secondary use case* might be for users to understand more about the skills of male atheletes playing in the Premier League during 2017-18. However, this version cannot be used for professional purposes, i.e. professional football scouting, partly because the data is out of date, but mainly because the functionality is limited. Professional use is thus *out of scope*, as is the use of the chat for queires not relating to the data at hand. 

## Factors

The Football Scout Wordalisation is applied to a very specific demographic group, namely male professional football players. This version is only for use within that group and thus excludes female atheletes. The ethnicity and social background of the players is not documented, but can be (anecdotely) considered to be more diverse than that of the English population as a whole (the Premier League is played in England). Players come from all over the world to play in the Premier League. The dataset was chosen because of avialability of a public dataset and because of the fact that the players will be recognisable names for many users. 

## Datasets

The dataframe used in this project was constructed from a dataset of actions taken by [Premier League players](https://figshare.com/collections/Soccer_match_event_dataset/4415000/2) during the 2017-18 season. A *preprocessing* step, largely based on this [tutotrial](https://soccermatics.readthedocs.io/en/latest/lesson3/ScoutingPlayers.html), converted player actions to a dataframe of counts per ninety and corrected for possession of the ball. Only those players who play as forwards (an attacking role) are included in the dataset.

## Quantitative model

The model applies a ranking on the football players in the dataset based on the z-scores. For each metric a z-scoure is calculated by 
subtracting the mean and dividing by the standard deviation over all players in the dataset. The players are then displayed in a distribution


## Normative model

The model applies a ranking on the football players in the dataset based on the z-scores. So players further to the left are described as worse on the metric and those to the right are described as better. These are socially agreed upon (in football and sporting situations) ways of viewing players, but nevertheless applies a norm whereby players who rank higher in these metrics are ranked more highly. 

The prompt to *tell it who it is* identifies a human role for the wordalisation as a "UK based scout". The user-assistant pairs in the stage of *tell it what it knows* describe how the data metrics can be [interpretted in footballing terms](https://github.com/soccermatics/twelve-gpt-educational/blob/main/data/describe/Forward.xlsx). These descriptions in

In the text which is generated at the stage of *tell it what data to use* we use the following function to translate z-scores to evaluation words:
```python
def describe_level(value):
    thresholds = [1.5, 1, 0.5, -0.5, -1]
    words = ["outstanding", "excellent", "good", "average", "below average", "poor"]
    return describe(thresholds, words, value)
```
There are three positive terms and two negative terms in the words used, with no negative equivalent of "outstanding" defined.

In the *tell it how to answer* step, two examples were created, based on data from 

The players' performances in the Premier League are regularly scrutinised in the media and on social media platforms in terms that are much more prejoritive than those used here. Informal discussions by the wordalisations creators with professional football scouts have suggested that the resulting texts can be "too positive" for professional settings, where clubs have to make important financial decisions based on performance. Nevertheless, this does not take away from the fact that the players analysed here are living human-beings, whose performance is being evaluated in words by an automated system that does not understand broader factors. 

## Evaluation

_Quantitative analyses should be disaggregated, that is, broken down by the chosen factors. Quantitative
analyses should provide the results of evaluating the model according to the chosen metrics, providing
confidence interval values when possible._

Review section 4.7 of the [model cards paper](https://arxiv.org/abs/1810.03993).

*Intersectional result*

## Ethical considerations

_This section is intended to demonstrate the ethical considerations that went into model development,
surfacing ethical challenges and solutions to stakeholders. Ethical analysis does not always lead to
precise solutions, but the process of ethical contemplation is worthwhile to inform on responsible
practices and next steps in future work._

Review section 4.8 of the [model cards paper](https://arxiv.org/abs/1810.03993).

### Data

### Human life

### Mitigations

### Risks and harms

### Use cases

## Caveats and recommendations

We have no further caveats and recommendations.

