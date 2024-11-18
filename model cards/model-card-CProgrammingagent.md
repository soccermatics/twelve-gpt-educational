## Model card for the CProgrammingagent
The CProgrammingagent is implemented using the [TwelveGPT Education Framework](https://github.com/soccermatics/twelve-gpt-educational) and is intended for helping learners learn C programming based on their current understading of the programming langauge. The design and development of the project was by Matthias Green, David Sumpter and Ágúst Pálmason Morthens and was adapted to the CPteachingagent by Selina Ochukut.
This model card is based on the model cards [paper](https://arxiv.org/abs/1810.03993) and is adapted specifically to the CProgrammingagent project. 

### Jump to section

- [Intended use](#inteded-use)
- [Factors](#factors)
- [Model](#model)
- [Evaluation](#evaluation)
- [Ethical Consideration](#ethical-consideration)
- [Caveats and considerations](#caveats-and-considerations)


### Inteded use

- The ***primary use*** case of the chatbot is to help undergraduate students learn the concepts of for loops as used in C programming in an interactive manner. The chatbot works by using a questioning technique where the user is prompted for their knowledge in the concept. Based on the user response, further questions that enhance the user understading of the concepts are asked.***The socondary*** use case might be to teach C programing langauge in an iteractive manner to other levels of education. The CPteaching agent can be used for teaching other programming langauges, However this version of the CPteachingagent will need to be modified for teaching another programing langauge.


### Factors

- The CPteaching agent has been created specifically for instructing students on for loop as used in the C programming langauge. The idea can be modified to instruct on the other concepts of C programming or any other programming langauge.
- The langauge of interaction with the chatBot is english. The chatBot has not been customized to be used with other human langauges.
- The chatBot relies on the embeddings generated using Gemini as base model

### Normative model
The model helps a user learn the concepts of ***for loops*** as used in the C programming langauge. The model responds to the user based on their response by asking them questions on the topic. The model uses the student response to gauge their knowledge on the topic. Related areas are revisted if the user knowledge is lacking in the area. Complex tasks are given to the user if they demonstrate existing knowledge on the topic. The model is provided with the "user" "assistant" pairs to guide it on how to respond to specific user queries.The next generated question to the user is based on the topic and the users previous response. 

The data used for training the model consisted of the ***user***, ***assistant*** pairs. The dataset was genenrated based on the knolwedgge of one of the authors on C programming.

### Langauge model
The CPteaching model supports both GPT4o and ChatGPT and related APIs as well as Gemini API. Since the training of this models cointain data on C programming and other programming langauges.
### Evaluation
Evaluated based on the factuality of the facts generated
### Ethical consideration
The effect on the use of the tool on the users of the tool should be considered as this might lead to over reliance. As with the general use of LLMs the danger of not well crafted messages might affect the immotional being of the users. Before its actual deployment, an extensive evaluation of the responses will be done to minimize the chances of misinformation and improper langauge.
The other ethical issue that might arise from the use of the tool in educational setup, is the creation of digitial divide those who have access and those who do not have access.

### Caveats and considerations
There are no  more caveats
