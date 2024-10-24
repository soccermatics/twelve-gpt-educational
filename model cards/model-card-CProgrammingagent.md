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

- The ***primary use*** case of the chatbot is to help undergraduate students learn the concepts of for loops as used in C programming in an interactive manner that chucks the content to be learnt.***The socondary*** use case might be to teach any programing langauge in an iteractive manner. However this version of the CPteaching agent will need to be modified for teaching another programing langauge

- The chat bot is inteded to be with English being the langaguge for the iteracting with the chatbot. 

### Factors

- The CPteaching agent has been created specifically for instructing students on for loop as used in the C programming langauge. The idea can be modified to instruct on the other concepts of C programming or any other programming langauge.
- 

### Normative model
The model helps a user learn the concepts of ***for loops*** as used in the C programming langauge. The model responds to the user based on their response by asking them questions on the topic. The model uses the student response to gauge their knowledge on the topic. Related areas are revisted if the user knowledge is lacking in the area. Complex tasks are given to the user if they demonstrate existing knowledge on the topic. The model is provided with the "user" "assistant" pairs to guide it on how to respond to specific user queries.
The next generated question to the user is based on the topic and the users previous response.

### Langauge model
The CPteaching model supports both GPT4o and ChatGPT and related APIs as well as Gemini API. Since the training of this models cointain data on C programming and other programming langauges.
### Evaluation
Evaluated based on the factuality of the facts generated
### Ethical consideration
The effect on the use of the tool on the users of the tool should be considered as this might lead to over reliance. As with the general use of LLMs the danger of not well crafted messages might affect the immotional being of the users. Before its actual deployment, an extensive evaluation of the responses will be done to minimize the chances of misinformation and improper langauge.
The other ethical issue that might arise from the use of the tool in educational setup, is the creation of digitial divide those who have access and those who do not have access.

### Caveats and considerations
There are no  more caveats
