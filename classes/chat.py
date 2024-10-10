import streamlit as st
import json
import openai
from itertools import groupby
from types import GeneratorType
import pandas as pd
import numpy as np

from settings import USE_GEMINI

if USE_GEMINI:
    from settings import USE_GEMINI, GEMINI_API_KEY, GEMINI_CHAT_MODEL
else:
    from settings import GPT_BASE, GPT_VERSION, GPT_KEY, GPT_ENGINE
    
from classes.description import (
    PlayerDescription
)

from classes.data_source import Arguments

from classes.embeddings import PlayerEmbeddings,TrolleyEmbeddings,LessonEmbeddings

from classes.visual import (
    Visual,DistributionPlot
)

import utils.sentences as sentences
from utils.gemini import convert_messages_format

openai.api_type = "azure"

class Chat:
    function_names = []
    def __init__(self, chat_state_hash, state="empty"):

        if "chat_state_hash" not in st.session_state or chat_state_hash != st.session_state.chat_state_hash:
            # st.write("Initializing chat")
            st.session_state.chat_state_hash = chat_state_hash
            st.session_state.messages_to_display = []
            st.session_state.chat_state = state

        # Set session states as attributes for easier access
        self.messages_to_display = st.session_state.messages_to_display
        self.state = st.session_state.chat_state
    
    def instruction_messages(self):
        """
        Sets up the instructions to the agent. Should be overridden by subclasses.
        """
        return []

    def add_message(self, content, role="assistant", user_only=True, visible = True):
        """
        Used by app.py to start off the conversation with plots and descriptions.
        """
        message = {"role": role, "content": content}
        self.messages_to_display.append(message)

    def get_input(self):
        """
        Get input from streamlit."""
  
        if x := st.chat_input(placeholder=f"What else would you like to know?"):
            if len(x) > 500:
                st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")

            self.handle_input(x)
                

    def handle_input(self, input):
        """
        The main function that calls the GPT-4 API and processes the response.
        """

        # Get the instruction messages. 
        messages = self.instruction_messages()

        # Add a copy of the user messages. This is to give the assistant some context.
        messages = messages + self.messages_to_display.copy()

        # Get relevent information from the user input and then generate a response.
        # This is not added to messages_to_display as it is not a message from the assistant.
        get_relevant_info = self.get_relevant_info(input)

        # Now add the user input to the messages. Don't add system information and system messages to messages_to_display.
        self.messages_to_display.append({"role": "user", "content": input})
                         
        messages.append({"role": "user", "content": f"Here is the relevant information to answer the users query: {get_relevant_info}\n\n```User: {input}```"})

        # Remove all items in messages where content is not a string
        messages = [message for message in messages if isinstance(message["content"], str)]

        # Show the messages in an expander
        st.expander("GPT Messages", expanded=False).write(messages)  

        # Check if use gemini is set to true
        if USE_GEMINI:
            import google.generativeai as genai
            converted_msgs = convert_messages_format(messages)

            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(
                model_name=GEMINI_CHAT_MODEL,
                system_instruction=converted_msgs["system_instruction"]
            )
            chat = model.start_chat(history=converted_msgs["history"])
            response = chat.send_message(content=converted_msgs["content"])

            answer = response.text
        else:
            # Call the GPT-4 API
            openai.api_base = GPT_BASE
            openai.api_version = GPT_VERSION
            openai.api_key = GPT_KEY

            response = openai.ChatCompletion.create(
                engine=GPT_ENGINE,
                messages=messages
                )
        
            answer=response['choices'][0]['message']['content']
        message = {"role": "assistant", "content": answer}
        
        # Add the returned value to the messages.
        self.messages_to_display.append(message)
   
    def display_content(self,content):
        """
        Displays the content of a message in streamlit. Handles plots, strings, and StreamingMessages.
        """
        if isinstance(content, str):
            st.write(content)

        # Visual
        elif isinstance(content, Visual):
            content.show()

        else:
            # So we do this in case
            try: content.show()
            except: 
                try: st.write(content.get_string())
                except:
                    raise ValueError(f"Message content of type {type(content)} not supported.")


    def display_messages(self):
        """
        Displays visible messages in streamlit. Messages are grouped by role.
        If message content is a Visual, it is displayed in a st.columns((1, 2, 1))[1].
        If the message is a list of strings/Visuals of length n, they are displayed in n columns. 
        If a message is a generator, it is displayed with st.write_stream
        Special case: If there are N Visuals in one message, followed by N messages/StreamingMessages in the next, they are paired up into the same N columns.
        """
        # Group by role so user name and avatar is only displayed once

        #st.write(self.messages_to_display)

        for key, group in groupby(self.messages_to_display, lambda x: x["role"]):
            group = list(group)

            if key == "assistant":
                avatar = "data/ressources/img/twelve_chat_logo.svg"
            else:
                try:
                    avatar = st.session_state.user_info["picture"]
                except:
                    avatar = None

            message=st.chat_message(name=key, avatar=avatar)   
            with message:
                for message in group:
                    content = message["content"]
                    self.display_content(content)
                

    def save_state(self):
        """
        Saves the conversation to session state.
        """
        st.session_state.messages_to_display = self.messages_to_display
        st.session_state.chat_state = self.state


class PlayerChat(Chat):
    def __init__(self, chat_state_hash, player, players, state="empty"):
        self.embeddings = PlayerEmbeddings()
        self.player = player
        self.players = players
        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": "You are a UK-based football scout."},
            {"role": "user", "content": (
                "After these messages you will be interacting with a user of a football scouting platform. "
                f"The user has selected the player {self.player.name}, and the conversation will be about them. "
                "You will receive relevant information to answer a user's questions and then be asked to provide a response. "
                "All user messages will be prefixed with 'User:' and enclosed with ```. "
                "When responding to the user, speak directly to them. "
                "Use the information provided before the query  to provide 2 sentence answers."
                " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                )
            },
        ]
        return first_messages


    def get_relevant_info(self, query):
 
        #If there is no query then use the last message from the user
        if query=='':
            query = self.visible_messages[-1]["content"]
        
        ret_val = "Here is a description of the player in terms of data: \n\n"   
        description = PlayerDescription(self.player)
        ret_val += description.synthesize_text()

        # This finds some relevant information
        results = self.embeddings.search(query, top_n=5)
        ret_val += "\n\nHere is a description of some relevant information for answering the question:  \n"   
        ret_val +="\n".join(results["assistant"].to_list())
        
        ret_val += f"\n\nIf none of this information is relevent to the users's query then use the information below to remind the user about the chat functionality: \n"
        ret_val += "This chat can answer questions about a player's statistics and what they mean for how they play football."
        ret_val += "The user can select the player they are interested in using the menu to the left."

        
        return ret_val


    def get_input(self):
        """
        Get input from streamlit."""
  
        if x := st.chat_input(placeholder=f"What else would you like to know about {self.player.name}?"):
            if len(x) > 500:
                st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")

            self.handle_input(x)

class TrolleyChat(Chat):
    def __init__(self, chat_state_hash, arguments, userstance, overallThesis,argumentsMade=[],totalscore=0, state="empty",gameOver=False):
        self.embeddings = TrolleyEmbeddings()
        self.arguments =arguments
        stanceSwap = {'Pro': 'Con', 'Con': 'Pro'}
        self.stance = stanceSwap[userstance]
        self.userOverallStance = userstance
        self.argumentsMade = argumentsMade
        self.overallThesis =overallThesis
        self.gameOver=gameOver

        # Initialize the total score as an int and originality score as float
        self.totalscore = totalscore
        self.originalityscore = np.float64(0.0)
        

        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": (
                "You are talking to a human user about the following thesis: " + self.overallThesis + ". "
                " You are currently arguing the " + self.stance + " side of the argument."
                )
            },
            {"role": "user", "content": (
                "After these messages you will be interacting with a user who will argue against your position. "
                "You will receive relevant information to address the user's argument and then be asked to provide a response. "
                "All user messages will be prefixed with 'User:' and enclosed with ```. "
                "When responding to the user, speak directly to them. "
                "Use the information provided before the query to provide 2-3 sentence answers."
                " Do not deviate from this information and provide minimal additional information. Only talk about the thesis and the arguments."
                )
            },
        ]
        return first_messages


    def get_relevant_info(self, query):
 
        #If there is no query then use the last message from the user
        if query=='':
            query = self.visible_messages[-1]["content"]
       
        numberofarguments = 10
        sidebar_container = st.sidebar.container()

        similaritythreshold = 0.75

        if self.totalscore>=100:
            ret_val = "The user has already won the game by getting the maximum score of 100. "
            ret_val += "They are the winners!!!You will not respond more to arguments"
            ret_val += "Tell the user to share the game with a friend or play again. "
            with sidebar_container:
            
                st.write(f'You have won!')
                st.write(f'Total score: 100')
        
            return ret_val
        
        if self.gameOver==True:
            ret_val = "Thank the user for playing the game. "
            ret_val = f"Tell them their final score of {int(np.ceil(self.totalscore))}. "
                        
            with sidebar_container:
            
                st.write(f'Your argument is over.')
                st.write(f'Final score: {int(np.ceil(self.totalscore))}')
        
            return ret_val

        # This finds the argument that is most similar to the user's query
        results = self.embeddings.search(query, top_n=numberofarguments)

        # All the arguments are not relevant, so tell the user and return
        if results.iloc[0]['similarities'] < similaritythreshold:
            ret_val = "\n\nThe user said:  \n"   
            ret_val +="\n".join(results["user"].to_list())
            ret_val = "but this is not a releavant argument. "
            ret_val += "Tell the user that they should try to make an argument that is more relevant to the thesis. "
            with sidebar_container:
                    st.write(f'Novelty: 0/{numberofarguments}')
                    st.write(f'Total score: {int(np.ceil(self.totalscore))}')
            return ret_val

        # Keep a track of similarity to previous arguments made
        # Set to similaritythreshold as minimum.
        if len(self.argumentsMade) > 0:
            previousArguments = results['assistant'].isin(self.argumentsMade)
            #Check if previousArgumnets contains at least one True value
            if previousArguments.any():
                similaritytoprevious = results[previousArguments]['similarities'].mean()
            else:
                similaritytoprevious = similaritythreshold
        else:
            similaritytoprevious = similaritythreshold


        # Remove the results that are in the argumentsMade list
        for argumentCode in self.argumentsMade:
            results = results[results["assistant"] != argumentCode]

        # Remove the results where the overall argument does not match the overall stance
        for r in results.iterrows():
            assistant=r[1]['assistant']
            overall = self.arguments.df[self.arguments.df['assistant']==assistant].iloc[0]['overall']
            results.at[r[0],'overall'] = overall     
        results = results[results["overall"] == self.userOverallStance]
        #st.write(results)

        # The first part of the originality score are number of arguments remaining unused, which shows relevance to the question.
        self.originalityscore = len(results)/2

        if len(results) == 0:
            ret_val = " Tell the user that they are no longer providing sufficiently novel arguments anymore or our perhaps making arguments that do not support their position. "
            ret_val += "Tell the user that you have no further arguments to make against theirs, but maybe next time they play they can try broader arguments. "
            ret_val += "Then let them know that their final score in the argument game is " + str(int(self.totalscore))
            with sidebar_container:
                st.write(f'Game over! Try again.')
                st.write(f'Total score: {int(np.ceil(self.totalscore))}')
            self.gameOver=True

            return ret_val

        # Get the 10 best arguments which oppose the current stance.
        currentArguments= []
        for _,r in results.iterrows():    
            argumentCode = r["assistant"]
            currentArgumentsdf=self.arguments.get_arguments(argumentCode,'Con')
            currentArguments = currentArguments + list(currentArgumentsdf['user'])
            if len(currentArguments) > 10:
                break
        
        # Remove that argument from the list of arguments
        self.argumentsMade.append(argumentCode)
        #st.write(self.argumentsMade)

        # Add a score based on how original is compared to arguments made so far.
        with sidebar_container:
            part2 = 10*numberofarguments*(results.iloc[0]["similarities"]-similaritytoprevious)
            self.originalityscore = min(10.0,self.originalityscore+max(part2,0.0))
            st.write(f'Novelty: {int(np.ceil(self.originalityscore))}/{numberofarguments}')
            self.totalscore = self.totalscore+np.ceil(self.originalityscore)
            st.write(f'Total score: {int(np.ceil(self.totalscore))}')
        
        if len(currentArguments) == 0:
            ret_val = "Write at most three sentences arguing against the user's point, based on the previous discussion. "
        else:
            ret_val = f"Here is a description of the points which you can make to refute their argument, which is: {query}: \n\n"   
            for a in currentArguments:
                ret_val += a + "\n"
            ret_val += "\n\nWrite two (or at most three) sentences arguing against the user's point, using the points above. "
        
        ret_val += "In the first sentence, restate the user's argument. In the second (and possibly third) sentence, provide a counter-argument."
        ret_val += "Always address the user directly as 'you' in the first person and write natural sounding lanaguage, with no headers. Write as if you are having a conversation with the user."

        if self.totalscore>=100:
            ret_val += "In addition to the above, also congratulate the user for winning game and getting the maximum score of 100. "
            ret_val += "They are the winners!!! "
            

        return ret_val
    
    def get_input(self):
        """
        Get input from streamlit."""
  
        if x := st.chat_input(placeholder=f"Please make your argument here"):
            if len(x) > 500:
                st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")

            self.handle_input(x)


#__________________________________________________________________________________________________________________________________

class LessonChat(Chat):
    def __init__(self, chat_state_hash, overallThesis,arguments, state="empty",gameOver=False):
        self.embeddings = LessonEmbeddings()
        self.arguments =arguments
        #stanceSwap = {'Pro': 'Con', 'Con': 'Pro'}
        #self.stance = stanceSwap[userstance]
        #self.userOverallStance = userstance
        #self.argumentsMade = argumentsMade
        self.overallThesis =overallThesis
        self.gameOver=gameOver

        # Initialize the total score as an int and originality score as float
        #self.totalscore = totalscore
        self.originalityscore = np.float64(0.0)
        

        super().__init__(chat_state_hash, state=state)

    def instruction_messages(self):
        """
        Instruction for the agent.
        """
        first_messages = [
            {"role": "system", "content": (
                #"You are talking to a learner about the following topic: " + self.overallThesis + ". "
                "You are an instructor, you guiding the user to learn about loops in C" 
                
                )
            },
            {"role": "user", "content": (
                "After these messages you will be interacting with the user who will tell you what they know about loops"
                "Your task is to gauge the user understading of the loops and ask them a question that will help fill the knowledge gaps they have"
                "You will receive relevant information to answer a user's questions and then be asked to provide a response in form of a question. "
                "All user messages will be prefixed with 'user:' and enclosed with ```. "
                "When responding to the user, speak directly to them. "
                "If the user has knowledge on loops, ask them to write code that demonstrate the use of loops, like displaying a range of numbers"
                "Evaluate the response"
                "If the user says they do not know about loops, ask them a question on topics that preceed loops"
                " Do not deviate from this information or provide additional information that is not in the text returned by the functions."
                )
            },
        ]
        return first_messages


    def get_relevant_info(self, query):
 
        #If there is no query then use the last message from the user
        if query=='':
            query = self.visible_messages[-1]["content"]
       
        numberofarguments = 8
        sidebar_container = st.sidebar.container()

        similaritythreshold = 0.75
        '''
        if self.totalscore>=100:
            ret_val = "The user has already won the game by getting the maximum score of 100. "
            ret_val += "They are the winners!!!You will not respond more to arguments"
            ret_val += "Tell the user to share the game with a friend or play again. "
            with sidebar_container:
            
                st.write(f'You have won!')
                st.write(f'Total score: 100')
        
            return ret_val '''
        
        if self.gameOver==True:
            ret_val = "Thank the user for using the system "
            #ret_val = f"Tell them their final score of {int(np.ceil(self.totalscore))}. "
            '''            
            with sidebar_container:
            
                st.write(f'Your argument is over.')
                st.write(f'Final score: {int(np.ceil(self.totalscore))}')
            '''
            return ret_val

        # This finds the argument that is most similar to the user's query
        results = self.embeddings.search(query, top_n=numberofarguments)

        # All the arguments are not relevant, so tell the user and return
        if results.iloc[0]['similarities'] < similaritythreshold:
            ret_val = "\n\nThe user said:  \n"   
            ret_val +="\n".join(results["user"].to_list())
            ret_val = "but this is not a releavant topic. "
            ret_val += "Tell the user that they should stick to loops in C. "
            with sidebar_container:
                    st.write(f'Novelty: 0/{numberofarguments}')
                    #st.write(f'Total score: {int(np.ceil(self.totalscore))}')
            return ret_val

        # Keep a track of similarity to previous arguments made
        # Set to similaritythreshold as minimum.
        #if len(self.argumentsMade) > 0:
         #   previousArguments = results['step'].isin(self.argumentsMade)
            #Check if previousArgumnets contains at least one True value
          #  if previousArguments.any():
           #     similaritytoprevious = results[previousArguments]['similarities'].mean()
            #else:
             #   similaritytoprevious = similaritythreshold
       # else:
        #    similaritytoprevious = similaritythreshold


        # Remove the results that are in the argumentsMade list
        #for argumentCode in self.argumentsMade:
           # results = results[results["step"] != argumentCode]

        # Remove the results where the overall argument does not match the overall stance
        for r in results.iterrows():
            assistant=r[1]['assistant']
            overall = self.arguments.df[self.arguments.df['step']=='1.'].iloc[0]['overall']
            results.at[r[0],'overall'] = overall     
        results = results[results["overall"] == self.overallThesis]
        #st.write(results)

        # The first part of the originality score are number of arguments remaining unused, which shows relevance to the question.
        self.originalityscore = len(results)/2

        if len(results) == 0:
            ret_val = " Tell the user that they are no longer providing sufficiently answers to the posted questions. "
            ret_val += "Tell the user that topics they can explore related to the presented topic. "
            ret_val += "Then let them know what they have lernt " 
            with sidebar_container:
                st.write(f'Game over! Try again.')
                #st.write(f'Total score: {int(np.ceil(self.totalscore))}')
            self.gameOver=True

            return ret_val

        # Get the 10 best arguments which oppose the current stance.
        currentArguments= []
        for _,r in results.iterrows():    
            argumentCode = r["step"]
            currentArgumentsdf=self.arguments.get_arguments(argumentCode,'for loop')
            currentArguments = currentArguments + list(currentArgumentsdf['assistant'])
            if len(currentArguments) > 20:
                break
        
        # Remove that argument from the list of arguments
        #self.argumentsMade.append(argumentCode)
        #st.write(self.argumentsMade)

        # Add a score based on how original is compared to arguments made so far.
        #with sidebar_container:
           # part2 = 10*numberofarguments*(results.iloc[0]["similarities"]-similaritytoprevious)
           # self.originalityscore = min(10.0,self.originalityscore+max(part2,0.0))
            #st.write(f'Novelty: {int(np.ceil(self.originalityscore))}/{numberofarguments}')
            #self.totalscore = self.totalscore+np.ceil(self.originalityscore)
            #st.write(f'Total score: {int(np.ceil(self.totalscore))}')
    
        if len(currentArguments) == 0:
            ret_val = "Write at most one sentences prompting the user to gauge on their knowledge on loops. "
        else:
            ret_val = f"Here is a description of the questions you can ask, which is: {query}: \n\n"   
            for a in currentArguments:
                ret_val += a + "\n"
            ret_val += "\n\nWrite one (or at most two) sentences prompting the learners understading of the topic concepts based on their previous response. "
        
        ret_val += "In the first sentence, restate the user's response. In the second sentence, ask the user a question based on their knowledge in loops that will help them grasp the concept of loops."
        ret_val += "Always address the user directly as 'you' in the first person and write natural sounding lanaguage, with no headers. Write as if you are having a socratic conversation with the user."

        ''' if self.totalscore>=100:
            ret_val += "In addition to the above, also congratulate the user for getting the right answer. "
            ret_val += "They are the Lerning!!! "
            

        return ret_val*/'''
    
    def get_input(self):
        """
        Get input from streamlit."""
  
        if x := st.chat_input(placeholder=f"Please respond here"):
            if len(x) > 500:
                st.error(f"Your message is too long ({len(x)} characters). Please keep it under 500 characters.")

            self.handle_input(x)