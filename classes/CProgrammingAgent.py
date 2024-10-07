import openai
import pandas as pd

class CProgrammingAgent:
    def __init__(self, excel_file_path, api_key, model="gpt-3.5-turbo"):
        """
        Initialize the CProgrammingAgent.
        
        :param excel_file_path: Path to the Excel file that contains the conversation flow.
        :param api_key: OpenAI API key for using the GPT model.
        :param model: The GPT model to use, default is "gpt-3.5-turbo".
        """
        self.conversation_flow = pd.read_excel(excel_file_path)  # Load the Excel conversation flow
        self.api_key = api_key  # OpenAI API Key
        self.model = model  # GPT model to use
        self.step = 0  # Track the learner's progress
        self.topic = None  # Current topic, if necessary to track
        self.knowledge_level = "beginner"  # Initialize the learner's knowledge level
        openai.api_key = self.api_key  # Set the API key for OpenAI

    def get_agent_prompt(self):
        """Retrieve the agent's prompt based on the current step in the conversation flow."""
        if self.step < len(self.conversation_flow):
            return self.conversation_flow.loc[self.step, "Agent Prompt"]
        else:
            return "End of conversation flow."

    def get_gpt_response(self, learner_input, agent_prompt):
        """
        Send learner input and context to the GPT model and return the response.

        :param learner_input: The input provided by the learner.
        :param agent_prompt: The current prompt from the agent.
        :return: The GPT-generated response.
        """
        messages = [
            {"role": "system", "content": "You are a C programming instructor."},
            {"role": "user", "content": learner_input},
            {"role": "assistant", "content": agent_prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message['content'].strip()

    def handle_learner_input(self, learner_input):
        """
        Process learner input, check against expected input, and generate the next agent response.

        :param learner_input: The input provided by the learner.
        :return: The agent's response based on learner's input.
        """
        if learner_input.lower() == "quit":
            return "Ending conversation. Thank you!"

        # Check the expected input from the conversation flow
        expected_input = self.conversation_flow.loc[self.step, "Learner Input Example"]

        if learner_input == expected_input:
            # If learner input matches the expected example, move to the next step
            agent_response = self.conversation_flow.loc[self.step, "Agent Response"]
            self.step += 1
        else:
            # Generate a dynamic response using GPT
            agent_prompt = self.conversation_flow.loc[self.step, "Agent Prompt"]
            agent_response = self.get_gpt_response(learner_input, agent_prompt)
        
        return agent_response

    def start_conversation(self):
        """Start the conversation with the C programming agent."""
        print("Welcome to the C Programming Interactive Teaching Agent.")
        
        while self.step < len(self.conversation_flow):
            # Get current agent prompt
            agent_prompt = self.get_agent_prompt()
            print(f"Agent: {agent_prompt}")

            # Get learner input
            learner_input = input("Learner: ")

            # Handle the learner input and provide response
            agent_response = self.handle_learner_input(learner_input)
            print(f"Agent: {agent_response}\n")

            # Termination condition
            if learner_input.lower() == "quit":
                break


# Example usage:
# Initialize the agent with the Excel file and API key
agent = CProgrammingAgent(excel_file_path='c_programming_agent_conversation_flow.xlsx', 
                          api_key='your-api-key')

# Start the conversation with the agent
agent.start_conversation()
