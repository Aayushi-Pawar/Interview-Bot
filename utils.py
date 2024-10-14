import openai
import re
from typing import List
from openai.error import InvalidRequestError
from config import Parameters
from shared import get_questions




def get_completion(complete_prompt: str) -> str:
    """
    Send a message to the OpenAI API to get a response.
    """
    try:
        messages = [{"role": "user", "content": complete_prompt}]
        response = openai.ChatCompletion.create(model=Parameters.MODEL, messages=messages,
                                                temperature = 0)
        return response.choices[0].message["content"]
    
    except InvalidRequestError as e:
        print(f"Encountered an error: {e}")
        return "Error: Input too long for model."


# utils.py
def get_questions(concatenated_questions):
    from config import Parameters  # Import here to avoid circular import
    
    # Simulate fetching questions based on concatenated_questions
    # You should replace this with your actual logic to generate questions
    return [f"{concatenated_questions} Question {i + 1}" for i in range(5)]  # Example: 5 questions
