import uuid
import streamlit as st
from streamlit_chat import message
from utils import get_questions
from config import Parameters
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import speech_recognition as sr  # For speech-to-text
from shared import Parameters


class BertModel:
    def __init__(self):
        # Load pre-trained BERT model and tokenizer from Hugging Face
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Adjust for task

    def classify_answer(self, question, answer):
        # Use this method to compare or classify the answer
        inputs = self.tokenizer(question, answer, return_tensors="pt", max_length=512, truncation=True, padding=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1)
        return predicted_class.item()


class InterviewBot:
    def __init__(self) -> None:
        # Initialize session state for interview progress
        if 'questions' not in st.session_state:
            st.session_state['questions'] = []

        if 'answers' not in st.session_state:
            st.session_state['answers'] = []

        if 'interview_step' not in st.session_state:
            st.session_state['interview_step'] = 0

        if 'selected_domain' not in st.session_state:
            st.session_state['selected_domain'] = None

        self.session_state = st.session_state
        self.bert_model = BertModel()  # Initialize BERT model

    def select_domain(self) -> None:
        # Ask the user to select the domain of the interview
        if self.session_state['selected_domain'] is None:
            st.write("Please select the domain for the interview:")
            selected_domain = st.selectbox("Choose a domain:", ["Software Engineering", "Data Science", "Marketing", "Sales"])
            if selected_domain:
                self.session_state['selected_domain'] = selected_domain
                self.prepare_questions(selected_domain)

    def prepare_questions(self, domain: str) -> None:
        # Prepare interview questions based on the selected domain
        concatenated_questions = Parameters.QUESTIONS_PROMPT.format(job_description=f"Interview for {domain} role")
        questions = get_questions(concatenated_questions)  # Fetch domain-specific questions

        self.session_state['questions'] = [(question, self._generate_uuid()) for question in questions]

    def ask_question(self) -> None:
        # Display the current question
        text, key = self.session_state['questions'][self.session_state['interview_step']]
        message(text, key=key)

    def get_answer(self) -> None:
        # Record the user's answer via microphone and convert it to text
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening for your answer... Please speak now.")
            audio = recognizer.listen(source)

        try:
            # Convert speech to text
            answer = recognizer.recognize_google(audio)
            st.write(f"You said: {answer}")
            
            # Append the answer and proceed to the next question
            self.session_state['answers'].append((answer, self._generate_uuid()))
            self.session_state['interview_step'] += 1

        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand the audio. Please try again.")
        except sr.RequestError as e:
            st.write(f"Could not request results from the speech recognition service; {e}")

    def display_past_questions_and_answers(self) -> None:
        # Display all past questions and answers
        for i in range(self.session_state['interview_step']):
            question_text, question_key = self.session_state['questions'][i]
            message(question_text, key=question_key)

            if i < len(self.session_state['answers']): 
                answer_text, answer_key = self.session_state['answers'][i]
                message(answer_text, is_user=True, key=answer_key)

    def evaluate_candidate(self) -> str:
        # Generate an evaluation for the candidate based on their answers
        interview_text = "".join([f"Question: {question}\nAnswer: {answer}\n" for (question, _), (answer, _) in zip(self.session_state['questions'], self.session_state['answers'])])
        
        # Use BERT for classification of answers
        scores = []
        for (question, _), (answer, _) in zip(self.session_state['questions'], self.session_state['answers']):
            score = self.bert_model.classify_answer(question, answer)
            scores.append(score)

        evaluation = f"Evaluation: Based on BERT model classification, your responses seem {'adequate' if sum(scores) > len(scores) // 2 else 'inadequate'}."
        return evaluation

    def execute_interview(self) -> None:
        # Execute the interview process
        if self.session_state['selected_domain'] is None:
            self.select_domain()  # Prompt user to select domain
        else:
            self.display_past_questions_and_answers()

            if self.session_state['interview_step'] < len(self.session_state['questions']):
                self.ask_question()
                self.get_answer()

            elif self.session_state['interview_step'] == len(self.session_state['questions']):
                evaluation = self.evaluate_candidate()
                st.write(f"BERT model's evaluation: {evaluation}")
                self.session_state['interview_step'] += 1

    @staticmethod
    def _generate_uuid() -> str:
        return str(uuid.uuid4())


def create_bot() -> None:
    # Create and manage the InterviewBot instance
    bot = InterviewBot()

    if len(bot.session_state['questions']) == 0:
        message("Hello! I'm your interviewer bot powered by BERT. Please select a domain for the interview.", key="greeting")
    
    bot.execute_interview()


st.title("InterviewBot - AI Interview Chatbot")
create_bot()
