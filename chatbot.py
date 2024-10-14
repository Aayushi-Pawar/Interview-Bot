import uuid
import streamlit as st
from streamlit_chat import message
import pandas as pd
import speech_recognition as sr  # For speech-to-text
import random  # To shuffle questions
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load pre-trained sentence-transformer model for better answer comparison
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Helper function to load questions and answers from a CSV
def get_questions_and_answers(csv_file, domain):
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')  # Specify encoding
    filtered_df = df[df['Category'] == domain]  # Filter by domain/category if required
    return filtered_df[['Question', 'Answer']].values.tolist()  # Return a list of (question, answer) pairs

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

        if 'recording_started' not in st.session_state:
            st.session_state['recording_started'] = False

        self.session_state = st.session_state

    def select_domain(self) -> None:
        # Ask the user to select the domain of the interview
        if self.session_state['selected_domain'] is None:
            st.write("Please select the domain for the interview:")
            selected_domain = st.selectbox("Choose a domain:", ["General Programming", "Data Structures", "Languages and Frameworks", "Web Development"])
            if selected_domain:
                self.session_state['selected_domain'] = selected_domain
                self.prepare_questions(selected_domain)

    def prepare_questions(self, domain: str) -> None:
        # Load and shuffle questions and answers from CSV based on the domain
        questions_and_answers = get_questions_and_answers('SoftwareEngineering.csv', domain)
        random.shuffle(questions_and_answers)  # Shuffle the questions to make them random
        self.session_state['questions'] = [(qa[0], qa[1], self._generate_uuid()) for qa in questions_and_answers]

    def ask_question(self) -> None:
        # Display the current question
        if self.session_state['interview_step'] < len(self.session_state['questions']): 
            text, answer, key = self.session_state['questions'][self.session_state['interview_step']]
            message(text, key=key)
        else:
            st.write("Interview completed!")

    def get_answer(self) -> None:
        # Continuous buttons for start and stop recording
        if st.button("Start Recording") and not self.session_state['recording_started']:
            self.session_state['recording_started'] = True
            self.start_recording()

        if st.button("Next Question") and self.session_state['recording_started']:
            st.write("Recording stopped.")
            self.session_state['recording_started'] = False

    def start_recording(self) -> None:
        # Record the user's answer via microphone and convert it to text
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("Listening... Please speak.")
            audio = recognizer.listen(source)

        try:
            # Convert speech to text
            answer = recognizer.recognize_google(audio)
            st.write(f"You said: {answer}")
            
            # Keyword matching: Get expected answer and check if the keywords match
            correct_answer = self.session_state['questions'][self.session_state['interview_step']][1]
            if self.match_keywords(answer, correct_answer):
                st.write("Perfect match!")
            else:
                st.write("Wrong Answer/ Needs Improvement!")
            
            # Append the answer and proceed to the next question
            self.session_state['answers'].append((answer, self._generate_uuid()))
            self.session_state['interview_step'] += 1
            self.session_state['recording_started'] = False

        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand the audio. Please try again.")
            self.session_state['recording_started'] = False
        except sr.RequestError as e:
            st.write(f"Could not request results from the speech recognition service; {e}")
            self.session_state['recording_started'] = False

    def match_keywords(self, user_answer: str, correct_answer: str) -> bool:
        user_keywords = set(user_answer.lower().split())
        correct_keywords = set(correct_answer.lower().split())
        common_keywords = user_keywords.intersection(correct_keywords)

        # Consider a match if at least one keyword matches
        return len(common_keywords) > 0

    def compare_answers(self, user_answer: str, correct_answer: str) -> float:
        # Use sentence embeddings to compare the answers
        user_embedding = sentence_model.encode([user_answer])
        correct_embedding = sentence_model.encode([correct_answer])

        # Calculate cosine similarity
        similarity_score = cosine_similarity(user_embedding, correct_embedding)[0][0]
        return similarity_score
    
    def evaluate_candidate(self) -> str:
        # Generate an evaluation for the candidate based on their answers
        evaluation_report = []
        for (question, correct_answer, _), (user_answer, _) in zip(self.session_state['questions'], self.session_state['answers']):
            similarity = self.compare_answers(user_answer, correct_answer)
            evaluation_report.append(f"Question: {question}\nYour Answer: {user_answer}\nExpected Answer: {correct_answer}\nSimilarity Score: {similarity:.2f}\n")
    
        if len(self.session_state['questions']) > 0:
            overall_score = sum([self.compare_answers(user_answer, correct_answer) for (_, correct_answer, _), (user_answer, _) in zip(self.session_state['questions'], self.session_state['answers'])])
            average_score = overall_score / len(self.session_state['questions'])
            evaluation_summary = f"Overall similarity score: {average_score:.2f}. {'Well done!' if average_score > 0.75 else 'Needs Improvement.'}"
        else:
            evaluation_summary = "No questions were asked, so no evaluation is available."

        return "\n".join(evaluation_report) + evaluation_summary

    def display_past_questions_and_answers(self) -> None:
        # Display all past questions and answers
        for i in range(self.session_state['interview_step']):
            question_text, _, question_key = self.session_state['questions'][i]
            message(question_text, key=question_key)

            if i < len(self.session_state['answers']): 
                answer_text, answer_key = self.session_state['answers'][i]
                message(answer_text, is_user=True, key=answer_key)

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
                st.write(f"Evaluation Report:\n{evaluation}")
                self.session_state['interview_step'] += 1

    @staticmethod
    def _generate_uuid() -> str:
        return str(uuid.uuid4())

def create_bot() -> None:
    # Create and manage the InterviewBot instance
    bot = InterviewBot()

    if len(bot.session_state['questions']) == 0:
        message("Hello! I'm your interviewer bot. Please select a domain for the interview.", key="greeting")
    
    bot.execute_interview()

st.title("InterviewBot - AI Interview Chatbot")
create_bot()
