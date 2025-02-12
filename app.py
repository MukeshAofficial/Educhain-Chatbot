import streamlit as st
import google.generativeai as genai
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport import requests
from httplib2 import Http
from oauth2client import file, client, tools
from educhain import Educhain, LLMConfig
from educhain.engines import qna_engine
from langchain_google_genai import ChatGoogleGenerativeAI  # Still need this for Educhain LLMConfig
import os
import json
import tempfile

# --- Configuration ---
SCOPES = [
    'https://www.googleapis.com/auth/forms.body',
]
FORM_TITLE = "QUIZ"

# --- Accessing Secrets ---
secrets_data = st.secrets["google"]  # Access all Google secrets

CLIENT_ID = secrets_data["installed"]["client_id"]
CLIENT_SECRET = secrets_data["installed"]["client_secret"]
PROJECT_ID = secrets_data["installed"]["project_id"]
AUTH_URI = secrets_data["installed"]["auth_uri"]
TOKEN_URI = secrets_data["installed"]["token_uri"]
AUTH_PROVIDER_X509_CERT_URL = secrets_data["installed"]["auth_provider_x509_cert_url"]
REDIRECT_URIS = secrets_data["installed"]["redirect_uris"]


# --- Function Schemas (using Python Dictionaries) ---
topic_schema = {'type': 'STRING', 'description': 'The topic for generating questions (e.g., Science, History).'}
num_questions_schema = {'type': 'INTEGER', 'description': 'The number of questions to generate.'}
custom_instructions_schema = {'type': 'STRING', 'description': 'Optional instructions for question generation.'}

mcq_params_schema = {
    'type': 'OBJECT',
    'properties': {
        'topic': topic_schema,
        'num_questions': num_questions_schema,
        'custom_instructions': custom_instructions_schema,
    },
    'required': ['topic', 'num_questions']
}

short_answer_params_schema = {
    'type': 'OBJECT',
    'properties': {
        'topic': topic_schema,
        'num_questions': num_questions_schema,
        'custom_instructions': custom_instructions_schema,
    },
    'required': ['topic', 'num_questions']
}

true_false_params_schema = {
    'type': 'OBJECT',
    'properties': {
        'topic': topic_schema,
        'num_questions': num_questions_schema,
        'custom_instructions': custom_instructions_schema,
    },
    'required': ['topic', 'num_questions']
}

fill_blank_params_schema = {
    'type': 'OBJECT',
    'properties': {
        'topic': topic_schema,
        'num_questions': num_questions_schema,
        'custom_instructions': custom_instructions_schema,
    },
    'required': ['topic', 'num_questions']
}

generate_form_params_schema = {
    'type': 'OBJECT',
    'properties': {
        'topic': topic_schema,
        'num_questions': num_questions_schema,
        'custom_instructions': custom_instructions_schema,
    },
    'required': ['topic', 'num_questions']
}

# Define FunctionDeclarations using dictionaries directly
function_declarations = [
    {
        'name': 'generate_mcq',
        'description': 'Generate multiple choice questions on a given topic.',
        'parameters': mcq_params_schema,
    },
    {
        'name': 'generate_short_answer',
        'description': 'Generate short answer questions on a given topic.',
        'parameters': short_answer_params_schema,
    },
    {
        'name': 'generate_true_false',
        'description': 'Generate true/false questions on a given topic.',
        'parameters': true_false_params_schema,
    },
    {
        'name': 'generate_fill_blank',
        'description': 'Generate fill in the blank questions on a given topic.',
        'parameters': fill_blank_params_schema,
    },
    {
        'name': 'generate_form',  # New Function
        'description': 'Generate a Google Form with multiple choice questions.',
        'parameters': generate_form_params_schema
    },
]

tools_config = {'function_declarations': function_declarations}  # Define tools config as dictionary


@st.cache_resource
def initialize_gemini_model(model_name, tools_config_):  # Renamed to tools_config_ to avoid shadowing
    """Initializes Gemini model directly with function calling config."""
    gemini_model = genai.GenerativeModel(  # Use genai.GenerativeModel
        model_name,
        tools=[tools_config_],  # Pass tools_config (which is now a dict) within a list
    )
    return gemini_model

def authenticate_google_api():
    """Authenticates with Google using OAuth2.
    Returns the authorization URL if authentication is needed, otherwise None."""
    creds = st.session_state.get("credentials", None)
    if creds and not creds.invalid:
        return None  # Already authenticated

    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmpfile:
            json.dump({"installed": {
                    "client_id": CLIENT_ID,
                    "project_id": PROJECT_ID,
                    "auth_uri": AUTH_URI,
                    "token_uri": TOKEN_URI,
                    "auth_provider_x509_cert_url": AUTH_PROVIDER_X509_CERT_URL,
                    "client_secret": CLIENT_SECRET,
                    "redirect_uris": REDIRECT_URIS
                }}, tmpfile)
            temp_file_path = tmpfile.name  # Get the path to the temp file

        # Pass the temporary file path to flow_from_clientsecrets
        flow = client.flow_from_clientsecrets(temp_file_path, SCOPES)

        # Clean up the temporary file
        os.remove(temp_file_path)

        auth_url = flow.step1_get_authorize_url() # Get the authorization URL
        return auth_url

    except Exception as e:
        st.error(f"Authentication error: {e}. {type(e).__name__} - {str(e)}")
        return None  # Authentication failed

def create_form_with_questions(creds, form_title, questions):
    """
    Creates a new Google Form with the given title and adds the provided questions.
    """
    try:
        form_service = build('forms', 'v1', http=creds.authorize(Http()))
        # Create the form with basic info.
        new_form = {
            'info': {
                'title': form_title,
            }
        }

        created_form = form_service.forms().create(body=new_form).execute()
        form_id = created_form.get('formId')

        # Build the batchUpdate request payload.
        requests = []
        for i, question_data in enumerate(questions.questions):
            if hasattr(question_data, 'options'):  # Multiple choice
                question = {
                    "createItem": {
                        "item": {
                            "title": question_data.question,
                            "questionItem": {
                                "question": {
                                    "choiceQuestion": {
                                        "type": "RADIO",  # For single-selection questions.
                                        "options": [{"value": opt} for opt in question_data.options]
                                    }
                                }
                            }
                        },
                        "location": {"index": i}
                    }
                }
                requests.append(question)
            else:  # Assuming short answer for now
                question = {
                    "createItem": {
                        "item": {
                            "title": question_data.question,
                            "questionItem": {
                                "question": {
                                    "textQuestion": {}
                                }
                            }
                        },
                        "location": {"index": i}
                    }
                }
                requests.append(question)

        if requests:
            body = {"requests": requests}
            form_service.forms().batchUpdate(formId=form_id, body=body).execute()

        st.success("Google Form created successfully!")
        form_url = f"https://docs.google.com/forms/d/{form_id}/viewform"
        st.markdown(f"Form URL: [Click here]({form_url})")
        return form_url

    except Exception as e:
        st.error(f"Error creating Google Form: {e}")
        return None

# --- Question Generation Functions (using Educhain's qna_engine) ---
# We still use the Educhain engine to actually generate questions, but the function *calling* is handled explicitly.
def generate_mcq(qna_engine_instance, topic, num_questions, custom_instructions=None):
    """Generates and displays Multiple Choice Questions."""
    st.info(f"Generating {num_questions} Multiple Choice Questions on topic: {topic}...")  # Added info message
    questions = qna_engine_instance.generate_questions(
        topic=topic,
        num=num_questions,
        question_type="Multiple Choice",
        custom_instructions=custom_instructions
    )
    return questions

def generate_short_answer(qna_engine_instance, topic, num_questions, custom_instructions=None):
    """Generates and displays Short Answer Questions."""
    st.info(f"Generating {num_questions} Short Answer Questions on topic: {topic}...")  # Added info message
    questions = qna_engine_instance.generate_questions(
        topic=topic,
        num=num_questions,
        question_type="Short Answer",
        custom_instructions=custom_instructions
    )
    return questions

def generate_true_false(qna_engine_instance, topic, num_questions, custom_instructions=None):
    """Generates and displays True/False Questions."""
    st.info(f"Generating {num_questions} True/False Questions on topic: {topic}...")  # Added info message
    questions = qna_engine_instance.generate_questions(
        topic=topic,
        num=num_questions,
        question_type="True/False",
        custom_instructions=custom_instructions
    )
    return questions

def generate_fill_blank(qna_engine_instance, topic, num_questions, custom_instructions=None):
    """Generates and displays Fill in the Blank Questions."""
    st.info(f"Generating {num_questions} Fill in the Blank Questions on topic: {topic}...")  # Added info message
    questions = qna_engine_instance.generate_questions(
        topic=topic,
        num=num_questions,
        question_type="Fill in the Blank",
        custom_instructions=custom_instructions
    )
    return questions

def generate_form(qna_engine_instance, topic, num_questions, custom_instructions=None):
    """Generates a Google Form with multiple-choice questions."""
    st.info(f"Generating a Google Form with {num_questions} questions on topic: {topic}...")

    creds = st.session_state.get("credentials", None) #Get the creds to not always generate authentication.
    if creds and not creds.invalid:
        questions = qna_engine_instance.generate_questions(
            topic=topic,
            num=num_questions,
            question_type="Multiple Choice",  # For now, only MCQs
            custom_instructions=custom_instructions
        )

        if creds:
            form_url = create_form_with_questions(creds, FORM_TITLE, questions)  # Call form creation
            return form_url #return the form
        else:
            st.error("Google Forms authentication failed.") #if it fails to create the form.
            return None #not able to create the form.
    else:
        auth_url = authenticate_google_api()
        if auth_url:
            return auth_url  # Return the authentication URL to display in chat.
        else:
            st.error("Failed to generate authentication URL.") #if it fails to generate authentication
            return None #not able to authenticate and not able to generate URL, return none.

def display_questions(questions):
    """Displays questions in Streamlit."""
    if questions and hasattr(questions, "questions"):
        for i, question in enumerate(questions.questions):
            st.subheader(f"Question {i + 1}:")
            if hasattr(question, 'options'):
                st.write(f"**Question:** {question.question}")
                st.write("Options:")
                for j, option in enumerate(question.options):
                    st.write(f"   {chr(65 + j)}. {option}")
                if hasattr(question, 'answer'):
                    st.write(f"**Correct Answer:** {question.answer}")
                if hasattr(question, 'explanation') and question.explanation:
                    st.write(f"**Explanation:** {question.explanation}")
            elif hasattr(question, 'keywords'):
                st.write(f"**Question:** {question.question}")
                st.write(f"**Answer:** {question.answer}")
                if question.keywords:
                    st.write(f"**Keywords:** {', '.join(question.keywords)}")
            elif hasattr(question, 'answer'):
                st.write(f"**Question:** {question.question}")
                st.write(f"**Answer:** {question.answer}")
                if hasattr(question, 'explanation') and question.explanation:
                    st.write(f"**Explanation:** {question.explanation}")
            else:
                st.write(f"**Question:** {question.question}")
                if hasattr(question, 'explanation') and question.explanation:
                    st.write(f"**Explanation:** {question.explanation}")
            st.markdown("---")
    else:
        st.error("No questions generated or invalid question format.")

# --- Function Dispatcher ---
function_map = {
    "generate_mcq": generate_mcq,
    "generate_short_answer": generate_short_answer,
    "generate_true_false": generate_true_false,
    "generate_fill_blank": generate_fill_blank,
    "generate_form": generate_form  # New Function
}

def main():
    """Main function to run the Educhain Question Generator chatbot with function calling in Streamlit."""
    st.set_page_config(page_title="Educhain Chatbot", page_icon="📚", layout="wide")
    st.title("📚 Educhain Question Generator Chatbot")

    with st.sidebar:
        api_key = st.text_input("Google API Key", type="password", help="Enter your Gemini API key here.")
        if not api_key:
            st.warning("Please enter your Google API Key in the sidebar.")
            st.stop()
        genai.configure(api_key=api_key)  # Configure API Key Globally

    model_options = {
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05": "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-pro-exp-02-05": "gemini-2.0-pro-exp-02-05",
    }
    model_name = st.selectbox("Select Model", options=list(model_options.keys()), format_func=lambda x: model_options[x])

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you generate questions?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        gemini_model = initialize_gemini_model(model_name, tools_config)  # Initialize Gemini Model with function calling

        educhain_client = Educhain(LLMConfig(custom_model=ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)))  # Initialize Educhain
        qna_engine_instance = educhain_client.qna_engine

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            function_called = False  # Flag to track if a function was called

            try:
                response = gemini_model.generate_content(prompt, stream=True)  # Stream response
                for chunk in response:
                    if hasattr(chunk.parts[0], 'function_call'):  # Function call detected in chunk
                        function_call = chunk.parts[0].function_call
                        function_name = function_call.name
                        arguments = function_call.args
                        full_response = f"Function Call: {function_name} with args: {arguments}"  # Simple text for UI
                        message_placeholder.markdown(full_response + "▌")  # Show function call info
                        function_called = True  # Set flag

                        if function_name in function_map:
                            question_generation_function = function_map[function_name]
                            function_result = question_generation_function(qna_engine_instance, **arguments) #Added

                            if function_result: #Added. If not authenticated it shows a URL. If authenticated it will show form URL
                                if "http" in function_result: #Added. Authentication or form url. If auth then:
                                    st.markdown(f"Please authenticate with Google: {function_result}") #Added.

                                else:
                                    st.markdown(f"Form created: {function_result}")#Added #if for is created it show this.

                            else:
                                st.error("Error occured. Not able to create question/authenticate") # Added if is not able to create questions/authenticate
                                st.stop()#added

                            function_called = True  # Redundant, but for clarity
                        else:
                            st.error(f"Error: Unknown function name '{function_name}' received from model.")
                            full_response = "Error processing function call."
                            message_placeholder.markdown(full_response)
                            function_called = True  # To prevent default text response handling
                        break  # Exit chunk streaming loop as function call is handled

                    else:  # Regular text response chunk
                        if not function_called:  # Only append if not already handled a function call
                            full_response += (chunk.text or "")
                            message_placeholder.markdown(full_response + "▌")  # Typing effect

                if not function_called:  # Finalize message if it was a text response (not function call)
                    message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "Error generating response."
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response if not function_called else "Function call processed. See questions below."})  # Store a simple message for function calls

if __name__ == "__main__":
    main()
