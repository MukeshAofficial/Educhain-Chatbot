import streamlit as st
from educhain import Educhain, LLMConfig
from educhain.engines import qna_engine
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import google.generativeai as genai
from googleapiclient.discovery import build  # Google Forms API
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport import requests
from httplib2 import Http
from oauth2client import file, client, tools

# --- Configuration ---
SCOPES = [
    'https://www.googleapis.com/auth/forms.body',
]
CREDENTIALS_FILE = 'config.json'  # Used only to temporarily store the secrets in the file
FORM_TITLE = "AI Generated Quiz"

# --- Function Schemas (using Python Dictionaries) ---
topic_schema = {'type': 'STRING', 'description': 'The topic for generating questions (e.g., Science, History).'}
num_questions_schema = {'type': 'INTEGER', 'description': 'The number of questions to generate.'}
custom_instructions_schema = {'type': 'STRING', 'description': 'Optional instructions for question generation.'}
question_type_schema = {'type': 'STRING', 'enum': ['Multiple Choice', 'Short Answer', 'True/False', 'Fill in the Blank'], 'description': 'The type of questions to generate.'}  # ADDED QUESTION TYPE

question_params_schema = {
    'type': 'OBJECT',
    'properties': {
        'topic': topic_schema,
        'num_questions': num_questions_schema,
        'custom_instructions': custom_instructions_schema,
        'question_type': question_type_schema,
    },
    'required': ['topic', 'num_questions', 'question_type']
}

generate_form_params_schema = {
    'type': 'OBJECT',
    'properties': {
        'topic': topic_schema,
        'num_questions': num_questions_schema,
        'question_type': question_type_schema,
        'custom_instructions': custom_instructions_schema
    },
    'required': ['topic', 'num_questions','question_type']
}

# Define FunctionDeclarations using dictionaries directly
function_declarations = [
    {
        'name': 'generate_questions',
        'description': 'Generate questions on a given topic and type.',
        'parameters': question_params_schema,
    },
    {
        'name': 'generate_form',  # New Function
        'description': 'Generate a Google Form with AI generated questions.',
        'parameters': generate_form_params_schema
    },
]

tools_config = {'function_declarations': function_declarations}

# --- Initialize Educhain with Gemini Model ---
@st.cache_resource
def initialize_gemini_model(api_key):
    gemini_model = genai.GenerativeModel(model_name='gemini-pro', tools=[tools_config])
    return gemini_model

@st.cache_resource
def initialize_educhain(api_key):
    if not api_key:
        return None
    gemini_model = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=api_key)
    llm_config = LLMConfig(custom_model=gemini_model)
    return Educhain(llm_config)

# --- Google Forms API Functions ---
def authenticate_google_api():
    """Authenticates with Google using OAuth2.
    Uses an existing token in 'storage.json' if available.
    Otherwise, starts the OAuth2 flow.
    Loads client secrets from Streamlit Secrets for deployment.
    """
    store = file.Storage('storage.json')
    creds = store.get()

    if not creds or creds.invalid:
        try:
            # Load secrets from Streamlit Secrets
            client_secrets = st.secrets["google_client_secrets"]

            # Ensure client_secrets is a dictionary
            if isinstance(client_secrets, str):
                client_secrets = json.loads(client_secrets)

            # Create a flow object and run it
            flow = client.flow_from_clientsecrets(
                CREDENTIALS_FILE,  # Not used for loading, only for type hint
                SCOPES
            )

            # Write client secrets to CREDENTIALS_FILE only if it's not a file path
            with open(CREDENTIALS_FILE, 'w') as f:
                json.dump(client_secrets, f)  # Writing as dictionary

            creds = tools.run_flow(flow, store) # Authenticate with the config
            return creds  # Return credentials

        except KeyError:
            st.error("Google Client Secrets not found in Streamlit Secrets. Please configure your secrets.")
            return None
        except Exception as e:
            st.error(f"Authentication error: {e}. Ensure config.json is correct and accessible. Error: {e}")
            return None
    else:
        return creds

def create_form_with_questions(creds, form_title, questions, question_type):  # Added question_type
    """Creates a new Google Form with the given title and adds the provided questions."""
    try:
        form_service = build('forms', 'v1', http=creds.authorize(Http()))
        new_form = {'info': {'title': form_title}}
        created_form = form_service.forms().create(body=new_form).execute()
        form_id = created_form.get('formId')

        requests = []
        for i, question_data in enumerate(questions.questions):
            item = {
                "createItem": {
                    "item": {
                        "title": question_data.question,
                        "questionItem": {
                            "question": {}  # Base question object
                        }
                    },
                    "location": {"index": i}
                }
            }

            if question_type == "Multiple Choice" and hasattr(question_data, 'options'):  # Multiple Choice

                item["createItem"]["item"]["questionItem"]["question"]["choiceQuestion"] = {
                            "type": "RADIO",
                            "options": [{"value": opt} for opt in question_data.options]
                        }

            else:  # Other question types use textQuestion
                item["createItem"]["item"]["questionItem"]["question"]["textQuestion"] = {}
            requests.append(item)

        body = {"requests": requests}
        form_service.forms().batchUpdate(formId=form_id, body=body).execute()

        st.success("Google Form created successfully!")
        form_url = f"https://docs.google.com/forms/d/{form_id}/viewform"
        st.markdown(f"Form URL: [Click here]({form_url})")
        return form_url

    except Exception as e:
        st.error(f"Error creating Google Form: {e}")
        return None

def generate_form(qna_engine_instance, topic, num_questions, question_type, custom_instructions=None):  # Included question_type
    """Generates a Google Form with AI-generated questions."""
    st.info(f"Generating a Google Form with {num_questions} {question_type} questions on topic: {topic}...")
    questions = qna_engine_instance.generate_questions(
        topic=topic,
        num=num_questions,
        question_type=question_type,
        custom_instructions=custom_instructions
    )

    creds = authenticate_google_api()
    if creds:
        form_url = create_form_with_questions(creds, FORM_TITLE, questions, question_type)  # Pass the question_type
        return form_url
    else:
        st.error("Google Forms authentication failed.")
        return None

# --- Educhain Question Generation Functions ---
def generate_questions(qna_engine_instance, topic, num_questions, question_type, custom_instructions=None):  # MODIFIED FUNCTION
    """Generates questions of a specific type."""
    st.info(f"Generating {num_questions} {question_type} Questions on topic: {topic}...")
    questions = qna_engine_instance.generate_questions(
        topic=topic,
        num=num_questions,
        question_type=question_type,
        custom_instructions=custom_instructions
    )
    return questions

# --- Utility Function to Display Questions ---
def display_questions(questions):
    if questions and hasattr(questions, "questions"):
        for i, question in enumerate(questions.questions):
            st.subheader(f"Question {i + 1}:")

            if hasattr(question, 'options'):  # Multiple Choice
                st.write(f"**Question:** {question.question}")
                st.write("Options:")
                for j, option in enumerate(question.options):
                    st.write(f"   {chr(65 + j)}. {option}")
                if hasattr(question, 'answer'):
                    st.write(f"**Correct Answer:** {question.answer}")
                if hasattr(question, 'explanation') and question.explanation:
                    st.write(f"**Explanation:** {question.explanation}")

            else:  # Short Answer, True/False, Fill in the Blank
                st.write(f"**Question:** {question.question}")
                if hasattr(question, 'answer'):
                    st.write(f"**Answer:** {question.answer}")
                if hasattr(question, 'explanation') and question.explanation:
                    st.write(f"**Explanation:** {question.explanation}")
                if hasattr(question, 'keywords') and question.keywords:  # Display keywords if present
                    st.write(f"**Keywords:** {', '.join(question.keywords)}")

            st.markdown("---")
    else:
        st.error("No questions generated or invalid question format.")

# --- Function Dispatcher ---
function_map = {
    "generate_questions": generate_questions,
    "generate_form": generate_form
}

def main():
    """Main function to run the Educhain Question Generator chatbot with function calling in Streamlit."""
    st.set_page_config(page_title="Educhain Chatbot", page_icon="📚", layout="wide")

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input("Google API Key", type="password")
        st.markdown("**Powered by** [Educhain](https://github.com/satvik314/educhain)")

    # Initialize Educhain client
    if api_key:
        educhain_client = initialize_educhain(api_key)
        if educhain_client:
            qna_engine = educhain_client.qna_engine
            gemini_model = initialize_gemini_model(api_key)  # Initialize Gemini with function calling
        else:
            st.error("Failed to initialize Educhain. Please check your API key.")
            return  # Prevents app from crashing
    else:
        st.warning("Please enter your Google API Key in the sidebar to continue.")
        return  # Prevents app from crashing

    # Main UI
    st.title("📚 Educhain Question Generator Chatbot")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you generate questions or create a Google Form?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            function_called = False

            try:
                response = gemini_model.generate_content(prompt, stream=True)
                for chunk in response:
                    if hasattr(chunk.parts[0], 'function_call'):
                        function_call = chunk.parts[0].function_call
                        function_name = function_call.name
                        arguments = function_call.args
                        full_response = f"Function Call: {function_name} with args: {arguments}"
                        message_placeholder.markdown(full_response + "▌")
                        function_called = True

                        if function_name in function_map:
                            function_result = function_map[function_name](qna_engine, **arguments)

                            if function_name == "generate_questions":  # Display questions

                                message_placeholder.empty()
                                display_questions(function_result)
                            elif function_name == "generate_form":  # Function generated a google form so display the link.
                                if function_result:  # Check if the response is a non-empty string.
                                    full_response = f"Google Form created: [Click here]({function_result})"
                                else:
                                    full_response = "Failed to create Google Form."
                                message_placeholder.markdown(full_response)
                                full_response = "Generating Form"

                        else:
                            st.error(f"Error: Unknown function name '{function_name}' received from model.")
                            full_response = "Error processing function call."
                            message_placeholder.markdown(full_response)

                        break

                    else:
                        if not function_called:
                            full_response += (chunk.text or "")
                            message_placeholder.markdown(full_response + "▌")

                if not function_called:
                    message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                full_response = "Error generating response."
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response if not function_called else "Function call processed. See questions below."})

if __name__ == "__main__":
    main()
