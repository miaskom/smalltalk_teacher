import streamlit as st
from audiorecorder import audiorecorder
from io import BytesIO
from dotenv import dotenv_values
from openai import OpenAI

from pydantic import BaseModel
import instructor  # nakładka na OpenAI pozwalająca na zwracanie przez OpenAI danych w strukturze  In-STRUCT-or dziedziczącej z BaseModel z biblioteki PyDantic
                   # np. na potrzeby zdefiniowania struktury response'u JSONa z OpenAI 
import pandas as pd
from pathlib import Path
import random

# EMBEDDING_MODEL = "text-embedding-3-large"
# EMBEDDING_DIM = 3072

EMBEDDING_MODEL = "text-embedding-3-small" #nazwa modelu dla embedding'sów w OpenAI
EMBEDDING_DIM = 1536  #długość wektora dla embedingsów z modelu small, tu definiujemy z góry jao stałą bo przy tworzeniu kolekcji w QDrant
                     #trzeba podać z góry wielkość wektora jakim będą opisywane poszczególne obiekty w kolekcji i tej wielkości trzeba sie potem trzymać 

GPT_MODEL = "gpt-4o-mini"
T2SPEECH_MODEL = "tts-1"
T2SPEECH_VOICE_ENG = "echo"
T2SPEECH_VOICE_ITA = "nova"
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
MP3_PATH = Path("mp3") 

class Questions(BaseModel):
    question_1: str
    question_2: str
    question_3: str
    question_4: str
    question_5: str
    question_6: str
    question_7: str
    question_8: str
    question_9: str
    question_10: str
    question_11: str
    question_12: str
    question_13: str
    question_14: str
    question_15: str
   


#
# OpenAI API key protection - zabezpieczenie klucza API-KEY
#
@st.cache_resource
def get_openai_API_KEY():
    env = dotenv_values(".env")  # plik .env zawiera klucz OPENAI_API_KEY
    # jak apka uruchamia się lokalnie to ma dostęp do pliku .env i wczyta API KEY z pliku, ale uruchamiana ze Streamlit Cloud nie ma dostępu do pliku .env
    # i trzeba te dane pobrać z st.secrets i nadpisać klucz wartością pobrana z st.secrets (na podst. advanced settings w konfig apki na streamlit)
    # https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
    if ("OPENAI_API_KEY" in env):
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    elif ('OPENAI_API_KEY' in st.secrets):
        env['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:  #ale jeśli nie ma klucza w env ani w st.secrets to poproś usera, żeby wpisał swój klucz
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password") #input text typu hasło (zamaskowane)
        if st.session_state["openai_api_key"]:
            st.rerun()

    if not st.session_state.get("openai_api_key"): #jeśli nie ma nadal klucza
        st.stop()  #to zatrzymaj apke, nie pozwól iść dalej


# dekorator st.cache_resources pozwala na cache'owanie tego co zrobi funkcja wewnętrzna
# i nie będzie to musiało się wykonywać przy każdym odświeżaniu UI (po każdej zmianie na formatce)
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.session_state.get("openai_api_key"))


#
# funkcja tenerująca mp3 text-2-speech
def generate_speech(text, output_audio_path):
    voice = T2SPEECH_VOICE_ENG
    if st.session_state["smalltalk_entry"].lang=="Włoski":
        voice = T2SPEECH_VOICE_ITA

    response = get_openai_client().audio.speech.create(
        model=T2SPEECH_MODEL,
        voice=voice,   # wybrany głos - w dokumentacji (https://platform.openai.com/docs/guides/text-to-speech) można znaleźć spis innych głosów, np. alloy, echo
        response_format="mp3",
        input=text,
    )
    response.write_to_file(output_audio_path)
    return output_audio_path

#
# funkcja do transkrypcji audio -> text
def transcribe_audio_to_text(audio_path):
    with open(audio_path, "rb") as f:
        transcript = get_openai_client().audio.transcriptions.create(
            file=f,
            model=AUDIO_TRANSCRIBE_MODEL, # model speech-to-text
            response_format="verbose_json",  # chcemy format odpowiedzi z rozpoznanym językiem
        )
    return transcript.text  # zwróć tylko sam tekst transkrypcji  
			# transcript.language   - zwraca rozpoznany język w jakim jest nagranie



#funkcja wysyla zapytanie do openAI i zwraca odpowiedź jako JSON o strukturze zdefiniowanej w klasie Questions
def get_chatbot_response(system_prompt, user_prompt): #, pamiec_chata):
    #każdy wpis dla OpenAI musi mieć min. 2 pola
    #  role: system/user/assistant <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!
    #  content: treść zapytania, polecnia, prompta
    messages_4_chat=[  #wysylamy do chata kilka wiadomosci: 1 w roli system i kilka zapamietanych wczesniejszych w roli asystenta/usera
            { #rola asystenta daje wstepne wytyczne -ustala jego "osobowość"
                "role": "system",
                "content": system_prompt #opis osobowości wybrany przez usera
             }
        ]
    # #dodaj to tablicy wiadomości zapamietane wpisy role+content w wcześniejszych zapytań (z pamici wpisów) aż do bieżącego(najnowszego)
    # for wpis in pamiec_chata:
    #         messages_4_chat.append({"role": wpis["rola"], "content": wpis["wpis"]})
    # dodaj wiadomość użytkownika
    messages_4_chat.append({"role": "user", "content": user_prompt})

    #wywołaj chatGPT i przekaź mu zapytanie z zapamiętaną historią
    instructor_openai_client = instructor.from_openai(get_openai_client())
    queries = instructor_openai_client.chat.completions.create(
        model= GPT_MODEL, #stała z nazwą modelu jakiego użyć - zdefiniowana u góry na początku pliku
        messages= messages_4_chat,
        response_model= Questions
    )
    
    # Konwersja instancji na słownik i losowy wybór pytania
    random_question = random.choice( list(queries.dict().values()) )

    #zwróć odpowiedź chata
    return random_question  # model_dump pozwala zrzucić odpowiedź JSON do struktury Pytonowej w celu wyświetlenia/użycia


# Funkcja do odczytania aktualnego licznika ID konwersacji z pliku
def get_current_smalltalk_id():
    try:
        with open('curr_id.txt', 'r') as file:
            current_id = int(file.read())
    except FileNotFoundError:
        current_id = 1000 # Jeśli plik nie istnieje, zaczynamy od 1000
    return current_id


# Funkcja do wygen3rowania nowego ID konwersacji (+zapamiętanie w pliku)
def get_new_smalltalk_id():
    new_id=get_current_smalltalk_id()+1
    with open('curr_id.txt', 'w') as file:
        file.write(str(new_id))
    return new_id






#
# funkcja generująca nowy small-talk
def generate_new_smalltalk( lang: str,
                            level: str,
                            subject: str,
                            simulate_ai: bool = False):
    st.write(f'generuj smalltalk dla: język: {lang}, poziom: {level}, temat: {subject}, symulator: {simulate_ai}')
    
    st.session_state["smalltalk_entry"].id=get_new_smalltalk_id()
    st.session_state["smalltalk_entry"].lang=lang
    st.session_state["smalltalk_entry"].level=level
    st.session_state["smalltalk_entry"].subject=subject
    
    if simulate_ai:    
        st.session_state["smalltalk_entry"].system_prompt_txt=""
        st.session_state["smalltalk_entry"].smalltalk_prompt_txt=""
        st.session_state["smalltalk_entry"].ai_query_txt="Pogadajmy o pogodzie...???"
        st.session_state["smalltalk_entry"].ai_query_mp3_path = ""
        st.session_state["smalltalk_entry"].user_answer_txt = "there'a fire on the sky"    # odpiowiedź usera na pytanie AI - do oceny przez "nauczyceila"
        st.session_state["smalltalk_entry"].user_answer_mp3_path = "mp3/example.mp3"
        st.session_state["smalltalk_entry"].ai_evaluation_txt = "bezbłędnie"  # ocena wypowiedzi usera dokonana przez nauczyciela (API) 
        st.session_state["smalltalk_entry"].ai_evaluation_mp3_path = "mp3/example.mp3"

        st.session_state["smalltalk_entry"].ai_query_mp3_played = False #czy odtworzono automatycznie 
        st.session_state["smalltalk_entry"].user_answer_mp3_played = False
        st.session_state["smalltalk_entry"].ai_evaluation_mp3_played = False    
        return
    
    st.session_state["smalltalk_entry"].system_prompt_txt=f'Jako nauczyciel prowadzisz krótkie konwersacje z uczniem w języku {st.session_state["smalltalk_entry"].lang} na poziomie {st.session_state["smalltalk_entry"].level}. Uczeń posługuje się językiem polskim.'
    st.session_state["smalltalk_entry"].smalltalk_prompt_txt=f'Przygotuj test składający się z 15 różnych pytań w języku {st.session_state["smalltalk_entry"].lang}m dla mnie jako ucznia na poziomie: {st.session_state["smalltalk_entry"].level}, z dziedziny: {st.session_state["smalltalk_entry"].subject}. Te pytania mają być różne i nie powtarzające się. I nie pytaj "What is money?"'
    st.session_state["smalltalk_entry"].ai_query_txt= get_chatbot_response(system_prompt = st.session_state["smalltalk_entry"].system_prompt_txt,
                                                                           user_prompt   = st.session_state["smalltalk_entry"].smalltalk_prompt_txt)
    st.session_state["smalltalk_entry"].ai_query_mp3_path = ""
    st.session_state["smalltalk_entry"].user_answer_txt = ""    # odpiowiedź usera na pytanie AI - do oceny przez "nauczyceila"
    st.session_state["smalltalk_entry"].user_answer_mp3_path = ""
    st.session_state["smalltalk_entry"].ai_evaluation_txt = ""  # ocena wypowiedzi usera dokonana przez nauczyciela (API) 
    st.session_state["smalltalk_entry"].ai_evaluation_mp3_path = ""

    st.session_state["smalltalk_entry"].ai_query_mp3_played = False #czy odtworzono automatycznie 
    st.session_state["smalltalk_entry"].ai_evaluation_mp3_played = False    
    
    return


#
# funkcja generująca nowy small-talk
def evaluate_user_response (user_response):
    #każdy wpis dla OpenAI musi mieć min. 2 pola
    #  role: system/user/assistant <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<!!!!!!!!!!!!
    #  content: treść zapytania, polecnia, prompta
    messages_4_chat=[  #wysylamy do chata kilka wiadomosci: 1 w roli system i kilka zapamietanych wczesniejszych w roli asystenta/usera
            { #rola asystenta daje wstepne wytyczne -ustala jego "osobowość"
                "role": "system",
                "content": st.session_state["smalltalk_entry"].system_prompt_txt
             }
        ]
    # #dodaj to tablicy wiadomości wszystkie wiadomości przesyłane do openAI w tej konwersacji
    messages_4_chat.append({"role": "user", "content": st.session_state["smalltalk_entry"].smalltalk_prompt_txt})
    messages_4_chat.append({"role": "assistant", "content": st.session_state["smalltalk_entry"].ai_query_txt})
    new_prompt = f'Napisz mi czy moja poniższa wypowiedź w języku {st.session_state["smalltalk_entry"].lang}m jest poprawna językowo i gramatycznie, czy pasuje do zadanego pytania. Jeśli popełniłem jakieś błędy lub użyłem złego języka to napisz co zrobiłem źle i to uzasadnij. Zaproponuj mi poprawną formę odpowiedzi. Moja odpowiedź to:  {user_response}'
    messages_4_chat.append({"role": "user", "content": new_prompt})
                           
    #wywołaj chatGPT i przekaź mu zapytanie z zapamiętaną historią
    response = get_openai_client().chat.completions.create(
        model= GPT_MODEL, #stała z nazwą modelu jakiego użyć - zdefiniowana u góry na początku pliku
        messages= messages_4_chat
    )
    #zwróć odpowiedź
    return response.choices[0].message.content


