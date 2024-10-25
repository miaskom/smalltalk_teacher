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
from hashlib import md5 # do obliczania skrótu MD5 (128 bit = 32 znaki):

#from qdrant_client import QdrantClient
#from qdrant_client.models import PointStruct, Distance, VectorParams 

# moje funkcje dodatkowe przerzucone do oddzielnych plików (w celu porządkowym)
from smalltalks_ai_lib import *
from smalltalks_db_lib import *


# definiujemy klase z danymi pojedynczej konwersacji
class SmallTalk_entry:
    def __init__(self):   # domyślny konstruktor inicjujący wszystkie zmienne pustym stringiem
        self.id = 0
        self.lang = ""
        self.level = ""
        self.subject= ""
        self.system_prompt_txt = ""
        self.smalltalk_prompt_txt = ""
        
        self.ai_query_txt = ""     # wygenerowane przez "nauczyciela" AI pytanie dla usera w wybranym jęz obcym
        self.ai_query_mp3_path = ""
        self.ai_query_mp3_played = False #czy odtworzono automatycznie 

        self.user_answer_txt = ""    # odpiowiedź usera na pytanie AI - do oceny przez "nauczyceila"
        self.user_answer_mp3_path = ""
        self.user_answer_mp3_MD5 = "" #skrót MD5 do porównania czy aktualnie trzymane w pamieci audiorecorder nagranie jest tym samym co już poprzenio nagrane

        self.ai_evaluation_txt = ""  # ocena wypowiedzi usera dokonana przez nauczyciela (API) 
        self.ai_evaluation_mp3_path = ""
        self.ai_evaluation_mp3_played = False #czy odtworzono automatycznie 

    def get_conv_text(self):
        #zwraca treści konwersjacji na potrzeby utworzenie indeksu wektorowego (embedings_ do zapisania w bazie danych QDRANT
        txt = f'{self.ai_query_txt} : {self.user_answer_txt} : {self.ai_evaluation_txt}'
        return txt

    #metoda do zrzucenia całego obiektu do JSONa (w celu wgrania do bazy Qdrant)    
    def to_dict(self):
        return {
            "id": self.id,
            "lang": self.lang,
            "level": self.level,
            "subject": self.subject,
            "system_prompt_txt": self.system_prompt_txt,
            "smalltalk_prompt_txt": self.smalltalk_prompt_txt,
            "ai_query_txt": self.ai_query_txt,
            "ai_query_mp3_path": self.ai_query_mp3_path,
            "ai_query_mp3_played": self.ai_query_mp3_played,
            "user_answer_txt": self.user_answer_txt,
            "user_answer_mp3_path": self.user_answer_mp3_path,
            "user_answer_mp3_MD5": self.user_answer_mp3_MD5,
            "ai_evaluation_txt": self.ai_evaluation_txt,
            "ai_evaluation_mp3_path": self.ai_evaluation_mp3_path,
            "ai_evaluation_mp3_played": self.ai_evaluation_mp3_played,
        }




# ustal i zwróć ścieżkę z nazwa dla pliku MP3 dla konwersjaci o podanym ID oraz dla typu zawartości nagrania 
def get_mp3_filepath(id, content_type):
    allowed_content_types = ["ai_query", "user_answer", "ai_evaluation", "ai_alt_example"]
    
    # Walidacja typu zawartości
    if content_type not in allowed_content_types:
        raise ValueError(f"Nieprawidłowy content_type: {content_type}. Dopuszczalne wartości to {allowed_content_types}.")

    fname=str(id)
    if content_type=="ai_query":
        fname=fname+"-01.mp3"
    elif content_type=="user_answer":
        fname=fname+"-02.mp3"
    elif content_type=="ai_evaluation":
        fname=fname+"-03.mp3"
    elif content_type=="ai_alt_example":
        fname=fname+"-04.mp3"
    return str(MP3_PATH/fname)




db_env = dotenv_values(".db_env")

#
# Funkcja do przełączania stron
#
def navigate_to(page):
    st.session_state.page = page
    st.rerun()  # Przeładuj stronę



#
#content sidebar'a z prawej strony 
#
def sidebar():
   #Dodatkowe elementy nawigacji, np. menu nawigacyjne
    st.header("Small-Talks")

    if st.session_state.page == "Page 1":
        if st.button("Szukaj konwersacji", key=104):
            navigate_to("Page 2")
    elif st.session_state.page == "Page 2":
        if st.button("Wróć do rozmówki", key=103):
            navigate_to("Page 1")


    st.divider()

    st.header(":blue[Nowa konwersacja]")
    st.subheader("Wybierz język i poziom rozmowy")
    lang = st.selectbox("Język", ['Angielski', 'Włoski'])
    level = st.selectbox("Poziom", ['początkujący od zera', 'początkujacy', 'średniozaawansowany', 'zaawansowany'])
    subject = st.selectbox("Temat", ['finanse', 'jedzenie', 'restauracja', 'zakupy', 'polityka', 'biznes', 'gospodarka'])
    simulate_ai = st.checkbox("Symulacja AI (bez wywołań OpenAI) - 4 test only")

    # Wstawiamy CSS dla przycisku
    st.markdown("""
        <style>
        div.stButton > button{
            background-color: green;  /* Kolor tła przycisku */
            height: 40px;  /* Dwukrotnie większa wysokość */
        }
        </style>
        """, unsafe_allow_html=True)

    if st.button("Let's talk about...", use_container_width=True): 
            generate_new_smalltalk(lang, level, subject, simulate_ai)
            if st.session_state.page != "Page 1":
                navigate_to("Page 1")
    

#
#content strony ze smalltalkiem
#
def page_1():
    st.title("Small-Talk Teacher")
    if st.session_state["smalltalk_entry"].ai_query_txt == "":
        st.header(f'🧑‍🏫 _:blue[Najpierw wybierz język, poziom i temat na który porozmawiamy]_', divider="blue")
    else:
        st.header(f'🧑‍🏫 _:blue[{st.session_state["smalltalk_entry"].ai_query_txt}]_', divider="blue")
        col1, col2 = st.columns( [3, 7] )
        with col1:
            if st.session_state["smalltalk_entry"].ai_query_mp3_path=="":
                if st.button("Przeczytaj", key="readQueryBtn"):
                    st.session_state["smalltalk_entry"].ai_query_mp3_path = generate_speech(text=st.session_state["smalltalk_entry"].ai_query_txt, 
                                                                                            output_audio_path= get_mp3_filepath(st.session_state["smalltalk_entry"].id,  "ai_query") )
        with col2:
            audio_file_1=st.session_state["smalltalk_entry"].ai_query_mp3_path
            if audio_file_1:
                st.audio(audio_file_1, autoplay= not st.session_state["smalltalk_entry"].ai_query_mp3_played) # wyświetl playera i odtwórz mp3 automatycznie
                st.session_state["smalltalk_entry"].ai_query_mp3_played=True # oznacz że już odtworzone żeby automatycznie sie wiecej nie odpalało

        st.divider()

        #
        # odpowiedź usera
        col1, col2 = st.columns( [3, 7] )
        with col1:
            try:
                user_answer_mp3 = audiorecorder(   #zwraca segment audio (obiekt) ale nie plik
                        start_prompt="Nagraj odpowiedź",
                        stop_prompt="Zatrzymaj nagranie"
                        )
                if user_answer_mp3:
                    # Zapisz nagranie do bufora w pamięci aby wyliczyć MD5
                    audio_buffer = BytesIO()
                    user_answer_mp3.export(audio_buffer, format="mp3") #segment audio zapisuje do pliku w pamięci (BytesIO)
                    audio_buffer.seek(0)  # Reset bufora na początek
                    # Wygenerowanie skrótu MD5 dla nagrania 
                    current_audio_MD5 = md5( audio_buffer.getvalue() ).hexdigest() #skrót MD5 aktualnego nagrania-  getvalue() zwraca ciąg bajtów pliku audio (a nie pythonowy segement audio)

                    if (st.session_state["user_answer_mp3_MD5"] != current_audio_MD5): #jeśli mamy inne nagranie niż w session_state (porównując MD5 nagrań)
                        #ustal i zapamiętaj nazwe pliku pod jakim zapisać nagranie
                        st.session_state["smalltalk_entry"].user_answer_mp3_path = get_mp3_filepath(st.session_state["smalltalk_entry"].id, "user_answer")
                        # Zapisz nagranie do pliku
                        user_answer_mp3.export(st.session_state["smalltalk_entry"].user_answer_mp3_path, format="mp3")
                        # transkrybuj nagranie usera na text 
                        st.session_state["smalltalk_entry"].user_answer_txt = transcribe_audio_to_text( st.session_state["smalltalk_entry"].user_answer_mp3_path )
                        st.session_state["user_answer_mp3_MD5"] = current_audio_MD5
            finally:
                # Wyczyszczenie pamięci audiorecordera  - ale to nie działa, przy każdym renderowaniu strony nagrywa sie ponownai poprzednie nagranie nawet jeśli user nic nowego nie nagrał - zaczęło działać dopiero sprawdzanie po MD5 nagrań
                user_answer_mp3 = None
            
        with col2: 
            # jeśli user nagrał swoją odpowiedź to pokaż playera w 2 kolumnie
            if st.session_state["smalltalk_entry"].user_answer_mp3_path:
                st.audio(st.session_state["smalltalk_entry"].user_answer_mp3_path ) # wyświetl playera 

        user_answer_text_edit = st.text_area("Twoja odpowiedź", value = st.session_state["smalltalk_entry"].user_answer_txt, key="user_answer_text_edit", label_visibility="hidden")
        if user_answer_text_edit:
            st.session_state["smalltalk_entry"].user_answer_txt = user_answer_text_edit #zapisz jesli user zmienił recznie treść
            if st.button("Oceń moją odpowiedź"):
                st.session_state["smalltalk_entry"].ai_evaluation_txt = evaluate_user_response(st.session_state["smalltalk_entry"].user_answer_txt)

        # audio_file_2=st.session_state["smalltalk_entry"].user_answer_mp3_path
        # if audio_file_2:
        #     st.audio(audio_file_2) 

        st.divider()
        #st.subheader("Twoja ocena", divider="green")
        with st.expander("Twoja ocena:", expanded=True):
            st.markdown(st.session_state["smalltalk_entry"].ai_evaluation_txt)
    

        col1, col2 = st.columns( [3, 7] )
        with col1:
            if st.session_state["smalltalk_entry"].ai_evaluation_txt  and st.session_state["smalltalk_entry"].ai_evaluation_mp3_path=="":
                if st.button("Przeczytaj", key="readEvaluationBtn"):
                    st.session_state["smalltalk_entry"].ai_evaluation_mp3_path = generate_speech(text=st.session_state["smalltalk_entry"].ai_evaluation_txt, 
                                                                                                output_audio_path=get_mp3_filepath(st.session_state["smalltalk_entry"].id, "ai_evaluation"))
        with col2:
            audio_file_3=st.session_state["smalltalk_entry"].ai_evaluation_mp3_path
            if audio_file_3:
                st.audio(audio_file_3, autoplay=not st.session_state["smalltalk_entry"].ai_evaluation_mp3_played) # wyświetl playera i odtwórz mp3 automatycznie
                st.session_state["smalltalk_entry"].ai_evaluation_mp3_played = True # oznacz że już odtworzone żeby automatycznie sie wiecej nie odpalało

        st.divider()

        if st.button("Zapisz konwersajce w bazie"): # dodać jeszcze blokowanie przycisku dopóki nie ma oceny wypowiedzi AI, np.   , disabled=not st.session_state["smalltalk_entry"].ai_evaluation_txt
            add_smalltalk_to_db(id          = st.session_state['smalltalk_entry'].id,
                                key_txt     = st.session_state["smalltalk_entry"].get_conv_text(),
                                payload_xml = st.session_state["smalltalk_entry"].to_dict()
                                )
            st.toast("Notatka zapisana", icon="🎉")





        st.markdown( f"<div style='text-align: right;'>id={st.session_state['smalltalk_entry'].id}</div>", 
                    unsafe_allow_html=True )

#
#content strony z wyszukiwarką 
#
def page_2():
    st.title("Wyszukiwarka")
    query = st.text_input("Wyszukaj wcześniejszą rozmówkę w bazie")
    if st.button("Szukaj", key="Search"):
        st.session_state["select_from_db_result"] = select_from_db(query)

    if st.session_state["select_from_db_result"]:
        for record in st.session_state["select_from_db_result"]:
            with st.container(border=True):
                #pokaż treści konwersacji
                with st.expander(record["payload"].get('ai_query_txt', "")):
                     st.markdown(":violet[**Odpowiedź:**] " + record["payload"].get('user_answer_txt', ""))
                     st.markdown(":violet[**Ocena:**] "  + record["payload"].get('ai_evaluation_txt', ""))
                #pokaż progressbar z podobieństwem i przycisk załadowania  
                col21, col22 = st.columns([7,3])
                with col21:
                    if record["score"]:
                        st.progress(record["score"], text=f':violet[Dopasowanie {round(record["score"]*100)}%]')
                with col22:
                     if st.button("Odtwórz", key=record["payload"].get('id', "0"), use_container_width=True):
                         load_smalltalk_entry_from_payload(record["payload"])
                         navigate_to("Page 1")



#załąowanie z pyaloadu z bazy danych do st.session_state['smalltalk_entry']
def load_smalltalk_entry_from_payload(payload):
    # Upewnij się, że mamy obiekt w session_state
    if "smalltalk_entry" not in st.session_state:
        st.session_state['smalltalk_entry'] = SmallTalk_entry()
    
    # Mapowanie danych z payload na odpowiednie pola w obiekcie SmallTalk_entry
    st.session_state['smalltalk_entry'].id = payload.get('id', 0)
    st.session_state['smalltalk_entry'].lang = payload.get('lang', "")
    st.session_state['smalltalk_entry'].level = payload.get('level', "")
    st.session_state['smalltalk_entry'].subject = payload.get('subject', "")
    st.session_state['smalltalk_entry'].system_prompt_txt = payload.get('system_prompt_txt', "")
    st.session_state['smalltalk_entry'].smalltalk_prompt_txt = payload.get('smalltalk_prompt_txt', "")
    
    st.session_state['smalltalk_entry'].ai_query_txt = payload.get('ai_query_txt', "")
    st.session_state['smalltalk_entry'].ai_query_mp3_path = payload.get('ai_query_mp3_path', "")
    st.session_state['smalltalk_entry'].ai_query_mp3_played = payload.get('ai_query_mp3_played', False)
    
    st.session_state['smalltalk_entry'].user_answer_txt = payload.get('user_answer_txt', "")
    st.session_state['smalltalk_entry'].user_answer_mp3_path = payload.get('user_answer_mp3_path', "")
    st.session_state['smalltalk_entry'].user_answer_mp3_MD5 = payload.get('user_answer_mp3_MD5', "")
    
    st.session_state['smalltalk_entry'].ai_evaluation_txt = payload.get('ai_evaluation_txt', "")
    st.session_state['smalltalk_entry'].ai_evaluation_mp3_path = payload.get('ai_evaluation_mp3_path', "")
    st.session_state['smalltalk_entry'].ai_evaluation_mp3_played = payload.get('ai_evaluation_mp3_played', False)
    
    st.session_state['smalltalk_entry'].ai_alt_example_txt = payload.get('ai_alt_example_txt', "")
    st.session_state['smalltalk_entry'].ai_alt_example_mp3_path = payload.get('ai_alt_example_mp3_path', "")

    st.toast("Pomyślnie załadowano dane do smalltalk_entry.")




#############################################################################################################
# MAIN
#
st.set_page_config( page_title="Small-Talk Teacher", layout="centered" )


get_openai_API_KEY()

assure_db_collection_exists()


#
# Session_state initializatoion
if "smalltalk_entry" not in st.session_state:   # przechowuje edytowaną przez usera wersje transkrypcji - zawartość z pola edycyjnego
    st.session_state["smalltalk_entry"] = SmallTalk_entry()
if "user_answer_mp3_MD5" not in st.session_state:
    st.session_state["user_answer_mp3_MD5"] = ""

if "select_from_db_result" not in st.session_state: #przechowuje wynik wyszukiwania na zakładce wyszukiwania rozmówek
    st.session_state["select_from_db_result"]=""

# Ustawienie domyślnej strony w sesji
if "page" not in st.session_state:
    st.session_state.page = "Page 1"  # Domyślna strona



#
# Sidebar
with st.sidebar:
    sidebar()

# Renderowanie aktualnej strony na podstawie st.session_state.page
if st.session_state.page == "Page 1":
    page_1()

elif st.session_state.page == "Page 2":
    page_2()










# #przykład wywołania
# result = select_from_db(query="przykładowe pytanie AI")
# if result:
#     # Wybierz pierwszy rezultat
#     payload = result[0]['payload']
    
#     # Załaduj dane z payload do smalltalk_entry
#     load_smalltalk_entry_from_payload(payload)


