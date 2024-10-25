from dotenv import dotenv_values
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams 

from smalltalks_ai_lib import *

## Sentence-Transformers (SBERT) jest zoptymalizowanym modelem do generowania embeddingów zdań, bardzo popularna do uzyskiwania semantycznie znaczących wektorów. - alternatywa dla AI
## instalowana przez:  pip install sentence-transformers
#from sentence_transformers import SentenceTransformer
#  # EMBEDDING_MODEL = "stsb-roberta-base-v2"
#  # EMBEDDING_DIM = 768


QDRANT_COLLECTION_NAME = "smalltalks_db"

# EMBEDDING_MODEL = "text-embedding-3-large"
# EMBEDDING_DIM = 3072

EMBEDDING_MODEL = "text-embedding-3-small" #nazwa modelu dla embedding'sów w OpenAI
EMBEDDING_DIM = 1536  #długość wektora dla embedingsów z modelu small, tu definiujemy z góry jao stałą bo przy tworzeniu kolekcji w QDrant
                     #trzeba podać z góry wielkość wektora jakim będą opisywane poszczególne obiekty w kolekcji i tej wielkości trzeba sie potem trzymać 




#
# DB -QDrant
#
@st.cache_resource #dekorator cache'ujący to co robi funkcja wewn - czyli tu cacheuje baze danych w pamięci
def get_qdrant_client():
    db_env = dotenv_values(".db_env")

    qdrant_client = QdrantClient(    # V6: łączymy się z bazą zdalną
        url     = db_env["QDRANT_URL"],  
        api_key = db_env["QDRANT_API_KEY"] 
    )
    ##print(qdrant_client.get_collections()) # wypisz liste kolekcji w bazie
    return qdrant_client

#
# funkcja do utworzenia kolekcji z wektorem wyszukiwania o podanej wielkości - o ile jeszcze taka kolekcja nie istnieje
#
def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        #print("Kolekcja już istnieje")
        return

#
#definicja fumkcji obliczjącej wektor embedingsów (wektor = kolekcja 1536 liczb zmiennoprzecinkowych wyliczanych dla podanego tekstu)
#
def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],  #na wejciu przekazujemy tablice tekstów - tu tworzymy tab. 1-elementową
        model=EMBEDDING_MODEL, #wybrany model
        dimensions=EMBEDDING_DIM, #wielkość wektora wynikowego - i tak dla modelu *-small będzie miał 1536 ale dla zabezpieczenia na przyszłość podaje wprost
    )
    embedding = result.data[0].embedding

    # # model = SentenceTransformer('stsb-roberta-base-v2') #'all-MiniLM-L6-v2')  # Możesz wybrać różne modele
    # # embedding = model.encode(text)
    # with st.expander(f'Embedings {len(embedding)}:', expanded=True):
    #      st.markdown(embedding)
    return embedding



#
#funkcja dodająca obiekt (text) do kolekcji 
#
def add_smalltalk_to_db(id, key_txt, payload_xml):
    qdrant_client = get_qdrant_client()
   
    #wstaw nowy obiekt - notatke
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id= id,
                vector=get_embedding(text=key_txt), #wyznacz wektor embedding
                payload= payload_xml #xml z serializowanym obiektem do wgrania do bazy
                )
            ]
        )





# funkcja wyszukująca w kolekcji 10 najbardziej podobnych tekstów(obiektów)
def select_from_db(query=None):
    # Pobierz klienta Qdrant
    qdrant_client = get_qdrant_client()
    
    # search() zwraca listę punktów (points), które można bezpośrednio iterować, a każdy punkt ma atrybuty takie jak id i payload.
    # scroll() zwraca krotkę składającą się z dwóch elementów: listy punktów (points) i flagi has_more, która mówi, czy są kolejne wyniki - points'y są o 1 poziom głębiej w search

    # Jeśli zapytanie nie jest określone, zwróć 10 ostatnich rekordów (domyślny przypadek)
    if query is None or len(query)<3:
        results, has_more = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=10,  # Maksymalnie 10 rekordów
        )
        show_score = False  # Przy scroll() w wyniku nie ma score
    else:
        # Użyj wyszukiwania na podstawie embeddingów, jeśli dostarczono query (np. kluczowy tekst)
        results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(query),  # Użycie wektora wygenerowanego na podstawie zapytania
            limit=10,  # Maksymalnie 10 rekordów
        )
        show_score = True # Przy search() w wyniku jest score
    
    # Przetwarzanie wyników i zwrócenie ich w formacie przyjaznym do analizy
    records = []
    for point in results:
        record = {
            "id": point.id,
            "payload": point.payload, #zawiera całego xmla z dumpem obiektu
            "score": point.score if show_score else 0,  #pole score - wynik dopasowania (-1 do +1) z Cosinusa
        }
        records.append(record)

# Przykłaowe użycie
#       results = select_from_db(query="przykładowe pytanie AI")
#       print(results)

    return records


