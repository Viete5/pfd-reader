import os
import shutil
from functools import lru_cache
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_ROOT_PATH
from src.tools.document_loader import universal_loader


def get_user_db_path(user_id: str | int) -> str:
    return os.path.join(VECTOR_DB_ROOT_PATH, f"user_{user_id}")


@lru_cache(maxsize=1)
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def index_user_document(file_path: str, user_id: str | int):
    user_persist_dir = get_user_db_path(user_id)
    print(f"Начало индексации файла: {file_path} для user_id: {user_id}")

    if os.path.exists(user_persist_dir):
        print(f"Удаление старой базы для user_id: {user_id}")
        shutil.rmtree(user_persist_dir)

    try:

        documents = universal_loader.load_document(file_path)
        if not documents:
            raise ValueError("Документ пуст или не удалось его загрузить.")

        print(f"✅ Успешно загружен документ: {len(documents)} страниц/секций")

    except Exception as e:
        print(f"❌ Ошибка загрузки документа: {e}")
        return False

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    texts = text_splitter.split_documents(documents)
    print(f"Документ разбит на {len(texts)} чанков.")

    try:
        print("Создание эмбеддингов...")
        embeddings = _get_embeddings()

        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=user_persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        vectorstore.persist()
        print(f"✅ Новая база успешно создана для user_id: {user_id} в {user_persist_dir}")
        return True

    except Exception as e:
        print(f"❌ Ошибка создания векторной базы: {e}")
        return False


index_user_pdf = index_user_document
