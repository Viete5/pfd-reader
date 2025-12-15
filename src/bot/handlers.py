from aiogram import Router, F
from aiogram.types import Message, Document
from aiogram.filters import Command
from aiogram.enums import ParseMode
import os
import tempfile
from src.core.orchestrator import (
    handle_document_upload,
    handle_user_query,
    get_help_message,
    handle_quiz,
    _pending_quiz_topic,
    _pending_quiz_count,
)

router = Router()

@router.message(Command("start", "help"))
async def cmd_start(message: Message):
    """Обработчик команд /start и /help"""
    await message.answer(get_help_message())

@router.message(F.document)
async def handle_document(message: Message):
    """Обработчик загрузки документов"""
    if not message.document.mime_type == "application/pdf":
        await message.answer("❌ Пожалуйста, отправьте PDF-файл.")
        return

    await message.answer("📥 Начинаю обработку вашего конспекта...")

    try:
        # Скачиваем файл во временную папку
        file = await message.bot.get_file(message.document.file_id)
        file_path = os.path.join(tempfile.gettempdir(), f"{message.from_user.id}_{message.document.file_name}")
        
        await message.bot.download_file(file.file_path, file_path)

        # Обрабатываем документ
        result = await handle_document_upload(message.from_user.id, file_path)
        
        # Удаляем временный файл
        if os.path.exists(file_path):
            os.remove(file_path)
            
        await message.answer(result)
        
    except Exception as e:
        await message.answer("❌ Произошла ошибка при обработке файла. Попробуйте еще раз.")

async def handle_text(message: Message):
    """Обработчик текстовых сообщений"""
    user_text = message.text.strip()
    
    if not user_text:
        return

    user_id = message.from_user.id

    # Показываем индикатор набора
    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        # Если пользователь уже в процессе квиза — продолжаем квиз
        if _pending_quiz_topic.get(user_id) is not None or _pending_quiz_count.get(user_id):
            result = await handle_quiz(user_id, user_text)
        else:
            # Иначе — обычная обработка запроса
            result = await handle_user_query(user_id, user_text)

        await message.answer(result, parse_mode=ParseMode.HTML)
    except Exception:
        await message.answer("❌ Произошла ошибка при обработке запроса. Попробуйте еще раз.")
