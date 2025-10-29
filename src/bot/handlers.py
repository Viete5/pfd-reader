from aiogram import Router, F
from aiogram.types import Message, Document
from src.core.orchestrator import handle_document_upload, handle_user_query
import os
import tempfile

router = Router()

@router.message(F.document)
async def handle_doc(message: Message):
    if not message.document.mime_type == "application/pdf":
        await message.answer("Пожалуйста, отправьте PDF-файл.")
        return

    # Скачиваем файл во временную папку
    file = await message.bot.get_file(message.document.file_id)
    file_path = os.path.join(tempfile.gettempdir(), f"{message.from_user.id}.pdf")
    await message.bot.download_file(file.file_path, file_path)

    # Передаём оркестратору
    result = await handle_document_upload(message.from_user.id, file_path)
    await message.answer(result)

@router.message(F.text)
async def handle_text(message: Message):
    result = await handle_user_query(message.from_user.id, message.text)
    await message.answer(result)