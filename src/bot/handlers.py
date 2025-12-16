import os
import tempfile
from aiogram import Router, F
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command
from aiogram.enums import ParseMode

# Импортируем функции из оркестратора
from src.core.orchestrator import handle_document_upload, handle_user_query, get_help_message

router = Router()


@router.message(Command("start", "help"))
async def cmd_start(message: Message):
    """Обработчик команд /start и /help"""
    await message.answer(get_help_message(), parse_mode=ParseMode.HTML)


@router.message(F.document)
async def handle_document(message: Message):
    """Обработчик загрузки документов"""
    if not message.document.mime_type == "application/pdf":
        await message.answer("❌ Пожалуйста, отправьте PDF-файл.")
        return

    wait_msg = await message.answer("📥 Начинаю обработку вашего конспекта...")

    try:
        # Скачиваем файл во временную папку
        # Используем tempfile.gettempdir() для кроссплатформенности
        file = await message.bot.get_file(message.document.file_id)

        # Сохраняем во временную директорию
        # Важно: если нужно сохранить надолго для math_agent,
        # orchestrator сам скопирует его куда надо (в pdf_cache)
        temp_path = os.path.join(tempfile.gettempdir(), f"{message.from_user.id}_{message.document.file_name}")

        await message.bot.download_file(file.file_path, temp_path)

        # Вызываем оркестратор для обработки
        result_text = await handle_document_upload(message.from_user.id, temp_path)

        # Удаляем временный файл (orchestrator уже сделал свою копию, если нужно)
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

        await wait_msg.edit_text(result_text)

    except Exception as e:
        # Логируем ошибку, если есть логгер, иначе просто пишем в чат
        print(f"Upload Error: {e}")
        await wait_msg.edit_text("❌ Произошла ошибка при обработке файла. Попробуйте еще раз.")


@router.message(F.text)
async def handle_text(message: Message):
    """Обработчик текстовых сообщений"""
    user_text = message.text.strip()

    if not user_text:
        return

    # Показываем индикатор набора текста (или upload_document если это решение задачи)
    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        # Получаем ответ от оркестратора
        result = await handle_user_query(message.from_user.id, user_text)

        # ПРОВЕРКА ТИПА ОТВЕТА
        # Если оркестратор вернул FSInputFile (это PDF с решением задачи)
        if isinstance(result, FSInputFile):
            await message.answer_document(result, caption="✅ Вот решение вашей задачи!")

        # Если вернулся просто текст (строка)
        elif isinstance(result, str):
            # Избегаем отправки пустых сообщений
            if not result:
                result = "⚠️ Нет ответа."
            if "<tg-spoiler>" in str(result) or "<b>" in str(result):
                await message.answer(result, parse_mode=ParseMode.HTML)
            else:
                # Для остальных случаев (MathAgent обычно шлет Markdown)
                await message.answer(result, parse_mode=ParseMode.MARKDOWN)

        # На всякий случай (если вернулось что-то странное)
        else:
            await message.answer(str(result))

    except Exception as e:
        print(f"Handler Error: {e}")
        await message.answer("❌ Произошла ошибка при обработке запроса. Попробуйте еще раз.")
