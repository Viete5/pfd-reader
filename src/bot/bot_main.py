from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from src.config import TELEGRAM_BOT_TOKEN
from src.bot.handlers import router

async def start_bot():
    """Запускает Telegram бота"""
    bot = Bot(
        token=TELEGRAM_BOT_TOKEN, 
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    dp = Dispatcher()
    dp.include_router(router)
    
    print("✅ Бот запущен и готов к работе!")
    await dp.start_polling(bot)
