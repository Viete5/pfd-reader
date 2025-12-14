import asyncio
import logging
from src.bot.bot_main import start_bot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("🚀 Запуск StudyMate бота...")
    asyncio.run(start_bot())
#test