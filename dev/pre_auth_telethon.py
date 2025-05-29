import asyncio
from telethon import TelegramClient
from settings import settings # Убедитесь, что settings.py доступен

async def main():
    if not settings.telegram.client_api_id or not settings.telegram.client_api_hash:
        print("Ошибка: TELEGRAM_CLIENT_API_ID и TELEGRAM_CLIENT_API_HASH должны быть установлены в .env")
        return

    client = TelegramClient(
        settings.telegram.client_session_name,
        settings.telegram.client_api_id,
        settings.telegram.client_api_hash
    )
    print("Connecting to Telegram...")
    await client.connect()
    if not await client.is_user_authorized():
        print("Client is not authorized. Authorizing...")
        if settings.telegram.client_phone_number:
            await client.send_code_request(settings.telegram.client_phone_number)
            code = input("Enter the code you received: ")
            await client.sign_in(phone=settings.telegram.client_phone_number, code=code)
        else:
            # Если номер не указан, Telethon запросит его
            await client.start() 
        print("Signed in successfully!")
    else:
        print("Already authorized.")
    
    me = await client.get_me()
    print(f"Authorized as: {me.first_name} {me.last_name or ''} (@{me.username or ''})")
    
    await client.disconnect()
    print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main())