import os
import requests
from dotenv import load_dotenv

# Load TELEGRAM_API_KEY from .env
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_API_KEY")

if not BOT_TOKEN:
    print("❌ TELEGRAM_API_KEY not found in .env file.")
    exit(1)

print("="*50)
print("🔗 Telegram Webhook Setup")
print("="*50)

# 1. Get the Railway URL
railway_url = input("Enter your Railway Public Domain (e.g., https://my-app-production.up.railway.app):\n> ").strip().rstrip("/")
webhook_url = f"{railway_url}/webhook"

# 2. Get the Webhook Secret
print("\nEnter a secure password for your webhook (e.g., random letters/numbers).")
print("You MUST add this exact same string as WEBHOOK_SECRET in your Railway dashboard!")
secret_token = input("> ").strip()

# 3. Call Telegram API to set it
print(f"\nSetting webhook to: {webhook_url}...")
response = requests.post(
    f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook",
    data={
        "url": webhook_url,
        "secret_token": secret_token
    }
)

if response.status_code == 200 and response.json().get("ok"):
    print("\n✅ Webhook successfully set!")
    print("\n⚠️ IMPORTANT NEXT STEPS:")
    print("1. Go to your Railway Dashboard -> Variables")
    print(f"2. Add a new variable named: WEBHOOK_SECRET")
    print(f"3. Set its value to: {secret_token}")
    print("4. Railway will rebuild. Your bot will then be fully connected!")
else:
    print(f"\n❌ Failed to set webhook. Telegram responded with:")
    print(response.text)
