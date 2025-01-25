import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import weaviate
from openai import OpenAI
import asyncio
from weaviate.classes.init import Auth
from weaviate.classes.init import AdditionalConfig, Timeout

load_dotenv()


TELEGRAM_TOKEN = os.getenv("BOT_TOKEN")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")

class RAGSystem:
    def __init__(self):

        self.weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
            headers={
                "X-OpenAI-Api-Key": OPENAI_API_KEY
            },
            additional_config=AdditionalConfig(
                timeout=Timeout(init=30, query=60, insert=120)
            )
        )
  
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        self.collection = self.weaviate_client.collections.get("Knowledge")

    async def search_vector_db(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:

        try:
            response = self.collection.query.bm25(
                query=query,
                limit=limit
            ).with_additional(["certainty"]).do()
            
            return [
                {
                    "content": obj.properties.get("content", ""),
                    "_additional": {"certainty": obj.additional.certainty}
                }
                for obj in response.objects
            ]
        except Exception as e:
            print(f"Error searching vector database: {e}")
            raise

    async def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:

        try:
            context_text = "\n\n".join([item['content'] for item in context])
            prompt = (
                "Based on the following context information, provide a clear and accurate "
                "response to the question. Include only information that is supported by "
                f"the context.\n\nContext:\n{context_text}\n\nQuestion: {query}\n\nAnswer:"
            )

            completion = await asyncio.to_thread(
                self.openai_client.completions.create,
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0.7
            )
            
            return completion.choices[0].text.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            raise

    async def process_query(self, query: str) -> Dict[str, Any]:
        search_results = await self.search_vector_db(query)
        response = await self.generate_response(query, search_results)
        
        return {
            "response": response,
            "sources": [
                {
                    "content": result['content'][:100] + "...",
                    "certainty": result['_additional']['certainty']
                }
                for result in search_results
            ]
        }

rag_system = RAGSystem()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    welcome_message = (
        "Welcome! I am here to help answer your questions using our knowledge base. "
        "You can ask me anything, and I will search through our database to provide "
        "relevant information with source citations."
    )
    await update.message.reply_text(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

    help_message = (
        "Here's how to use this bot:\n\n"
        "1. Simply type your question and send it to me\n"
        "2. I will search our knowledge base and provide relevant information\n"
        "3. Each response includes source citations and confidence scores\n\n"
        "Available commands:\n"
        "/start - Begin interacting with the bot\n"
        "/help - Show this help message"
    )
    await update.message.reply_text(help_message)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not update.message or not update.message.text:
        return

    message_text = update.message.text
    
    if message_text.startswith('/'):
        return

    try:
        await update.message.reply_text("Searching knowledge base for relevant information...")
        
        result = await rag_system.process_query(message_text)
        
        response_text = f"{result['response']}\n\nReferences:\n"
        for idx, source in enumerate(result['sources'], 1):
            response_text += (
                f"{idx}. {source['content']} "
                f"(Confidence: {source['certainty']*100:.1f}%)\n"
            )
        
        await update.message.reply_text(response_text)
    except Exception as e:
        error_message = (
            "I apologize, but I encountered an error while processing your request. "
            "Please try again later or rephrase your question."
        )
        print(f"Error processing message: {e}")
        await update.message.reply_text(error_message)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):

    print(f"Error occurred: {context.error}")
    if update and update.effective_chat:
        await update.effective_chat.send_message(
            "I apologize, but an error occurred while processing your request. "
            "Please try again later."
        )

def main():

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.add_error_handler(error_handler)

    print("Initializing the Telegram bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()