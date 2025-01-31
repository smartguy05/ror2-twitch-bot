import os
import openai
from twitchio.ext import commands
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import asyncio 

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Credentials
TWITCH_BOT_TOKEN = os.getenv("TWITCH_BOT_TOKEN")
TWITCH_CHANNEL = os.getenv("TWITCH_CHANNEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI API
openai.api_key = OPENAI_API_KEY

# Set up Chroma
client = chromadb.Client(Settings(persist_directory="./chroma_db",anonymized_telemetry=False))
collection = client.get_collection(
    name="ror2_wiki",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-embedding-ada-002"
    )
)

BOT_PREFIX = "!"
ALLOWED_TOPIC_KEYWORDS = ["build", "strategy", "items", "gameplay", "enemies", "boss", "stage", "map", "artifact"]

class RoR2ChatBot(commands.Bot):

    def __init__(self):
        super().__init__(token=TWITCH_BOT_TOKEN, prefix=BOT_PREFIX, initial_channels=[TWITCH_CHANNEL])

    async def event_ready(self):
        print(f"Logged in as | {self.nick}")
        print(f"User id is | {self.user_id}")

    async def event_message(self, message):
        # Ignore messages from the bot itself
        if message is None or message.author is None or message.author.name.lower() == self.nick.lower():
            return

        # Process commands
        print(f"Message received: {message.content} from {message.author.name}")
        await self.handle_commands(message)

    @commands.command(name="ror2")
    async def ror2_command(self, ctx):
        """
        Usage: !ror2 <question about Risk of Rain 2 gameplay>
        """
        user_question = ctx.message.content[len("!ror2 "):].strip()

        # Quick guard: only answer if it seems related to gameplay
        # if not self.is_gameplay_question(user_question):
        #     print("Not a gameplay question")
        #     return

        # Step 1: Retrieve relevant wiki chunks
        relevant_context = self.get_relevant_context(user_question)

        # Step 2: Use LLM (OpenAI) to form an answer
        answer = self.generate_answer(user_question, relevant_context)

        # Step 3: Send the answer to Twitch chat (limit if it’s very long)
        if len(answer) > 400:
            answer = answer[:400] + "..."
        await ctx.send(answer)

    def is_gameplay_question(self, question):
        # Simple heuristic check: does the question contain some key gameplay words?
        q_lower = question.lower()
        return any(keyword in q_lower for keyword in ALLOWED_TOPIC_KEYWORDS)

    def get_relevant_context(self, query, top_k=3):
        """
        Query Chroma for the most relevant wiki chunks.
        """
        results = collection.query(query_texts=[query], n_results=top_k)
        # Each result is a dictionary with "documents", "metadatas", etc.
        docs = results["documents"][0]  # top_k docs for the single query
        return "\n\n".join(docs)

    def generate_answer(self, question, context):
        """
        Use an OpenAI completion call. Provide the RAG context in the prompt.
        """
        system_prompt = (
            "You are an expert on Risk of Rain 2 gameplay. "
            "You have access to the official wiki content. "
            "Answer concisely, focusing strictly on gameplay. "
            "Answers should be no more than 2 or 3 sentences in length"
            "If the question is not about gameplay, you should refuse to answer."
        )
        user_content = (
            f"Question: {question}\n\n"
            f"Relevant Wiki Context: {context}\n\n"
            "Answer:"
        )

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=300,
            temperature=0.7
        )
        print(response)
        answer = response.choices[0].message.content.strip()
        return answer


if __name__ == "__main__":
    bot = RoR2ChatBot()
    
    async def run_bot():
        await bot.start()

    try:
        # Use the existing running loop
        loop = asyncio.get_running_loop()
        loop.run_until_complete(run_bot())
    except RuntimeError:  # If no loop is running, create a new one
        asyncio.run(run_bot())
    