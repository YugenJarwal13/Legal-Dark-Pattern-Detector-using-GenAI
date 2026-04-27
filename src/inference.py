import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise ValueError("GROQ_API_KEY not set")

client = Groq(api_key=API_KEY)


def generate(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"ERROR: {str(e)}"