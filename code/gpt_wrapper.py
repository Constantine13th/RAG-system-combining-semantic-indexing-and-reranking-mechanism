from openai import OpenAI
import json


with open("config.json", "r") as f:
    config = json.load(f)


client = OpenAI(api_key=config["openai_api_key"])

def ask_gpt(prompt, model=None):
    model = model or config.get("model", "gpt-4")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content.strip()
