import openai

client = openai.OpenAI(
    api_key="sk-zDO6n93JO_Uu1pcP0zD47w",
    base_url="http://131.220.150.238:8080",
)

response = client.chat.completions.create(
    model="openai/Qwen3.5-27B-Q5_K_M.gguf",
    messages=[
        {
            "role": "user",
            "content": "this is a test request, write a short poem",
        }
    ],
)

print(response)
