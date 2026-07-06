import openai
import time

client = openai.OpenAI(
    api_key="sk-zDO6n93JO_Uu1pcP0zD47w",
    base_url="http://131.220.150.238:8080",
)

# Test multiple models to see which ones work through the proxy
models_to_test = [
    "openai/gpt-4o-mini",
    "openai/gpt-4.1-mini",
    "mistral/mistral-small-latest",
    "openai/gpt-5-nano",
]

for model in models_to_test:
    print(f"\n{'='*50}")
    print(f"Testing: {model}")
    print(f"{'='*50}")
    try:
        t0 = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Reply with exactly one word: hello"}
            ],
            max_tokens=10,
            timeout=30.0,
        )
        elapsed = time.time() - t0
        content = response.choices[0].message.content
        print(f"  Response: {content}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Status: OK")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  Error after {elapsed:.1f}s: {e}")
        print(f"  Status: FAILED")
