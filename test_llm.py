from openai import OpenAI

client = OpenAI(
    base_url="https://api.ofox.ai/v1",
    api_key="sk-of-phZLjTyuEFVMhoDyoeUWSEIwQffjAvMQpQUMoACmHKxuQufKKqDapEpIPVSCBAyd",
)

response = client.chat.completions.create(
    model="z-ai/glm-5.2",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
)

print(response.choices[0].message.content)