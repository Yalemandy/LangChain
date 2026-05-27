from openai import OpenAI

client = OpenAI(api_key="")

response = client.responses.create(
    model = "gpt-4o-mini",
    input = "介绍一下你自己"
)

print(response.output_text)