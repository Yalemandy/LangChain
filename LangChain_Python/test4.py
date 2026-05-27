from langchain_ollama import ChatOllama

ollama_model = ChatOllama(model="qwen3.5:latest",base_url="http://127.0.0.1:11434")

print(ollama_model.invoke("who are you?").content)