from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


llm = OllamaLLM(model="llama3.1:8b",
    base_url="http://localhost:11434",  # Ollama 서버 주소
    call_kwargs={
        "num_gpu_layers": 1000,      # Ollama 서버에 GPU 레이어 옵션 전달(서버와 호환되는 경우)
        "n-gpu": 2
    }
)


template = "{country}의 환율 구조에 대해 이해하고 있는 걸 설명해줘."

prompt = PromptTemplate.from_template(template=template)
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"country", "한국"})
print(result)