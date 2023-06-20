from langchain.llms import LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
model_path = r"C:\gpt4all\models\ggml-gpt4all-j-v1.3-groovy.bin"
llm = LlamaCpp(model_path=model_path, n_ctx=1000, callbacks=[StreamingStdOutCallbackHandler()])


