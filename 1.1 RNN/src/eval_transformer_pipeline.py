from transformers import pipeline

class Distilgpt2Generator():

    def __init__ (self):
        self.generator = pipeline("text-generation", model="distilgpt2")

    def generate_output_text (self, input_text:str, max_length:int=20):
        result = self.generator(input_text, max_length=max_length, do_sample=True, top_k=50)
        return result[0]["generated_text"][len(input_text):]