from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')
set_seed(42)

#generator("Hello, I'm a language model,", max_length=144, num_return_sequences=1)

def gen_text_by_prompt(text, qnt_sentences, max_length):
    return generator(text, max_length=max_length, num_return_sequences=qnt_sentences)
    