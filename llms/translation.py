from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("unicamp-dl/translation-en-pt-t5")

model = AutoModelForSeq2SeqLM.from_pretrained("unicamp-dl/translation-en-pt-t5")

enpt_pipeline = pipeline('text2text-generation', model=model, tokenizer=tokenizer)

# translation = enpt_pipeline("translate English to Portuguese: I like to eat rice.")

def translate_en_pt(text):
    return enpt_pipeline(f"translate English to Portuguese: {text}")