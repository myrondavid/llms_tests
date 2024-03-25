from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length, min_length, do_sample=False):
    return summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)