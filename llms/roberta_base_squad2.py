from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

def get_answer(QA_input):
    return nlp(QA_input)
