from transformers import VitsModel, AutoTokenizer
import torch
import scipy
from IPython.display import Audio

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

    
def gen_audio_from_text(text, dir, file_name):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    audio = Audio(output.numpy(), rate=model.config.sampling_rate)
    with open(f"{dir}/{file_name}.wav", 'wb') as f:
        f.write(audio.data)

    