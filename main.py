from transformers import VitsTokenizer, VitsModel
import numpy as np
import torch
import scipy
from IPython.display import Audio

# Load the Bengali tokenizer and model
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-ben")
model = VitsModel.from_pretrained("facebook/mms-tts-ben")

# Set the device for computation (GPU if available, otherwise CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Input text for TTS
text = "এই প্রকল্পটি আমাকে নতুন প্রযুক্তি অন্বেষণ এবং আমার দক্ষতা বাড়াতে সাহায্য করেছে।"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Set a random seed for reproducibility
from transformers import set_seed
set_seed(555)

# Generate the waveform without tracking gradients
with torch.no_grad():
    outputs = model(**inputs.to(device))

# Extract the waveform and ensure it's a 1D numpy array
waveform = outputs.waveform[0].cpu().float().numpy()
if waveform.ndim > 1:
    waveform = waveform.flatten()  # Flatten if it's multi-dimensional

# Normalize the waveform to int16 range for saving as a WAV file
if waveform.max() != 0:  
    waveform_int16 = np.int16(waveform / np.max(np.abs(waveform)) * 32767)

# Save the audio to a .wav file
sampling_rate = model.config.sampling_rate
scipy.io.wavfile.write("bengali.wav", rate=sampling_rate, data=waveform_int16)

# Play the generated audio
Audio("bengali.wav", rate=sampling_rate)
