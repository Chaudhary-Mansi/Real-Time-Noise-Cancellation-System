# Real-Time-Noise-Cancellation-System

**Overview**
This project implements a real-time noise cancellation system using Python, leveraging the pyaudio library for audio processing and scipy for spectral processing. The system uses an adaptive noise estimation algorithm and a Wiener filter to reduce background noise in real-time.

**Features**
Real-Time Processing: The system processes audio streams in real-time, capturing input from a microphone and outputting the processed audio.
Adaptive Noise Estimation: Uses an adaptive algorithm to estimate and update the noise level dynamically.
Wiener Filter: Applies a Wiener filter for effective noise cancellation.
Audio Saving: Saves the processed audio to a .wav file upon termination.

**Requirements**
Python 3.x
Required Libraries:
pyaudio
numpy
scipy
wavfile (part of scipy.io)

**Optional Libraries:**
matplotlib (for visualization, not included in this version)

**Installation**
To install the required libraries, run the following command:

bash

pip install pyaudio numpy scipy
Usage
Run the Script:
bash


python noise_cancellation.py
Start and Stop:
The script will start capturing and processing audio immediately.
To stop the process, press Ctrl+C.
Output File:
The processed audio will be saved as processed_audio.wav in the current directory.
Code Structure
NoiseCancellationSystem Class
This class encapsulates the noise cancellation functionality:

__init__ Method: Initializes the system with an alpha value for noise estimation.
adaptive_noise_estimation Method: Estimates noise from the audio chunk using an adaptive algorithm.
wiener_filter Method: Applies a Wiener filter for noise cancellation.
process_audio_stream Method: Main loop for processing the audio stream.
save_to_wav Method: Saves the processed audio to a .wav file.
Main Function
The main function initializes the NoiseCancellationSystem and starts the audio processing stream.

Example Workflow
Initialization:
python


system = NoiseCancellationSystem(alpha=0.9)
Start Processing:
python


system.process_audio_stream()
Stop Processing:
Press Ctrl+C to terminate the process.
Save Processed Audio:
The processed audio is saved automatically as processed_audio.wav.
Parameters
Alpha (α): The smoothing factor for noise estimation. Default value is 0.9.
Technical Details
Audio Settings:
CHUNK: Number of audio samples per frame (default: 1024).
RATE: Sample rate (default: 44100 Hz).
FORMAT: Audio format (default: pyaudio.paInt16).
CHANNELS: Number of audio channels (default: 1, mono).
Noise Cancellation:
Uses an adaptive noise estimation algorithm to track changes in noise levels.
Applies a Wiener filter to cancel out the estimated noise.
Future Improvements
Enhanced Noise Filtering: Explore other noise reduction techniques such as deep learning-based models (e.g., RNNoise) or adaptive filters (e.g., LMS, NLMS).
User Interface: Implement a graphical user interface (GUI) to control parameters and provide real-time feedback.
Visualization: Add optional visualization of audio signals and noise levels using matplotlib.

**References**
RNNoise: A real-time neural network-based noise suppression library.
Wiener and Adaptive Filters: Detailed comparison and implementation of Wiener and adaptive filters for noise cancellation.
Deep Learning-Based Noise Removal: Using CNNs for detecting and removing noise from audio.
By following this README, you can set up and run a basic real-time noise cancellation system using Python. This project serves as a foundation for more advanced noise reduction techniques and can be extended with additional features and improvements.
