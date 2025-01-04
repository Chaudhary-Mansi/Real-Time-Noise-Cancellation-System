import pyaudio
import numpy as np
import scipy.io.wavfile as wav
import tkinter as tk
from tkinter import messagebox
import threading
import datetime

# Constants
CHUNK = 1024  # Number of audio samples per frame
RATE = 44100  # Sample rate
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono audio

class NoiseCancellationSystem:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.noise_est = 0
        self.processed_audio_data = []
        self.is_running = False
        self.stream = None
        self.p = pyaudio.PyAudio()

    def adaptive_noise_estimation(self, audio_chunk):
        """Estimate noise from the audio chunk using an adaptive algorithm."""
        current_noise_est = np.mean(np.abs(audio_chunk))
        self.noise_est = self.alpha * self.noise_est + (1 - self.alpha) * current_noise_est

    @staticmethod
    def wiener_filter(audio_chunk, noise_estimation):
        """Apply Wiener filter for noise cancellation."""
        audio_fft = np.fft.fft(audio_chunk)
        magnitude = np.abs(audio_fft)
        phase = np.angle(audio_fft)

        noise_estimation = np.maximum(noise_estimation, 1e-10)
        gain = (magnitude ** 2) / (magnitude ** 2 + noise_estimation ** 2)

        processed_magnitude = gain * magnitude
        processed_audio = processed_magnitude * np.exp(1j * phase)

        return np.fft.ifft(processed_audio).real

    def process_audio_stream(self):
        """Main loop for processing audio stream."""
        try:
            self.stream = self.p.open(format=FORMAT,
                                       channels=CHANNELS,
                                       rate=RATE,
                                       input=True,
                                       output=True,
                                       frames_per_buffer=CHUNK)
            self.is_running = True
            print("Starting audio stream (Press Ctrl+C to stop)...")
            
            while self.is_running:
                audio_chunk = self.stream.read(CHUNK)
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)

                self.adaptive_noise_estimation(audio_np)

                processed_audio = self.wiener_filter(audio_np, self.noise_est)

                output_audio = processed_audio.astype(np.int16)
                self.stream.write(output_audio.tobytes())
                self.processed_audio_data.extend(output_audio)

        except Exception as e:
            print("Error occurred:", e)
            self.is_running = False
            messagebox.showerror("Error", f"An error occurred during audio processing: {e}")
        finally:
            self.stop_stream()

    def start_stream(self):
        """Start processing audio in a separate thread."""
        if not self.is_running:
            threading.Thread(target=self.process_audio_stream, daemon=True).start()

    def stop_stream(self):
        """Stop the audio processing stream."""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        self.save_to_wav()

    def save_to_wav(self):
        """Save the processed audio to a .wav file with a timestamp."""
        if not self.processed_audio_data:
            messagebox.showwarning("Warning", "No audio data to save.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_audio_{timestamp}.wav"

        try:
            wav.write(filename, RATE, np.array(self.processed_audio_data).astype(np.int16))
            print(f"Processed audio saved as {filename}")
            messagebox.showinfo("Success", f"Processed audio saved as {filename}")
        except Exception as e:
            print(f"Error saving file: {e}")
            messagebox.showerror("Error", f"Error saving file: {e}")

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Noise Cancellation System")
        self.root.geometry("400x300")

        self.nc_system = NoiseCancellationSystem()

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=20)

        self.parameter_frame = tk.Frame(self.root)
        self.parameter_frame.pack(pady=10)

        self.start_button = tk.Button(self.control_frame, text="Start", command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(self.control_frame, text="Stop", command=self.stop)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.alpha_label = tk.Label(self.parameter_frame, text="Noise Adaptation Factor (Alpha):")
        self.alpha_label.pack(side=tk.LEFT, padx=5)
        self.alpha_entry = tk.Entry(self.parameter_frame, width=5)
        self.alpha_entry.insert(0, str(self.nc_system.alpha))
        self.alpha_entry.pack(side=tk.LEFT, padx=5)

        self.update_alpha_button = tk.Button(self.parameter_frame, text="Update Alpha", command=self.update_alpha)
        self.update_alpha_button.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(self.root, text="Status: Not Running")
        self.status_label.pack(pady=10)

    def start(self):
        """Start the noise cancellation process."""
        if not self.nc_system.is_running:
            self.nc_system.start_stream()
            self.status_label.config(text="Status: Running")
            messagebox.showinfo("Info", "Noise cancellation started.")

    def stop(self):
        """Stop the noise cancellation process."""
        if self.nc_system.is_running:
            self.nc_system.stop_stream()
            self.status_label.config(text="Status: Not Running")
            messagebox.showinfo("Info", "Noise cancellation stopped.")

    def update_alpha(self):
        """Update the alpha value based on user input."""
        try:
            new_alpha = float(self.alpha_entry.get())
            if 0 <= new_alpha <= 1:
                self.nc_system.alpha = new_alpha
                messagebox.showinfo("Info", f"Alpha updated to {new_alpha}.")
            else:
                messagebox.showerror("Error", "Alpha must be between 0 and 1.")
        except ValueError:
            messagebox.showerror("Error", "Invalid alpha value.")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    