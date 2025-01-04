import pyaudio
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

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

        # Avoid division by zero
        noise_estimation = np.maximum(noise_estimation, 1e-10)

        # Compute the Wiener gain
        gain = (magnitude**2) / (magnitude**2 + noise_estimation**2)
        
        # Apply the gain to the original signal
        processed_magnitude = gain * magnitude
        processed_audio = processed_magnitude * np.exp(1j * phase)

        return np.fft.ifft(processed_audio).real

    def process_audio_stream(self):
        """Main loop for processing audio stream."""
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True,
                        frames_per_buffer=CHUNK)

        try:
            print("Starting audio stream (Press Ctrl+C to stop)...")
            while True:
                # Read audio chunk
                audio_chunk = stream.read(CHUNK)
                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)

                # Update noise estimation
                self.adaptive_noise_estimation(audio_np)

                # Perform noise cancellation
                processed_audio = self.wiener_filter(audio_np, self.noise_est)

                # Convert back to int16 for audio output
                output_audio = processed_audio.astype(np.int16)

                # Output the processed audio
                stream.write(output_audio.tobytes())
                self.processed_audio_data.extend(output_audio)

        except KeyboardInterrupt:
            print("Terminating...")

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Save the processed audio to a .wav file
            self.save_to_wav(np.array(self.processed_audio_data), "processed_audio.wav")

    @staticmethod
    def save_to_wav(processed_audio, filename):
        """Save the processed audio to a .wav file."""
        wav.write(filename, RATE, processed_audio.astype(np.int16))
        print(f"Processed audio saved as {filename}")

def main():
    system = NoiseCancellationSystem(alpha=0.9)
    system.process_audio_stream()

if __name__ == "__main__":
    main()