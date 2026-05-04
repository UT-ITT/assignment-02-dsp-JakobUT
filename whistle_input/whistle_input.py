"""
Whistle Input - Detect whistled frequency chirps and trigger key presses
Detects upward (ooouuuiii) and downward (iiiuuuooo) frequency chirps
to navigate GUI menus using pynput.
"""

import numpy as np
import sounddevice as sd
from collections import deque
import threading
import time
from pynput.keyboard import Key, Controller
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Audio configuration
CHUNK_SIZE = 2048
RATE = 44100
CHANNELS = 1

# Chirp detection configuration
CHIRP_MIN_FREQ = 400  # Hz
CHIRP_MAX_FREQ = 2000  # Hz
CHIRP_MIN_DURATION = 0.3  # seconds
CHIRP_MAX_DURATION = 2.0  # seconds
FREQUENCY_CHANGE_THRESHOLD = 100  # Hz change to detect chirp
NOISE_THRESHOLD = 0.15  # Confidence threshold

# Buffers for chirp detection
BUFFER_SIZE = int(RATE * CHIRP_MAX_DURATION / CHUNK_SIZE)  # Number of chunks


class FrequencyDetector:
    """Detects frequency from audio using FFT with autocorrelation fallback"""
    
    def __init__(self, rate=RATE, chunk_size=CHUNK_SIZE):
        self.rate = rate
        self.chunk_size = chunk_size
        
    def detect_frequency(self, audio_chunk):
        """Detect dominant frequency using FFT"""
        # Apply Hanning window
        windowed = audio_chunk * np.hanning(len(audio_chunk))
        
        # Compute FFT
        spectrum = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1/self.rate)
        magnitude = np.abs(spectrum)
        
        # Normalize magnitude
        if np.max(magnitude) == 0:
            return 0, 0
        
        magnitude_normalized = magnitude / np.max(magnitude)
        
        # Filter frequency range
        mask = (freqs > CHIRP_MIN_FREQ) & (freqs < CHIRP_MAX_FREQ)
        if not np.any(mask):
            return 0, 0
        
        filtered_magnitude = magnitude_normalized[mask]
        filtered_freqs = freqs[mask]
        
        # Find peak frequency
        peak_idx = np.argmax(filtered_magnitude)
        peak_freq = filtered_freqs[peak_idx]
        peak_magnitude = filtered_magnitude[peak_idx]
        
        # Calculate confidence based on peak prominence
        # Compare peak to average of surrounding frequencies
        if len(filtered_magnitude) > 10:
            surrounding = np.concatenate([
                filtered_magnitude[max(0, peak_idx-5):peak_idx],
                filtered_magnitude[peak_idx+1:min(len(filtered_magnitude), peak_idx+6)]
            ])
            if len(surrounding) > 0:
                confidence = peak_magnitude - np.mean(surrounding)
            else:
                confidence = peak_magnitude
        else:
            confidence = peak_magnitude
        
        confidence = np.clip(confidence, 0, 1)
        
        return peak_freq, confidence
    
    def apply_noise_reduction(self, audio_chunk):
        """Simple noise reduction using spectral subtraction"""
        # Estimate noise from silent parts (not implemented in detail)
        # For now, just return the audio with amplitude thresholding
        threshold = 0.02
        audio_chunk[np.abs(audio_chunk) < threshold] = 0
        return audio_chunk


class ChirpDetector:
    """Detects upward and downward frequency chirps"""
    
    def __init__(self):
        self.frequency_buffer = deque(maxlen=BUFFER_SIZE)
        self.confidence_buffer = deque(maxlen=BUFFER_SIZE)
        self.time_buffer = deque(maxlen=BUFFER_SIZE)
        self.chunk_count = 0
        self.detector = FrequencyDetector()
        self.keyboard = Controller()
        self.lock = threading.Lock()
        
    def process_audio(self, audio_chunk):
        """Process audio chunk and detect chirps"""
        # Noise reduction
        audio_chunk = self.detector.apply_noise_reduction(audio_chunk)
        
        # Detect frequency
        freq, conf = self.detector.detect_frequency(audio_chunk)
        
        with self.lock:
            self.frequency_buffer.append(freq)
            self.confidence_buffer.append(conf)
            self.time_buffer.append(self.chunk_count * CHUNK_SIZE / RATE)
            self.chunk_count += 1
            
            # Check for chirps periodically
            if len(self.frequency_buffer) >= 10:
                self.analyze_for_chirps()
    
    def analyze_for_chirps(self):
        """Analyze buffer for upward/downward chirps"""
        # Filter out low confidence detections
        freqs = [f for f, c in zip(self.frequency_buffer, self.confidence_buffer) 
                 if c > NOISE_THRESHOLD and f > 0]
        
        if len(freqs) < 5:
            return
        
        # Calculate frequency change
        freq_array = np.array(freqs)
        
        # Fit line to detect trend
        x = np.arange(len(freq_array))
        coeffs = np.polyfit(x, freq_array, 1)
        slope = coeffs[0]  # Frequency change rate
        
        # Detect frequency range
        freq_range = np.max(freq_array) - np.min(freq_array)
        
        # Duration of detected signal
        duration = len(freqs) * CHUNK_SIZE / RATE
        
        # Upward chirp: frequency increasing significantly
        if slope > 20 and freq_range > 100 and duration > CHIRP_MIN_DURATION:
            print(f"↑ UPWARD CHIRP detected! (slope: {slope:.1f} Hz/s, duration: {duration:.2f}s)")
            self.trigger_up_key()
            self.frequency_buffer.clear()
            self.confidence_buffer.clear()
            self.time_buffer.clear()
        
        # Downward chirp: frequency decreasing significantly
        elif slope < -20 and freq_range > 100 and duration > CHIRP_MIN_DURATION:
            print(f"↓ DOWNWARD CHIRP detected! (slope: {slope:.1f} Hz/s, duration: {duration:.2f}s)")
            self.trigger_down_key()
            self.frequency_buffer.clear()
            self.confidence_buffer.clear()
            self.time_buffer.clear()
    
    def trigger_up_key(self):
        """Trigger UP arrow key press"""
        self.keyboard.press(Key.up)
        time.sleep(0.1)
        self.keyboard.release(Key.up)
    
    def trigger_down_key(self):
        """Trigger DOWN arrow key press"""
        self.keyboard.press(Key.down)
        time.sleep(0.1)
        self.keyboard.release(Key.down)
    
    def get_frequency_history(self):
        """Get current frequency history for visualization"""
        with self.lock:
            return list(self.frequency_buffer), list(self.confidence_buffer), list(self.time_buffer)


class WhistleInputApp:
    """Main Whistle Input Application"""
    
    def __init__(self, visualize=True):
        self.chirp_detector = ChirpDetector()
        self.is_running = True
        self.visualize = visualize
        self.stream = None
        self.setup_audio_stream()
        
        if visualize:
            self.setup_visualization()
    
    def setup_audio_stream(self):
        """Setup audio input stream"""
        def audio_callback(indata, frames, time_obj, status):
            if status:
                print(f"Audio status: {status}")
            
            audio_chunk = indata[:, 0].copy()
            self.chirp_detector.process_audio(audio_chunk)
        
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            latency='low'
        )
        self.stream.start()
    
    def setup_visualization(self):
        """Setup real-time frequency visualization"""
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Frequency plot
        self.line_freq, = self.ax1.plot([], [], lw=2, color='cyan')
        self.ax1.set_ylim(CHIRP_MIN_FREQ, CHIRP_MAX_FREQ)
        self.ax1.set_ylabel('Frequency (Hz)', fontsize=12)
        self.ax1.set_title('Detected Frequency Over Time', fontsize=14, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        
        # Confidence plot
        self.line_conf, = self.ax2.plot([], [], lw=2, color='lime')
        self.ax2.set_ylim(0, 1)
        self.ax2.set_ylabel('Confidence', fontsize=12)
        self.ax2.set_xlabel('Time (s)', fontsize=12)
        self.ax2.set_title('Detection Confidence', fontsize=14, fontweight='bold')
        self.ax2.axhline(y=NOISE_THRESHOLD, color='red', linestyle='--', 
                         label=f'Noise Threshold ({NOISE_THRESHOLD})', alpha=0.7)
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.02, 'Listening for chirps...', 
                                        ha='center', fontsize=12, 
                                        color='yellow', weight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        # Animation
        self.anim = FuncAnimation(self.fig, self.update_plot, interval=100, 
                                 blit=False, cache_frame_data=False)
        
        self.fig.show()
    
    def update_plot(self, frame):
        """Update visualization"""
        freqs, confs, times = self.chirp_detector.get_frequency_history()
        
        if freqs:
            # Convert times to relative time
            if times:
                times_rel = np.array(times) - times[0]
                
                self.line_freq.set_data(times_rel, freqs)
                self.line_conf.set_data(times_rel, confs)
                
                # Auto-scale x-axis
                if len(times_rel) > 0:
                    self.ax1.set_xlim(max(0, times_rel[-1] - 5), times_rel[-1])
                    self.ax2.set_xlim(max(0, times_rel[-1] - 5), times_rel[-1])
                
                # Update status
                if freqs[-1] > 0:
                    status = f"Current: {freqs[-1]:.0f} Hz | Confidence: {confs[-1]:.2f}"
                else:
                    status = "Listening for chirps... (Whistle now!)"
                self.status_text.set_text(status)
        
        return self.line_freq, self.line_conf, self.status_text
    
    def run(self):
        """Run the application"""
        print("\n" + "="*60)
        print("WHISTLE INPUT - Frequency Chirp Detection")
        print("="*60)
        print("\nInstructions:")
        print("1. Whistle an UPWARD chirp (low to high: ooouuuiii)")
        print("   → Triggers UP arrow key")
        print("\n2. Whistle a DOWNWARD chirp (high to low: iiiuuuooo)")
        print("   → Triggers DOWN arrow key")
        print("\n3. Use these to navigate menus")
        print("\nPress Ctrl+C to stop\n")
        print("="*60)
        
        print("\nAudio stream started. Listening for chirps...")
        
        try:
            if self.visualize:
                plt.show()
            else:
                # Run without visualization
                while self.is_running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Closed.")


def main():
    """Main entry point"""
    app = WhistleInputApp(visualize=True)
    app.run()


if __name__ == '__main__':
    main()
