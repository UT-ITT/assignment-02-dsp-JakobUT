"""
Karaoke Game - Real-time frequency detection based game using pyglet
Detects the user's sung frequency and compares it to target notes.
"""

import numpy as np
import sounddevice as sd
import pyglet
from pyglet.gl import *
import mido
from mido import MidiFile
from collections import deque
import threading
from datetime import datetime

# Audio configuration
CHUNK_SIZE = 2048  # Smaller for lower latency
RATE = 44100
CHANNELS = 1

# Game configuration
TARGET_TOLERANCE = 50  # Hz tolerance for note matching
NOTE_TIMEOUT = 0.8  # Seconds to detect a note
MIN_FREQUENCY = 80  # Hz - below this, ignore
MAX_FREQUENCY = 400  # Hz - above this, ignore

# Musical note frequencies (Hz)
NOTE_FREQUENCIES = {
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
    'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46,
    'G5': 783.99, 'A5': 880.00, 'B5': 987.77
}

def read_midi_file(filepath):
    """Read MIDI file and extract note frequencies"""
    notes = []
    mid = MidiFile(filepath)
    for msg in mid.play():
        if msg.type == 'note_on' and msg.velocity > 0:
            # Convert MIDI note number to frequency
            frequency = 440 * (2 ** ((msg.note - 69) / 12))
            notes.append(frequency)
    return notes

def get_closest_note(frequency, notes_dict=NOTE_FREQUENCIES):
    """Find the closest note frequency to the detected frequency"""
    if not notes_dict:
        return None
    min_diff = float('inf')
    closest_note = None
    for note_name, note_freq in notes_dict.items():
        diff = abs(frequency - note_freq)
        if diff < min_diff:
            min_diff = diff
            closest_note = (note_name, note_freq)
    return closest_note

class FrequencyDetector:
    """Detects fundamental frequency from audio using FFT"""
    
    def __init__(self, rate=RATE, chunk_size=CHUNK_SIZE):
        self.rate = rate
        self.chunk_size = chunk_size
        self.current_frequency = 0
        self.confidence = 0
        
    def detect_frequency(self, audio_chunk):
        """Detect dominant frequency using FFT"""
        # Apply Hanning window to reduce spectral leakage
        windowed = audio_chunk * np.hanning(len(audio_chunk))
        
        # Compute FFT
        spectrum = np.fft.rfft(windowed)
        freqs = np.fft.rfftfreq(len(windowed), 1/self.rate)
        magnitude = np.abs(spectrum)
        
        # Filter out very low frequencies
        mask = freqs > MIN_FREQUENCY
        magnitude = magnitude[mask]
        freqs = freqs[mask]
        
        if len(magnitude) == 0:
            return 0, 0
        
        # Find peak frequency
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx]
        peak_magnitude = magnitude[peak_idx]
        
        # Normalize confidence (0-1)
        confidence = min(peak_magnitude / np.max(magnitude), 1.0)
        
        # Only return frequency if confidence is high enough
        if confidence > 0.3 and MIN_FREQUENCY < peak_freq < MAX_FREQUENCY:
            return peak_freq, confidence
        return 0, 0

class KaraokeGame:
    """Main Karaoke Game class"""
    
    def __init__(self, notes_source='list'):
        """
        Initialize the karaoke game.
        notes_source: 'list' for predefined notes, 'midi' for MIDI file
        """
        self.notes_source = notes_source
        self.target_notes = []
        self.current_note_idx = 0
        self.score = 0
        self.correct_count = 0
        self.total_count = 0
        self.note_start_time = None
        self.detected_frequency_buffer = deque(maxlen=10)
        
        # Load notes
        if notes_source == 'midi':
            try:
                self.target_notes = read_midi_file('freude.mid')
            except FileNotFoundError:
                print("MIDI file not found, using default list")
                self.use_default_notes()
        else:
            self.use_default_notes()
            
        self.total_count = len(self.target_notes)
        self.detector = FrequencyDetector()
        self.audio_lock = threading.Lock()
        self.is_running = True
        
        # Setup pyglet window
        self.window = pyglet.window.Window(800, 600, caption='Karaoke Game')
        self.window.set_background_color(0.1, 0.1, 0.2, 1)
        
        # Setup audio stream
        self.setup_audio_stream()
        
        # Pyglet event handlers
        @self.window.event
        def on_draw():
            self.draw()
            
    def use_default_notes(self):
        """Use a default list of 15+ notes"""
        # "Freude" melody (Beethoven)
        melody = ['G4', 'G4', 'G4', 'Eb4', 'F4',
                  'G4', 'D4', 'Eb4', 'F4', 'G4',
                  'G4', 'G4', 'G4', 'Eb4', 'F4',
                  'G4', 'D4', 'Eb4', 'F4']
        self.target_notes = [NOTE_FREQUENCIES[note] for note in melody]
        
    def setup_audio_stream(self):
        """Setup the audio input stream"""
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            
            with self.audio_lock:
                audio_chunk = indata[:, 0].copy()
                freq, conf = self.detector.detect_frequency(audio_chunk)
                self.detected_frequency_buffer.append((freq, conf))
        
        self.stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=RATE,
            blocksize=CHUNK_SIZE,
            callback=audio_callback,
            latency='low'
        )
        self.stream.start()
        
    def get_average_frequency(self):
        """Get average frequency from buffer"""
        with self.audio_lock:
            if not self.detected_frequency_buffer:
                return 0, 0
            freqs = [f for f, c in self.detected_frequency_buffer if f > 0]
            confs = [c for f, c in self.detected_frequency_buffer if f > 0]
            if not freqs:
                return 0, 0
            return np.mean(freqs), np.mean(confs)
    
    def update_game_state(self):
        """Update game state based on detected frequency"""
        if self.current_note_idx >= len(self.target_notes):
            return  # Game finished
        
        detected_freq, confidence = self.get_average_frequency()
        
        if detected_freq == 0:
            return
        
        target_freq = self.target_notes[self.current_note_idx]
        
        # Initialize note timer
        if self.note_start_time is None:
            self.note_start_time = datetime.now()
        
        current_time = datetime.now()
        note_duration = (current_time - self.note_start_time).total_seconds()
        
        # Check if frequency matches target
        if abs(detected_freq - target_freq) < TARGET_TOLERANCE and confidence > 0.4:
            self.score += max(0, 100 - int(abs(detected_freq - target_freq)))
            self.correct_count += 1
            self.advance_to_next_note()
        
        # Timeout - move to next note anyway
        elif note_duration > NOTE_TIMEOUT:
            self.advance_to_next_note()
    
    def advance_to_next_note(self):
        """Move to the next note"""
        self.current_note_idx += 1
        self.note_start_time = None
    
    def draw(self):
        """Draw game UI using pyglet"""
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Title
        title = pyglet.text.Label(
            'KARAOKE GAME',
            font_size=32,
            bold=True,
            x=400, y=550,
            anchor_x='center', anchor_y='top',
            color=(255, 200, 100, 255)
        )
        title.draw()
        
        # Current note display
        if self.current_note_idx < len(self.target_notes):
            target_freq = self.target_notes[self.current_note_idx]
            detected_freq, confidence = self.get_average_frequency()
            
            # Find closest note
            closest = get_closest_note(target_freq)
            closest_detected = get_closest_note(detected_freq) if detected_freq > 0 else None
            
            target_label = f"Target: {closest[0] if closest else 'N/A'} ({target_freq:.1f} Hz)"
            current_label = f"Detected: {closest_detected[0] if closest_detected else 'Listening...'} ({detected_freq:.1f} Hz)"
            
            target_text = pyglet.text.Label(
                target_label,
                font_size=24,
                x=400, y=450,
                anchor_x='center',
                color=(100, 255, 100, 255)
            )
            target_text.draw()
            
            current_text = pyglet.text.Label(
                current_label,
                font_size=20,
                x=400, y=400,
                anchor_x='center',
                color=(100, 100, 255, 255)
            )
            current_text.draw()
            
            # Progress bar
            progress = self.current_note_idx / max(len(self.target_notes), 1)
            bar_width = 600
            bar_x = 100
            bar_y = 300
            
            # Background bar
            glColor3f(0.3, 0.3, 0.3)
            glBegin(GL_QUADS)
            glVertex2f(bar_x, bar_y)
            glVertex2f(bar_x + bar_width, bar_y)
            glVertex2f(bar_x + bar_width, bar_y + 20)
            glVertex2f(bar_x, bar_y + 20)
            glEnd()
            
            # Progress bar
            glColor3f(0, 1, 0.5)
            glBegin(GL_QUADS)
            glVertex2f(bar_x, bar_y)
            glVertex2f(bar_x + bar_width * progress, bar_y)
            glVertex2f(bar_x + bar_width * progress, bar_y + 20)
            glVertex2f(bar_x, bar_y + 20)
            glEnd()
        
        # Score display
        score_text = pyglet.text.Label(
            f"Score: {self.score}",
            font_size=28,
            bold=True,
            x=400, y=200,
            anchor_x='center',
            color=(255, 255, 0, 255)
        )
        score_text.draw()
        
        # Notes progress
        progress_text = pyglet.text.Label(
            f"Notes: {self.current_note_idx} / {len(self.target_notes)}  |  Correct: {self.correct_count}",
            font_size=16,
            x=400, y=150,
            anchor_x='center',
            color=(200, 200, 200, 255)
        )
        progress_text.draw()
        
        # Game over message
        if self.current_note_idx >= len(self.target_notes):
            game_over = pyglet.text.Label(
                'SONG FINISHED!',
                font_size=36,
                bold=True,
                x=400, y=80,
                anchor_x='center',
                color=(100, 255, 100, 255)
            )
            game_over.draw()
            
            final_score = pyglet.text.Label(
                f"Final Score: {self.score} / {len(self.target_notes) * 100}",
                font_size=20,
                x=400, y=30,
                anchor_x='center',
                color=(255, 200, 100, 255)
            )
            final_score.draw()

def main():
    """Main game loop"""
    print("Starting Karaoke Game...")
    print(f"Available notes: {list(NOTE_FREQUENCIES.keys())}")
    print("Using predefined melody (Freude)...")
    
    game = KaraokeGame(notes_source='list')
    
    def update(dt):
        game.update_game_state()
    
    pyglet.clock.schedule_interval(update, 0.05)  # Update every 50ms
    
    try:
        pyglet.app.run()
    finally:
        game.stream.stop()
        game.stream.close()

if __name__ == '__main__':
    main()
