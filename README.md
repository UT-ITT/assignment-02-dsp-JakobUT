[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/yaOQIQlj)

# Assignment 02 - Digital Signal Processing

This assignment implements real-time audio signal processing for two interactive applications:
1. **Karaoke Game** - Frequency detection based singing game
2. **Whistle Input** - Chirp detection for menu navigation

## Setup

### Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 1. Karaoke Game (`karaoke_game/karaoke.py`)

A real-time karaoke game that detects your sung frequency and compares it to target notes.

### Features
- **Real-time Frequency Detection**: Uses FFT with windowing for accurate frequency detection
- **19 Note Melody**: Implements "Ode to Joy" melody with 19 notes
- **Scoring System**: Awards points based on accuracy of detected frequency
- **Pyglet Interface**: Real-time visualization of:
  - Target note
  - Detected frequency
  - Progress bar
  - Current score
  - Correctly sung notes counter

### How to Run
```bash
cd karaoke_game
python karaoke.py
```

### Game Instructions
1. The game displays a target note to sing
2. Sing the note into your microphone
3. The game detects your frequency and compares it to the target
4. If close enough (within 50 Hz tolerance), you get points!
5. Complete all 19 notes to finish the song

### Technical Details
- **Audio Configuration**:
  - Chunk size: 2048 samples (low latency)
  - Sample rate: 44,100 Hz
  - Frequency range: 80-400 Hz
  - Note tolerance: ±50 Hz
  - Confidence threshold: > 0.3

- **Scoring**:
  - Base points: 100 per note
  - Deduction: 1 point per Hz off
  - Total possible: ~1,900 points

## 2. Whistle Input (`whistle_input/whistle_input.py`)

Detects whistled frequency chirps (upward and downward) to control GUI menus.

### Features
- **Chirp Detection**: 
  - **Upward chirps** (ooouuuiii) → UP arrow key
  - **Downward chirps** (iiiuuuooo) → DOWN arrow key
- **Noise Robustness**: 
  - Spectral analysis with confidence thresholding
  - Filters out background noise
  - Minimum confidence threshold: 0.15
- **Real-time Visualization**:
  - Frequency trend display
  - Confidence plot
  - Status updates
- **Keyboard Simulation**: Uses `pynput` to trigger arrow keys

### How to Run
```bash
cd whistle_input
python whistle_input.py
```

### Usage Instructions
1. **Upward Chirp** (low frequency → high frequency)
   - Whistle: "ooouuuiii" (transition from low to high)
   - Triggers: UP arrow key
   - Use for: Navigate up in menus

2. **Downward Chirp** (high frequency → low frequency)
   - Whistle: "iiiuuuooo" (transition from high to low)
   - Triggers: DOWN arrow key
   - Use for: Navigate down in menus

### Technical Details
- **Frequency Analysis**:
  - Detection range: 400-2000 Hz
  - Window function: Hanning window
  - FFT-based frequency estimation
  - Autocorrelation fallback for robustness

- **Chirp Detection**:
  - Minimum frequency change: 20 Hz/second
  - Minimum total frequency change: 100 Hz
  - Minimum duration: 0.3 seconds
  - Noise threshold: 0.15 confidence

- **Noise Reduction**:
  - Amplitude thresholding (0.02)
  - Spectral analysis to exclude noise
  - Confidence-based filtering

## Audio Devices

Both programs will list available audio input devices on startup. Make sure your microphone is properly connected and selected.

## Troubleshooting

### No audio input detected
- Check microphone connection
- Verify microphone is selected in OS audio settings
- Try different input device when prompted

### Frequency not detected
- Sing/whistle louder (within 80-400 Hz for karaoke, 400-2000 Hz for whistle)
- Check microphone levels
- Try different audio device
- Reduce background noise

### Low accuracy
- Maintain consistent frequency while singing/whistling
- Avoid background music or ambient noise
- Use a better microphone
- Sing clearly and with sufficient volume

## Dependencies
- **sounddevice**: Real-time audio I/O
- **numpy**: Numerical computing (FFT, signal processing)
- **pyglet**: Graphics and game window (karaoke)
- **matplotlib**: Real-time visualization (whistle input)
- **mido**: MIDI file support (for future MIDI-based melody)
- **pynput**: Keyboard event simulation (whistle input)

## Performance Notes
- Latency: ~50-100ms between sound and detection
- FFT window: 2048 samples @ 44.1kHz ≈ 46ms
- Thread-safe audio processing
- Optimized for real-time performance

## Future Enhancements
- Load custom MIDI files for different songs
- Multiple difficulty levels
- Recording and playback
- More sophisticated noise filtering
- Spectral subtraction for background noise removal
- Voice activity detection
