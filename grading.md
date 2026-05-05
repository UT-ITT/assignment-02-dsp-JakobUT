# JakobUT (11.5/15P)

## 1 - Karaoke Game (4.5/7P)
* frequency detection works correctly and robustly
    * tracking is not always correct (2P)
* the game is playable, does not crash, and is (kind of) fun to play
    * yep (2P)
* the game tracks some kind of score for correctly sung notes
    * yep (1P)
* low latency between input and detection
    * yep (1P)

Had to heavily debug your code to get the script starting (-1.5P)
* Eb4 missing
* import OpenGL is missing
* "bold" doesn't work - we had do comment that out
* microphone selection is hard coded, that's not good

## 2 - Whistle Input (6/7P)
* upwards and downwards whistling is detected correctly and robustly
    * sometimes, but mostly (2.5P)
*  detection is robust against background noise
    * speaking triggeres the input sometimes (1.5P)
* low latency between input and detection
    * yep (1P)
* triggered key events work
    * yep (1P)


## Code-Quality and .venv used (1/1P)
* really detailled readme :)