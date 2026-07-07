# 🖊️ SmartPen

SmartPen is an AI-powered voice-controlled desktop assistant built with Python. It enables users to control presentations, perform live voice typing, draw mathematical shapes, automate keyboard and mouse actions, search files, launch applications, and execute voice commands—all primarily offline using Faster-Whisper.

---

## ✨ Features

- 🎤 Offline speech recognition using Faster-Whisper
- 📝 Real-time live voice typing
- 🎯 Voice Activity Detection (VAD)
- 🔇 Noise reduction for improved transcription
- 📑 Presentation mode with voice-controlled slide navigation
- 🎨 Draw mathematical shapes using voice commands
- 🤖 AI search support (ChatGPT, Gemini, Copilot)
- 🌐 Open websites and desktop applications
- 📂 Smart file search
- ⌨️ Keyboard automation (copy, paste, undo, save, etc.)
- 🖱️ Mouse automation
- 😴 Wake/Sleep voice commands
- ⚙️ Configurable settings through `smart_pen_config.json`

---

## 📂 Project Structure

```
SMARTPEN/
│
├── model/
├── offPen.py
├── smart_pen_config.json
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Requirements

- Python 3.10 or later
- Windows 10/11 (Recommended)
- Working microphone
- FFmpeg (required by Faster-Whisper)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/SmartPen.git
cd SmartPen
```

Create a virtual environment (recommended):

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
python offPen.py
```

---

## 🎙️ Example Voice Commands

### Typing

- Start typing
- New paragraph
- Delete word
- Undo
- Copy
- Paste
- Save document

### Presentation

- Next slide
- Previous slide
- First slide
- Last slide
- Start presentation
- End presentation
- Add slide

### Drawing

- Draw circle
- Draw rectangle
- Draw square
- Draw triangle
- Draw ellipse
- Draw pentagon
- Draw hexagon

### Applications

- Open Chrome
- Open Notepad
- Open Calculator
- Open YouTube
- Open VS Code

### AI Search

- Search Python decorators on ChatGPT
- Search Machine Learning on Gemini
- Ask Copilot how to create an API

---

## 🌐 Internet Requirement

Most SmartPen features work completely **offline**, including:

- Offline speech recognition
- Live voice typing
- Presentation control
- Drawing tools
- File search
- Keyboard and mouse automation
- Opening local applications

Internet is required only for:

- AI search (ChatGPT, Gemini, Copilot)
- Opening websites
- Downloading the Whisper model on the first run (if it is not already available)

Once the Whisper model is downloaded, speech recognition works offline.

---

## 📚 Main Libraries Used

- Faster-Whisper
- NumPy
- SoundDevice
- PyAutoGUI
- WebRTCVAD
- Noisereduce
- DeepMultilingualPunctuation
- PyGetWindow
- Keyboard
- PyStray
- Pillow
- CTranslate2
- ONNX Runtime
- Tokenizers
- AV

---

## ⚠️ Notes

- A microphone is required.
- Some automation features may require administrator/accessibility permissions depending on the operating system.
- The project is primarily developed and tested on Windows.

---

## 🚀 Future Improvements

- Custom wake word
- Multiple language support
- Voice authentication
- Custom command creation
- GUI dashboard
- Plugin support

---

## 👨‍💻 Author

**Ankur Sharma**
