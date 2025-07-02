🎬 Automatic Video Dubbing for Indian Regional Languages

A **Flask-based web application** that allows users to upload a video and get an automatically dubbed version in a selected **Indian language**. It uses **automatic speech recognition (ASR)**, **translation**, and **text-to-speech (TTS)** to generate the dubbed video.

---

## ✅ Key Features

- **User-friendly web interface** to upload and process videos
- **ASR (Speech-to-Text)** using tools like Whisper
- **Translation** to Indian languages (Hindi, Urdu, Bengali, etc.)
- **TTS (Text-to-Speech)** for generating dubbed audio
- **Audio merging** with the original video using FFmpeg
- **Output video** available for instant download

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **ASR:** OpenAI Whisper or SpeechRecognition
- **Translation:** Google Translate API
- **TTS:** gTTS or Amazon Polly
- **Audio/Video Handling:** FFmpeg, moviepy
- **Frontend:** HTML, CSS, JavaScript

---

## 🚀 How to Run Locally

bash
# 1. Clone the repository
git clone https://github.com/Abdulmoiz918/video-dubbing-project.git
cd video-dubbing-project

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py

Then open your browser and go to: http://localhost:5000

⸻

📁 Folder Structure

video-dubbing-project/
├── app.py
├── requirements.txt
├── templates/
│   ├── index.html
│   └── upload_result.html
├── uploads/          # (git-ignored) uploaded videos
├── downloads/        # (git-ignored) dubbed output videos
├── .gitignore
└── README.md


⸻

🔄 Workflow
	1.	User uploads a video through the web interface
	2.	Audio is extracted using FFmpeg
	3.	Speech is transcribed to text (ASR)
	4.	Text is translated to the target language
	5.	Translated text is converted to speech (TTS)
	6.	New audio is merged back into the original video
	7.	Final dubbed video is available for download

⸻

🤝 Contributing

Want to improve the project? Feel free to:
	•	Fork the repository
	•	Create a new branch (git checkout -b feature-name)
	•	Commit your changes
	•	Push and open a Pull Request

⸻

📄 License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute with credit.

⸻


✅ You can **paste this directly into GitHub’s “Add README” editor**, and it will render beautifully with bold headings, code blocks, and bullets.

Let me know if you want to add a screenshot, video demo, or badges next!
