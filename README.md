# AI-Driven Minutes of Meeting (MOM) System

An intelligent web application that automates the process of capturing, transcribing, and summarizing meetings in real-time.

## Features

- Real-time audio capture and transcription
- Automatic action item extraction
- Meeting summarization
- Live updates via WebSocket
- Audio file upload support
- MongoDB storage for meeting data
- User-friendly web interface

## Prerequisites

- Python 3.8+
- MongoDB
- Virtual environment (recommended)
- System audio dependencies (for PyAudio)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MOM
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file:
```bash
touch .env
```
Add the following configurations:
```
MONGODB_URI=mongodb://localhost:27017/
DB_NAME=mom_db
```

5. Initialize the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
MOM/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── models.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
├── services/
│   ├── audio_service.py
│   ├── transcription_service.py
│   ├── nlp_service.py
│   └── storage_service.py
├── config.py
├── requirements.txt
└── run.py
```

## Usage

1. Start the MongoDB service
2. Run the application:
```bash
python run.py
```
3. Access the web interface at `http://localhost:5000`

## Features in Detail

### Real-time Transcription
- Uses OpenAI's Whisper for accurate speech-to-text conversion
- Supports multiple audio input sources
- Real-time display of transcribed text

### Action Item Extraction
- Automatically identifies and extracts action items from the transcription
- Uses spaCy's NLP capabilities for entity recognition
- Assigns responsibility and deadlines when mentioned

### Meeting Summarization
- Generates concise meeting summaries
- Highlights key discussion points
- Includes list of participants and decisions made

### Data Storage
- Stores all meeting data in MongoDB
- Enables easy retrieval and search of past meetings
- Maintains relationships between transcripts, action items, and summaries

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 