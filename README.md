# VoiceIQ — Voice-Powered Q&A Agent

Ask questions about **any PDF** using your voice or keyboard.  
The app retrieves the most relevant passage via a **RAG pipeline** (FAISS + sentence-transformers) and reads the answer back to you with **text-to-speech**.

---

## Tech Stack

| Layer | Library |
|---|---|
| Web framework | Flask |
| PDF extraction | PyPDF2 |
| Sentence embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`faiss-cpu`) |
| Text-to-speech | gTTS |
| Voice input (browser) | Web Speech API |

---

## Project Structure

```
voiceagent/
├── app.py               ← Flask backend + RAG pipeline
├── notes.pdf            ← Your PDF document (place here)
├── requirements.txt
├── README.md
├── templates/
│   └── index.html       ← Single-page UI
└── static/
    └── audio/           ← Auto-created; stores generated MP3 files
```

---

## Setup & Run

### 1. Prerequisites

- Python 3.10 or newer
- `pip` (or `pip3`)

### 2. Install dependencies

```bash
# (optional) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~90 MB).  
> Subsequent runs use the cached model.

### 3. Add your PDF

Place your document in the project root and name it **`notes.pdf`**.  
You can change the filename in `app.py` → `PDF_PATH`.

### 4. Start the server

```bash
python app.py
```

You should see:

```
🔄  Loading sentence-transformer model …
🔄  Reading PDF: notes.pdf …
✅  Extracted N chunks from PDF.
🔄  Building FAISS index …
✅  FAISS index ready  (N vectors, dim=384).
 * Running on http://127.0.0.1:5000
```

### 5. Open the app

Navigate to **http://localhost:5000** in **Chrome** or **Edge** (required for Web Speech API).

---

## How It Works

```
User speaks / types question
        │
        ▼
Flask POST /ask
        │
        ├─ Embed question with all-MiniLM-L6-v2
        ├─ FAISS cosine search → top-K chunks
        ├─ Return best matching chunk as answer
        │
        ├─ gTTS converts answer → MP3 saved to static/audio/
        └─ JSON { "answer": "...", "audio_url": "/audio" }

Browser receives response
        │
        ├─ Displays answer text
        └─ Auto-plays MP3 via <audio> element
```

---

## API Reference

### `POST /ask`

**Request body (JSON):**
```json
{ "question": "What is the main topic of this document?" }
```

**Response (JSON):**
```json
{
  "answer": "The document covers …",
  "audio_url": "/audio"
}
```

### `GET /audio`

Streams the most recently generated MP3 file.

---

## Configuration

Edit the constants at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `PDF_PATH` | `"notes.pdf"` | Path to the PDF file |
| `MODEL_NAME` | `"all-MiniLM-L6-v2"` | Sentence-transformer model |
| `TOP_K` | `3` | Number of chunks to retrieve |
| `CHUNK_SIZE` | `3` | Sentences per chunk |

---

## Browser Compatibility

| Feature | Chrome | Edge | Firefox | Safari |
|---|:---:|:---:|:---:|:---:|
| Web Speech API (mic) | ✅ | ✅ | ❌ | ❌ |
| Audio playback | ✅ | ✅ | ✅ | ✅ |

> Use **Chrome** or **Edge** for full voice functionality.  
> Keyboard input works in all browsers.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: notes.pdf` | Place your PDF in the project root |
| Mic not working | Allow microphone in browser settings; use Chrome/Edge |
| Slow first startup | Downloading model (~90 MB) on first run; wait for completion |
| `faiss-cpu` install fails | Ensure you're using Python 3.10+ on a 64-bit OS |
