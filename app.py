import os
import uuid
import numpy as np
import faiss
from flask import Flask, request, jsonify, send_file, render_template
from gtts import gTTS
from sentence_transformers import SentenceTransformer
import PyPDF2
import re

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
PDF_PATH = "notes.pdf"
AUDIO_DIR = os.path.join("static", "audio")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3                    # top chunks to retrieve
CHUNK_SIZE = 3               # sentences per chunk
FALLBACK_ANSWER = (
    "I'm sorry, I couldn't find a relevant answer in the document. "
    "Please try rephrasing your question."
)

os.makedirs(AUDIO_DIR, exist_ok=True)

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PDF Loading & Text Chunking
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text into overlapping sentence-level chunks."""
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks = []
    for i in range(0, len(sentences), max(1, chunk_size - 1)):
        chunk = " ".join(sentences[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Startup: Build FAISS Index
# ─────────────────────────────────────────────────────────────────────────────

print("[*] Loading sentence-transformer model ...")
embedder = SentenceTransformer(MODEL_NAME)

print(f"[*] Reading PDF: {PDF_PATH} ...")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        f"'{PDF_PATH}' not found. Place your PDF in the project root and restart."
    )

raw_text = extract_text_from_pdf(PDF_PATH)
chunks = split_into_chunks(raw_text, CHUNK_SIZE)
print(f"[+] Extracted {len(chunks)} chunks from PDF.")

print("[*] Building FAISS index ...")
chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
chunk_embeddings = chunk_embeddings.astype("float32")

# Normalise for cosine similarity via inner product
faiss.normalize_L2(chunk_embeddings)
dimension = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)   # Inner Product == cosine after normalisation
faiss_index.add(chunk_embeddings)
print(f"[+] FAISS index ready ({faiss_index.ntotal} vectors, dim={dimension}).")


# ─────────────────────────────────────────────────────────────────────────────
# RAG Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_answer(question: str, top_k: int = TOP_K) -> str:
    """Embed the question, search FAISS, and return the best-matching chunk."""
    q_emb = embedder.encode([question], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = faiss_index.search(q_emb, top_k)

    if indices[0][0] == -1:
        return FALLBACK_ANSWER

    # Return the single best chunk (highest cosine score)
    best_idx = indices[0][0]
    best_score = scores[0][0]

    if best_score < 0.25:           # similarity too low → no relevant content
        return FALLBACK_ANSWER

    return chunks[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Audio Generation
# ─────────────────────────────────────────────────────────────────────────────

# Keep track of the latest audio file so /audio always serves the right one
_latest_audio: dict = {"path": None}


def text_to_speech(text: str) -> str:
    """Convert text to MP3 via gTTS and return the file path."""
    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(filepath)
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400

    answer = retrieve_answer(question)

    # Generate TTS audio
    audio_path = text_to_speech(answer)
    _latest_audio["path"] = audio_path

    return jsonify({"answer": answer, "audio_url": "/audio"})


@app.route("/audio")
def audio():
    path = _latest_audio.get("path")
    if not path or not os.path.exists(path):
        return jsonify({"error": "No audio available."}), 404
    return send_file(path, mimetype="audio/mpeg")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
