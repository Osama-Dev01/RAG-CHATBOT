from flask import Flask, render_template, request, jsonify
import requests
import uuid

app = Flask(__name__)

# FastAPI backend
BACKEND_URL = "http://localhost:8000/api/v1"

# session ID for the chat session
SESSION_ID = str(uuid.uuid4())

@app.route('/')
def index():
    return render_template('index.html', session_id=SESSION_ID)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"success": False, "detail": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "detail": "No selected file"}), 400

    try:
        # Forward the file payload to your FastAPI microservice
        files = {'file': (file.filename, file.stream, file.mimetype)}
        
        # FIXED: Added timeout=300 to allow up to 5 minutes for heavy document parsing (Docling + Embeddings)
        response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
        
        # Safely unpack the status codes for downstream rendering blocks
        if response.status_code != 200:
            try:
                error_info = response.json()
                detail_msg = error_info.get("detail", "Processing error")
            except Exception:
                detail_msg = response.text or "Processing error"
            return jsonify({"success": False, "detail": detail_msg}), response.status_code
            
        return jsonify(response.json())

    except requests.exceptions.Timeout:
        return jsonify({"success": False, "detail": "The server took too long to process this large document. Please wait a moment or try a smaller file."}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"success": False, "detail": f"Backend connection failed: {str(e)}"}), 502

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    payload = {
        "question": data.get("question"),
        "session_id": data.get("session_id")
    }
    # Added defensive timeout for chats as well
    response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
    return jsonify(response.json())

@app.route('/history/<session_id>', methods=['GET'])
def history(session_id):
    response = requests.get(f"{BACKEND_URL}/history/{session_id}", timeout=30)
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(port=5000, debug=True)