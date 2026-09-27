"""
StartupLex Backend API.
Flask application with JWT authentication, RAG-powered legal queries,
persistent chat sessions, and dynamic PDF document ingestion.
"""
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
from functools import wraps
import os
import jwt
from dotenv import load_dotenv

import db
import rag_service

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['JSON_SORT_KEYS'] = False
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
JWT_EXPIRY_DAYS = 7

# Initialize database tables
db.init_db()


# ──────────────────────────────── JWT Helpers ────────────────────────────────

def generate_jwt(user_id, email):
    """Generate a JWT token for a user."""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm='HS256')


def decode_jwt(token):
    """Decode and validate a JWT token. Returns payload dict or None."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return payload
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def auth_required(f):
    """Decorator that requires a valid JWT Bearer token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header:
            return jsonify({'error': 'Missing Authorization header'}), 401

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return jsonify({'error': 'Invalid Authorization header format'}), 401

        token = parts[1]
        payload = decode_jwt(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401

        user = db.get_user_by_id(payload.get('user_id'))
        if not user:
            return jsonify({'error': 'User not found'}), 401

        g.current_user = user
        return f(*args, **kwargs)

    return decorated


# ──────────────────────────────── Health Check ────────────────────────────────

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'active',
        'service': 'StartupLex Backend',
        'rag_available': rag_service.is_rag_available(),
        'timestamp': datetime.now().isoformat()
    }), 200


# ──────────────────────────────── Authentication ────────────────────────────────

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Register a new user and return a JWT token."""
    try:
        data = request.get_json() or {}
        name = data.get('name') or ''
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''

        if not email or not password:
            return jsonify({'error': 'Missing email or password'}), 400

        existing = db.get_user_by_email(email)
        if existing:
            return jsonify({'error': 'User with this email already exists'}), 409

        user = db.create_user(name, email, password)
        if not user:
            return jsonify({'error': 'Failed to create user'}), 500

        token = generate_jwt(user['id'], user['email'])

        return jsonify({
            'status': 'success',
            'user': {'id': user['id'], 'name': user['name'], 'email': user['email']},
            'token': token
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/signin', methods=['POST'])
def signin():
    """Sign in and return a JWT token."""
    try:
        data = request.get_json() or {}
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''

        if not email or not password:
            return jsonify({'error': 'Missing email or password'}), 400

        user = db.get_user_by_email(email)
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401

        if not db.verify_password(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401

        token = generate_jwt(user['id'], user['email'])

        return jsonify({
            'status': 'success',
            'user': {'id': user['id'], 'name': user['name'], 'email': user['email']},
            'token': token
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/auth/me', methods=['GET'])
@auth_required
def me():
    """Get the current authenticated user's info."""
    user = g.get('current_user')
    if not user:
        return jsonify({'error': 'Unauthorized'}), 401
    safe_user = {
        'id': user['id'],
        'name': user['name'],
        'email': user['email'],
        'created_at': user['created_at']
    }
    return jsonify({'status': 'success', 'user': safe_user}), 200


# ──────────────────────────────── Single Query (backward-compatible) ────────────────────────────────

@app.route('/api/query', methods=['POST'])
def handle_query():
    """
    Handle a single legal query using the RAG pipeline.
    Expected JSON: { "question": "user question" }
    """
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({'error': 'Missing required field: question'}), 400

        question = data.get('question')

        if not rag_service.is_rag_available():
            return jsonify({'error': 'RAG model not available. Check configuration.'}), 503

        answer = rag_service.run_query(question)

        return jsonify({
            'status': 'success',
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────── Multi-Turn Chat ────────────────────────────────

@app.route('/api/chat', methods=['POST'])
@auth_required
def chat():
    """
    Handle multi-turn conversation with persistent chat history.
    Expected JSON: { "question": "...", "session_id": "optional" }
    """
    try:
        data = request.get_json()

        if not data or 'question' not in data:
            return jsonify({'error': 'Missing required field: question'}), 400

        question = data.get('question')
        session_id = data.get('session_id')
        user_id = g.current_user['id']

        # Create a new session if none provided
        if not session_id:
            title = question[:50] + ('...' if len(question) > 50 else '')
            session = db.create_chat_session(user_id, title)
            session_id = session['id']

        # Save user message
        db.save_chat_message(session_id, 'user', question)

        # Retrieve chat history for context
        history_rows = db.get_session_messages(session_id)
        chat_history = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in history_rows[:-1]  # exclude the message we just saved
        ]

        # Get RAG response with history
        if not rag_service.is_rag_available():
            answer = "RAG model is not available. Please check server configuration."
        else:
            answer = rag_service.run_chat_query(question, chat_history)

        # Save assistant response
        db.save_chat_message(session_id, 'assistant', answer)

        return jsonify({
            'status': 'success',
            'session_id': session_id,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions', methods=['GET'])
@auth_required
def get_chat_sessions():
    """Get all chat sessions for the current user."""
    try:
        user_id = g.current_user['id']
        sessions = db.get_user_sessions(user_id)
        return jsonify({'status': 'success', 'sessions': sessions}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>/messages', methods=['GET'])
@auth_required
def get_session_messages(session_id):
    """Get all messages for a specific chat session."""
    try:
        messages = db.get_session_messages(session_id)
        return jsonify({'status': 'success', 'messages': messages}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
@auth_required
def delete_session(session_id):
    """Delete a chat session and its messages."""
    try:
        db.delete_chat_session(session_id)
        return jsonify({'status': 'success', 'message': 'Session deleted'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────── Document Upload ────────────────────────────────

@app.route('/api/documents/upload', methods=['POST'])
@auth_required
def upload_document():
    """
    Upload a PDF document to be ingested into the FAISS vector store.
    Accepts multipart form data with field name 'file'.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are accepted'}), 400

        file_bytes = file.read()
        result = rag_service.ingest_pdf_bytes(file_bytes, file.filename)

        return jsonify({
            'status': 'success',
            'chunks_added': result['chunks_added'],
            'filename': result['filename']
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ──────────────────────────────── Static Documents List ────────────────────────────────

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get available compliance documents."""
    documents = [
        {'id': 1, 'title': 'LLC Formation Guide', 'category': 'Formation'},
        {'id': 2, 'title': 'SAFE Agreements', 'category': 'Fundraising'},
        {'id': 3, 'title': 'Equity Split Guide', 'category': 'Cap Table'},
    ]
    return jsonify(documents), 200


@app.route('/api/documents/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get a specific document."""
    document = {
        'id': doc_id,
        'title': f'Document {doc_id}',
        'content': 'Document content here',
        'category': 'Legal'
    }
    return jsonify(document), 200


# ──────────────────────────────── Error Handlers ────────────────────────────────

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


# ──────────────────────────────── Entry Point ────────────────────────────────

if __name__ == '__main__':
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=5000)
