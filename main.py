"""
Book AI Backend Server
A Flask server that processes PDFs and answers questions using AI
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import io
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from your GitHub Pages site

# Initialize AI models (loaded once when server starts)
print("🤖 Loading AI models...")
search_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ AI models loaded!")

# In-memory storage (simple version - books stored while server runs)
books_db = {}
all_chunks = []
chunk_to_book = []
search_index = None


def split_text_into_chunks(text, chunk_size=500):
    """Split text into manageable chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def create_search_index():
    """Create FAISS search index from all chunks"""
    global search_index
    if not all_chunks:
        return False
    
    print(f"🔍 Creating search index for {len(all_chunks)} chunks...")
    embeddings = search_model.encode(all_chunks, show_progress_bar=False)
    
    dimension = embeddings.shape[1]
    search_index = faiss.IndexFlatL2(dimension)
    search_index.add(embeddings.astype('float32'))
    print("✅ Search index created!")
    return True


@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': '📚 Book AI Backend is running!',
        'books_loaded': len(books_db),
        'total_chunks': len(all_chunks)
    })


@app.route('/upload', methods=['POST'])
def upload_book():
    """Upload and process a PDF book"""
    try:
        # Check if file was sent
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read PDF
        print(f"📖 Processing: {file.filename}")
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text() + " "
        
        if not text.strip():
            return jsonify({'error': 'No text found in PDF'}), 400
        
        # Split into chunks
        chunks = split_text_into_chunks(text)
        
        # Store book
        book_id = len(books_db)
        books_db[file.filename] = {
            'id': book_id,
            'name': file.filename,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'size': len(text)
        }
        
        # Add to global chunks list
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_to_book.append(file.filename)
        
        # Rebuild search index
        create_search_index()
        
        print(f"✅ Added '{file.filename}' ({len(chunks)} chunks)")
        
        return jsonify({
            'success': True,
            'book_name': file.filename,
            'chunks': len(chunks),
            'total_books': len(books_db)
        })
    
    except Exception as e:
        print(f"❌ Error uploading book: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer a question about the loaded books"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        if not all_chunks or search_index is None:
            return jsonify({'error': 'No books loaded yet'}), 400
        
        print(f"❓ Question: {question}")
        
        # Search for relevant chunks
        question_embedding = search_model.encode([question])
        num_results = min(5, len(all_chunks))
        distances, indices = search_index.search(
            question_embedding.astype('float32'), 
            num_results
        )
        
        # Get relevant passages and which books they're from
        relevant_passages = []
        books_used = set()
        
        for idx in indices[0]:
            book_name = chunk_to_book[idx]
            passage = all_chunks[idx]
            books_used.add(book_name)
            relevant_passages.append({
                'text': passage[:300] + '...' if len(passage) > 300 else passage,
                'book': book_name
            })
        
        # Combine passages for context
        combined_text = " ".join([all_chunks[idx] for idx in indices[0]])
        
        # Limit context length
        if len(combined_text.split()) > 1000:
            combined_text = " ".join(combined_text.split()[:1000])
        
        # For now, return passages (you can add real AI generation later)
        answer = f"Based on {len(books_used)} book(s), here's what I found:\n\n"
        answer += f"The relevant information suggests: {combined_text[:500]}..."
        
        print(f"✅ Found answer from {len(books_used)} book(s)")
        
        return jsonify({
            'success': True,
            'answer': answer,
            'passages': relevant_passages,
            'books_used': list(books_used)
        })
    
    except Exception as e:
        print(f"❌ Error answering question: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/books', methods=['GET'])
def list_books():
    """Get list of all loaded books"""
    books_list = [
        {
            'name': book['name'],
            'chunks': book['chunk_count'],
            'size': f"{book['size'] // 1024} KB"
        }
        for book in books_db.values()
    ]
    
    return jsonify({
        'success': True,
        'books': books_list,
        'total_books': len(books_db),
        'total_chunks': len(all_chunks)
    })


@app.route('/which-book', methods=['POST'])
def which_book():
    """Find which books discuss a topic"""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        
        if not topic:
            return jsonify({'error': 'No topic provided'}), 400
        
        if not all_chunks or search_index is None:
            return jsonify({'error': 'No books loaded'}), 400
        
        # Search for topic
        topic_embedding = search_model.encode([topic])
        num_results = min(10, len(all_chunks))
        distances, indices = search_index.search(
            topic_embedding.astype('float32'),
            num_results
        )
        
        # Count mentions per book
        book_counts = {}
        for idx in indices[0]:
            book = chunk_to_book[idx]
            book_counts[book] = book_counts.get(book, 0) + 1
        
        # Sort by relevance
        sorted_books = sorted(
            book_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        results = [
            {'book': book, 'mentions': count}
            for book, count in sorted_books
        ]
        
        return jsonify({
            'success': True,
            'topic': topic,
            'results': results
        })
    
    except Exception as e:
        print(f"❌ Error finding books: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear_books():
    """Clear all books from memory"""
    global books_db, all_chunks, chunk_to_book, search_index
    
    books_db = {}
    all_chunks = []
    chunk_to_book = []
    search_index = None
    
    print("🗑️ All books cleared")
    
    return jsonify({
        'success': True,
        'message': 'All books cleared'
    })


if __name__ == '__main__':
    print("="*60)
    print("📚 Book AI Backend Server Starting...")
    print("="*60)
    # For Replit, use 0.0.0.0 and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
