"""
Book AI Backend Server - LIGHTWEIGHT VERSION
Uses smaller models that fit in Render's free tier
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import PyPDF2
import io
import json
import numpy as np
from collections import Counter
import re

app = Flask(__name__)
CORS(app)

# Simple in-memory storage
books_db = {}
all_chunks = []
chunk_to_book = []

# Simple text similarity using TF-IDF (no heavy AI models!)
def simple_tokenize(text):
    """Convert text to lowercase words"""
    return re.findall(r'\w+', text.lower())

def calculate_similarity(query_words, chunk_words):
    """Simple word overlap similarity"""
    query_set = set(query_words)
    chunk_set = set(chunk_words)
    
    if not query_set or not chunk_set:
        return 0
    
    overlap = len(query_set & chunk_set)
    return overlap / len(query_set)

def split_text_into_chunks(text, chunk_size=500):
    """Split text into manageable chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': '📚 Book AI Backend is running! (Lightweight version)',
        'books_loaded': len(books_db),
        'total_chunks': len(all_chunks),
        'version': 'lightweight'
    })

@app.route('/upload', methods=['POST'])
def upload_book():
    """Upload and process a PDF book"""
    try:
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
        
        if not all_chunks:
            return jsonify({'error': 'No books loaded yet'}), 400
        
        print(f"❓ Question: {question}")
        
        # Tokenize question
        query_words = simple_tokenize(question)
        
        # Calculate similarity for each chunk
        similarities = []
        for idx, chunk in enumerate(all_chunks):
            chunk_words = simple_tokenize(chunk)
            similarity = calculate_similarity(query_words, chunk_words)
            similarities.append((idx, similarity))
        
        # Get top 5 most similar chunks
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:5]]
        
        # Get relevant passages and books
        relevant_passages = []
        books_used = set()
        
        for idx in top_indices:
            book_name = chunk_to_book[idx]
            passage = all_chunks[idx]
            books_used.add(book_name)
            relevant_passages.append({
                'text': passage[:300] + '...' if len(passage) > 300 else passage,
                'book': book_name
            })
        
        # Combine passages for answer
        combined_text = " ".join([all_chunks[idx] for idx in top_indices])
        
        # Create simple answer
        if combined_text:
            answer = f"Based on {len(books_used)} book(s), here's what I found:\n\n"
            answer += combined_text[:500] + "..."
        else:
            answer = "I couldn't find relevant information in your books about that topic."
        
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
        
        if not all_chunks:
            return jsonify({'error': 'No books loaded'}), 400
        
        # Tokenize topic
        topic_words = simple_tokenize(topic)
        
        # Find books that mention the topic
        book_counts = {}
        for idx, chunk in enumerate(all_chunks):
            chunk_words = simple_tokenize(chunk)
            similarity = calculate_similarity(topic_words, chunk_words)
            
            if similarity > 0:  # Any overlap
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
    global books_db, all_chunks, chunk_to_book
    
    books_db = {}
    all_chunks = []
    chunk_to_book = []
    
    print("🗑️ All books cleared")
    
    return jsonify({
        'success': True,
        'message': 'All books cleared'
    })

if __name__ == '__main__':
    print("="*60)
    print("📚 Book AI Backend Server Starting...")
    print("Version: Lightweight (no heavy AI models)")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)
```




