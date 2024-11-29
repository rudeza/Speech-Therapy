from flask import Flask, request, render_template, jsonify
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector
from matplotlib.ticker import MaxNLocator

app = Flask(__name__)

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Chainsawman21',  # Replace with your MySQL password
    'database': 'speech_analysis'
}

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


def extract_speech_features(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)

    # 1. Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # 2. Average Pitch
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]  # Valid pitch values
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0

    # 3. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    avg_spectral_centroid = np.mean(spectral_centroid)

    # 4. Zero Crossing Rate
    zero_crossings = librosa.feature.zero_crossing_rate(y=y)
    avg_zero_crossings = np.mean(zero_crossings)

    # 5. Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma)

    return tempo, avg_pitch, avg_spectral_centroid, avg_zero_crossings, chroma_mean

def save_audio_record_with_features(user_id, audio_file_path, features):
    tempo, avg_pitch, avg_spectral_centroid, avg_zero_crossings, chroma_mean = features
    diagnosis = "Stammering" if avg_zero_crossings > 0.07 else "Normal"

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        insert_query = """
        INSERT INTO user_audio_records 
        (user_id, audio_file_path, tempo, average_pitch, spectral_centroid, zero_crossing_rate, chroma_mean, diagnosis)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (user_id, audio_file_path, tempo, avg_pitch, avg_spectral_centroid, avg_zero_crossings, chroma_mean, diagnosis)
        cursor.execute(insert_query, data)
        conn.commit()
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

    return diagnosis

def generate_graph(features, diagnosis, audio_file_path):
    labels = ['Tempo', 'Pitch', 'Spectral Centroid', 'Zero Crossing Rate', 'Chroma']
    features = [np.mean(f) if isinstance(f, (np.ndarray, list)) else f for f in features]

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.bar(labels, features, color='skyblue')
    ax.set_title(f"Speech Features Analysis ({diagnosis})")
    ax.set_ylabel("Feature Values")
    ax.set_xlabel("Speech Features")
    ax.yaxis.set_major_locator(MaxNLocator(integer=False))
    plt.tight_layout()

    graph_path = os.path.splitext(audio_file_path)[0] + "_graph.png"
    plt.savefig(graph_path)
    plt.close()
    
    return graph_path

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    file = request.files['audio']
    user_id = request.form.get('user_id', 1)

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    features = extract_speech_features(file_path)
    diagnosis = save_audio_record_with_features(user_id, file_path, features)
    graph_path = generate_graph(features, diagnosis, file_path)

    return jsonify({
        "message": "Audio file successfully uploaded and analyzed",
        "diagnosis": diagnosis,
        "graph_url": graph_path
    })

if __name__ == "__main__":
    app.run(debug=True)
