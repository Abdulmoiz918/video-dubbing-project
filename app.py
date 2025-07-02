from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
import subprocess
from moviepy.editor import VideoFileClip, AudioFileClip,CompositeAudioClip
from gtts import gTTS
import speech_recognition as sr
from googletrans import Translator
from moviepy.audio.fx.all import audio_normalize
import webbrowser
from threading import Timer
import tempfile
from pydub import AudioSegment
from pydub.effects import speedup
from pydub.silence import split_on_silence, detect_silence
import time
import threading
import math
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import webrtcvad
import wave
import contextlib

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Language mapping with proper gTTS language codes
LANGUAGE_MAP = {
    'hi': 'hi',  # Hindi
    'ta': 'ta',  # Tamil
    'te': 'te',  # Telugu
    'kn': 'kn',  # Kannada
    'ml': 'ml',  # Malayalam
    'bn': 'bn',  # Bengali
    'gu': 'gu',  # Gujarati
    'mr': 'mr',  # Marathi
    'pa': 'pa',  # Punjabi
    'ur': 'ur'   # Urdu
}

processing_progress = {}
processing_lock = threading.Lock()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AudioSyncEngine:
    """Advanced audio synchronization engine with multiple techniques"""
    
    def __init__(self):
        self.vad = webrtcvad.Vad(2)  # Voice Activity Detection
        
    def detect_speech_segments(self, audio_path, frame_duration_ms=30):
        """Detect speech segments using Voice Activity Detection"""
        try:
            # Load audio and convert to 16kHz mono for VAD
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_int16.tobytes()
            
            segments = []
            frame_size = int(16000 * frame_duration_ms / 1000)
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size].tobytes()
                if len(frame) == frame_size * 2:  # 16-bit = 2 bytes per sample
                    is_speech = self.vad.is_speech(frame, 16000)
                    timestamp = i / 16000
                    segments.append({
                        'timestamp': timestamp,
                        'is_speech': is_speech,
                        'duration': frame_duration_ms / 1000
                    })
            
            return segments
        except Exception as e:
            print(f"VAD Error: {e}")
            return self._fallback_silence_detection(audio_path)
    
    def _fallback_silence_detection(self, audio_path):
        """Fallback silence detection using pydub"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            silence_ranges = detect_silence(
                audio, 
                min_silence_len=500,  # 500ms minimum silence
                silence_thresh=audio.dBFS - 16  # 16dB below average
            )
            
            segments = []
            last_end = 0
            
            for start, end in silence_ranges:
                # Add speech segment before silence
                if start > last_end:
                    segments.append({
                        'timestamp': last_end / 1000,
                        'is_speech': True,
                        'duration': (start - last_end) / 1000
                    })
                
                # Add silence segment
                segments.append({
                    'timestamp': start / 1000,
                    'is_speech': False,
                    'duration': (end - start) / 1000
                })
                
                last_end = end
            
            # Add final speech segment if exists
            if last_end < len(audio):
                segments.append({
                    'timestamp': last_end / 1000,
                    'is_speech': True,
                    'duration': (len(audio) - last_end) / 1000
                })
            
            return segments
        except Exception as e:
            print(f"Silence detection error: {e}")
            return []
    
    def analyze_prosody(self, audio_path):
        """Analyze prosodic features (pitch, energy, rhythm)"""
        try:
            y, sr = librosa.load(audio_path)
            
            # Extract features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_track = []
            
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                pitch_track.append(pitch if pitch > 0 else 0)
            
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            return {
                'pitch_track': pitch_track,
                'rms_energy': rms,
                'tempo': tempo,
                'beat_frames': beats,
                'duration': len(y) / sr
            }
        except Exception as e:
            print(f"Prosody analysis error: {e}")
            return None
    
    def time_stretch_with_prosody(self, audio_path, target_duration, prosody_data=None):
        """Advanced time stretching preserving prosodic features"""
        try:
            y, sr = librosa.load(audio_path)
            current_duration = len(y) / sr
            stretch_ratio = target_duration / current_duration
            
            if 0.5 <= stretch_ratio <= 2.0:
                # Use phase vocoder for quality time stretching
                y_stretched = librosa.effects.time_stretch(y, rate=1/stretch_ratio)
            else:
                # For extreme ratios, use WSOLA (Waveform Similarity Overlap-Add)
                y_stretched = self._wsola_stretch(y, stretch_ratio, sr)
            
            # Ensure exact duration
            target_samples = int(target_duration * sr)
            if len(y_stretched) > target_samples:
                y_stretched = y_stretched[:target_samples]
            elif len(y_stretched) < target_samples:
                # Pad with silence
                padding = target_samples - len(y_stretched)
                y_stretched = np.pad(y_stretched, (0, padding), mode='constant')
            
            # Save stretched audio
            output_path = audio_path.replace('.wav', '_stretched.wav')
            sf.write(output_path, y_stretched, sr)
            
            return output_path
        except Exception as e:
            print(f"Time stretch error: {e}")
            return audio_path
    
    def _wsola_stretch(self, y, stretch_ratio, sr, frame_length=2048, hop_length=512):
        """WSOLA algorithm for high-quality time stretching"""
        try:
            if stretch_ratio == 1.0:
                return y
            
            # Calculate new hop length
            synthesis_hop = int(hop_length * stretch_ratio)
            
            # Initialize output
            output_length = int(len(y) * stretch_ratio)
            output = np.zeros(output_length)
            
            # WSOLA parameters
            tolerance = hop_length // 2
            
            output_pos = 0
            input_pos = 0
            
            while input_pos + frame_length < len(y) and output_pos + frame_length < len(output):
                # Current frame
                current_frame = y[input_pos:input_pos + frame_length]
                
                if output_pos > 0:
                    # Find best match in tolerance region
                    search_start = max(0, input_pos - tolerance)
                    search_end = min(len(y) - frame_length, input_pos + tolerance)
                    
                    best_pos = input_pos
                    best_correlation = -1
                    
                    overlap_size = min(frame_length // 4, output_pos)
                    if overlap_size > 0:
                        target_overlap = output[output_pos - overlap_size:output_pos]
                        
                        for pos in range(search_start, search_end):
                            candidate_overlap = y[pos:pos + overlap_size]
                            correlation = np.corrcoef(target_overlap, candidate_overlap)[0, 1]
                            if not np.isnan(correlation) and correlation > best_correlation:
                                best_correlation = correlation
                                best_pos = pos
                    
                    current_frame = y[best_pos:best_pos + frame_length]
                    input_pos = best_pos
                
                # Overlap-add
                if output_pos > 0:
                    overlap_size = min(frame_length // 4, output_pos, len(output) - output_pos)
                    if overlap_size > 0:
                        # Linear crossfade
                        fade_out = np.linspace(1, 0, overlap_size)
                        fade_in = np.linspace(0, 1, overlap_size)
                        
                        output[output_pos - overlap_size:output_pos] *= fade_out
                        output[output_pos - overlap_size:output_pos] += current_frame[:overlap_size] * fade_in
                        
                        # Add remaining part
                        remaining = min(frame_length - overlap_size, len(output) - output_pos)
                        output[output_pos:output_pos + remaining] = current_frame[overlap_size:overlap_size + remaining]
                else:
                    # First frame
                    output[:frame_length] = current_frame
                
                # Update positions
                output_pos += synthesis_hop
                input_pos += hop_length
            
            return output
        except Exception as e:
            print(f"WSOLA error: {e}")
            return y
    
    def create_synchronized_audio(self, original_audio_path, tts_audio_path, target_duration):
        """Create perfectly synchronized audio using multiple techniques"""
        try:
            # Analyze original audio structure
            speech_segments = self.detect_speech_segments(original_audio_path)
            prosody_data = self.analyze_prosody(original_audio_path)
            
            # Load TTS audio
            tts_audio = AudioSegment.from_wav(tts_audio_path)
            tts_duration = len(tts_audio) / 1000.0
            
            # Calculate speech and silence ratios
            total_speech_time = sum(seg['duration'] for seg in speech_segments if seg['is_speech'])
            total_silence_time = target_duration - total_speech_time
            
            if total_speech_time > 0:
                # Time-stretch TTS to match speech duration
                speech_stretch_ratio = total_speech_time / tts_duration
                stretched_tts_path = self.time_stretch_with_prosody(tts_audio_path, total_speech_time, prosody_data)
                stretched_tts = AudioSegment.from_wav(stretched_tts_path)
                
                # Create synchronized timeline
                synchronized_audio = AudioSegment.empty()
                tts_position = 0
                
                for segment in speech_segments:
                    segment_duration_ms = int(segment['duration'] * 1000)
                    
                    if segment['is_speech']:
                        # Add speech segment from stretched TTS
                        tts_segment = stretched_tts[tts_position:tts_position + segment_duration_ms]
                        
                        # Ensure segment is exact duration
                        if len(tts_segment) < segment_duration_ms:
                            # Pad with silence
                            tts_segment += AudioSegment.silent(segment_duration_ms - len(tts_segment))
                        elif len(tts_segment) > segment_duration_ms:
                            # Trim
                            tts_segment = tts_segment[:segment_duration_ms]
                        
                        synchronized_audio += tts_segment
                        tts_position += len(tts_segment)
                    else:
                        # Add silence segment
                        synchronized_audio += AudioSegment.silent(segment_duration_ms)
                
                # Ensure exact target duration
                target_duration_ms = int(target_duration * 1000)
                if len(synchronized_audio) > target_duration_ms:
                    synchronized_audio = synchronized_audio[:target_duration_ms]
                elif len(synchronized_audio) < target_duration_ms:
                    synchronized_audio += AudioSegment.silent(target_duration_ms - len(synchronized_audio))
                
                # Apply dynamic range compression for better audibility
                synchronized_audio = self._apply_audio_processing(synchronized_audio)
                
                # Save synchronized audio
                output_path = tts_audio_path.replace('.wav', '_synchronized.wav')
                synchronized_audio.export(output_path, format="wav")
                
                # Cleanup
                if os.path.exists(stretched_tts_path):
                    os.remove(stretched_tts_path)
                
                return output_path
            else:
                # Fallback to simple time stretching
                return self.time_stretch_with_prosody(tts_audio_path, target_duration)
                
        except Exception as e:
            print(f"Synchronization error: {e}")
            # Fallback to simple duration matching
            return self._simple_duration_match(tts_audio_path, target_duration)
    
    def _apply_audio_processing(self, audio_segment):
        """Apply audio processing for better quality"""
        try:
            # Normalize audio
            normalized = audio_segment.normalize()
            
            # Apply gentle compression
            compressed = normalized.compress_dynamic_range(
                threshold=-20.0,
                ratio=4.0,
                attack=5.0,
                release=50.0
            )
            
            return compressed
        except Exception as e:
            print(f"Audio processing error: {e}")
            return audio_segment
    
    def _simple_duration_match(self, audio_path, target_duration):
        """Simple fallback duration matching"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            current_duration = len(audio) / 1000.0
            
            if current_duration == 0:
                return audio_path
            
            ratio = target_duration / current_duration
            
            if 0.5 <= ratio <= 2.0:
                # Use pydub's speedup for moderate changes
                if ratio > 1:
                    # Slow down (stretch)
                    adjusted = audio._spawn(audio.raw_data, overrides={
                        "frame_rate": int(audio.frame_rate / ratio)
                    }).set_frame_rate(audio.frame_rate)
                else:
                    # Speed up
                    adjusted = speedup(audio, playback_speed=1/ratio)
            else:
                # For extreme ratios, just truncate or loop
                target_ms = int(target_duration * 1000)
                if ratio < 0.5:
                    adjusted = audio[:target_ms]
                else:
                    loops = math.ceil(ratio)
                    adjusted = audio
                    for _ in range(loops - 1):
                        adjusted += audio
                    adjusted = adjusted[:target_ms]
            
            output_path = audio_path.replace('.wav', '_matched.wav')
            adjusted.export(output_path, format="wav")
            return output_path
            
        except Exception as e:
            print(f"Simple duration match error: {e}")
            return audio_path

# Initialize the sync engine
sync_engine = AudioSyncEngine()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        original_ext = filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if original_ext == 'mov':
            mp4_filename = filename.rsplit('.', 1)[0] + '.mp4'
            mp4_filepath = os.path.join(app.config['UPLOAD_FOLDER'], mp4_filename)

            try:
                subprocess.run([
                    'ffmpeg', '-i', filepath,
                    '-vcodec', 'libx264', '-acodec', 'aac', mp4_filepath
                ], check=True)
                os.remove(filepath)
                filepath = mp4_filepath
                result = {
                    "filepath": filepath,
                    "status": "Uploaded and converted from .mov to .mp4"
                }
            except subprocess.CalledProcessError as e:
                return jsonify({"error": f"FFmpeg conversion failed: {e}"}), 500
        else:
            result = {"filepath": filepath, "status": "Uploaded"}

        return render_template('upload_result.html', result=result)
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/progress/<video_id>')
def get_progress(video_id):
    with processing_lock:
        progress = processing_progress.get(video_id, {'progress': 0, 'status': 'Not started'})
    return jsonify(progress)

@app.route('/download', methods=['POST'])
def download_video():
    try:
        filepath = request.form.get('filepath')
        language = request.form.get('language')
        video_id = request.form.get('video_id')

        if not filepath or not language or not video_id:
            return jsonify({"error": "Missing file, language, or video_id"}), 400

        with processing_lock:
            processing_progress[video_id] = {'progress': 0, 'status': 'Starting...'}

        def process_in_background():
            try:
                result = process_video(filepath, language, video_id)
                with processing_lock:
                    if result['status'] != 'Processed':
                        processing_progress[video_id] = {'progress': 100, 'status': f'Error: {result["status"]}'}
                    else:
                        processing_progress[video_id] = {'progress': 100, 'status': 'Completed'}
            except Exception as e:
                with processing_lock:
                    processing_progress[video_id] = {'progress': 100, 'status': f'Error: {str(e)}'}

        processing_thread = threading.Thread(target=process_in_background)
        processing_thread.daemon = True
        processing_thread.start()

        while True:
            with processing_lock:
                current_progress = processing_progress.get(video_id, {'progress': 0, 'status': 'Processing...'})
            
            if current_progress['progress'] >= 100:
                break
            time.sleep(0.5)

        if 'Error' in current_progress['status']:
            return jsonify({"error": current_progress['status']}), 500

        result = process_video(filepath, language, video_id, return_path_only=True)
        
        if not result or 'output_video' not in result:
            return jsonify({"error": "Failed to get output video path"}), 500

        output_video = result['output_video']

        if not os.path.exists(output_video):
            return jsonify({"error": "Processed video file not found"}), 500

        return send_file(output_video, as_attachment=True)
        
    except Exception as e:
        print(f"Error in /download: {str(e)}")
        if 'video_id' in locals():
            with processing_lock:
                processing_progress[video_id] = {'progress': 100, 'status': f'Error: {str(e)}'}
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

def update_progress(video_id, progress, status):
    with processing_lock:
        processing_progress[video_id] = {'progress': progress, 'status': status}

def generate_tts_audio(text, language):
    """Enhanced TTS generation with better quality settings"""
    temp_audio_path = tempfile.mktemp(suffix='.wav')
    
    try:
        lang_code = LANGUAGE_MAP.get(language, language)
        max_chunk_length = 4000  # Smaller chunks for better processing
        
        # Split text more intelligently at sentence boundaries
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        combined_audio = None
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            tts = gTTS(text=chunk, lang=lang_code, slow=False)
            temp_mp3_path = tempfile.mktemp(suffix='.mp3')
            tts.save(temp_mp3_path)
            
            chunk_audio = AudioSegment.from_mp3(temp_mp3_path)
            
            # Add small pause between sentences
            if combined_audio is None:
                combined_audio = chunk_audio
            else:
                pause = AudioSegment.silent(duration=200)  # 200ms pause
                combined_audio = combined_audio + pause + chunk_audio
                
            os.remove(temp_mp3_path)
            
        if combined_audio:
            # Enhance audio quality
            enhanced_audio = combined_audio.set_frame_rate(22050).set_channels(1)
            enhanced_audio.export(temp_audio_path, format="wav")
            return temp_audio_path
        else:
            raise Exception("Failed to generate TTS audio")
            
    except Exception as e:
        print(f"TTS Generation Error: {e}")
        raise e

def process_video(filepath, target_language, video_id, return_path_only=False):
    try:
        update_progress(video_id, 5, "Initializing advanced video processing...")
        
        download_folder = 'downloads/'
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        base_name = os.path.basename(filepath).replace('.mp4', '')
        output_filepath = os.path.join(
            download_folder, 
            f"{base_name}_dubbed_{target_language}.mp4"
        )

        if return_path_only:
            if os.path.exists(output_filepath):
                return {
                    "filepath": filepath,
                    "status": "Processed",
                    "output_video": output_filepath
                }
            else:
                return None
        
        video = VideoFileClip(filepath)
        if video.audio is None:
            return {
                "filepath": filepath,
                "status": "Error: Video has no audio component"
            }

        video_duration = video.duration
        update_progress(video_id, 15, "Extracting and analyzing audio...")

        # Extract audio with higher quality
        audio_filepath = filepath.replace('.mp4', '_audio.wav')
        video.audio.write_audiofile(
            audio_filepath, 
            codec='pcm_s16le', 
            verbose=False, 
            logger=None,
            bitrate="192k"
        )

        update_progress(video_id, 30, "Advanced speech recognition...")

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        
        with sr.AudioFile(audio_filepath) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio_data = recognizer.record(source)
            transcript = recognizer.recognize_google(audio_data, language='en-IN')
        
        update_progress(video_id, 50, "Translating with context awareness...")

        translated_text = translate_text(transcript, target_language)
        update_progress(video_id, 65, "Generating high-quality dubbed audio...")

        tts_filepath = generate_tts_audio(translated_text, target_language)
        
        update_progress(video_id, 80, "Performing advanced audio synchronization...")
        
        # Use the enhanced synchronization engine
        synchronized_audio_path = sync_engine.create_synchronized_audio(
            audio_filepath, tts_filepath, video_duration
        )
        
        update_progress(video_id, 90, "Creating final dubbed video...")
        
        try:
            new_audio = AudioFileClip(synchronized_audio_path).fx(audio_normalize)
            
            # Ensure perfect duration match
            if abs(new_audio.duration - video_duration) > 0.01:  # 10ms tolerance
                new_audio = new_audio.subclip(0, video_duration)

            update_progress(video_id, 95, "Finalizing high-quality video...")

            # Create dubbed video with better quality settings
            dubbed_video = video.set_audio(new_audio)
            dubbed_video.write_videofile(
                output_filepath, 
                codec='libx264', 
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None,
                bitrate="5000k",  # Higher video bitrate
                audio_bitrate="192k"  # Higher audio bitrate
            )

            # Cleanup temporary files
            cleanup_files = [audio_filepath, tts_filepath, synchronized_audio_path]
            for file_path in cleanup_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            
            video.close()
            new_audio.close()
            dubbed_video.close()

            update_progress(video_id, 100, "Advanced video processing completed!")

            return {
                "filepath": filepath,
                "status": "Processed",
                "output_video": output_filepath
            }

        except Exception as e:
            print(f"Error during video processing: {e}")
            return {
                "filepath": filepath,
                "status": f"Error: Could not process video - {e}"
            }

    except Exception as e:
        print(f"General processing error: {e}")
        update_progress(video_id, 100, f"Error: {str(e)}")
        return {
            "filepath": filepath,
            "status": f"Error: {str(e)}"
        }

def translate_text(text, target_language):
    try:
        translator = Translator()
        # Enhanced translation with context preservation
        max_chunk_size = 4000
        
        # Split on sentence boundaries to preserve context
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        translated_chunks = []
        for chunk in chunks:
            if chunk.strip():
                translated = translator.translate(chunk, dest=target_language)
                translated_chunks.append(translated.text)
            
        return ' '.join(translated_chunks)
    except Exception as e:
        print(f"Translation error: {e}")
        return f"Translation Error: {str(e)}"

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True)