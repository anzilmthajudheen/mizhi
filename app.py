import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import logging
from datetime import datetime

# Import your detection system from the modified mizhi_detector.py
from mizhi_detector import SecuritySurveillanceSystem, logger # Import the logger too

# --- Flask App Setup ---
app = Flask(__name__, static_folder='../mizhi-frontend-html', template_folder='../mizhi-frontend-html')
app.config['SECRET_KEY'] = 'your_secret_key_here' # Change this to a strong, random key!
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet') # Use eventlet as async mode
CORS(app) # Enable CORS for all routes and origins

# --- Initialize MIZHI Detection System ---
# One instance of the system for all connections, or one per connection?
# For simplicity and to share model loading, one global instance for now.
# If performance is an issue or clients need isolated states,
# you might create an instance per client connection.
try:
    mizhi_system = SecuritySurveillanceSystem()
    logger.info("MIZHI SecuritySurveillanceSystem initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize MIZHI system: {e}", exc_info=True) # <--- UPDATED LINE
    mizhi_system = None


# --- Flask Routes ---

# Serve the main HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Serve other static files (CSS, JS)
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# Route for handling video file uploads (future implementation, placeholder)
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"message": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    if file:
        # Save the file temporarily or process it directly
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)
        logger.info(f"Received video file: {filepath}")
        # Here you would typically start a separate process or queue the video
        # for offline processing using mizhi_detector functions.
        # For this guide, we'll focus on real-time webcam streams.
        return jsonify({"message": "Video uploaded successfully", "filename": file.filename}), 200
    return jsonify({"message": "File upload failed"}), 500


# --- SocketIO Event Handlers ---

# When a client connects
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected to SocketIO')
    emit('server_status', {'status': 'Connected to MIZHI backend.'})


# When a client disconnects
@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected from SocketIO')

# Handler for incoming video frames from the frontend
@socketio.on('video_frame')
def handle_video_frame(data):
    if not mizhi_system:
        logger.error("MIZHI system not initialized. Cannot process frame.")
        emit('processing_error', {'message': 'MIZHI system not ready.'})
        return

    try:
        # Decode base64 image data
        # data['image'] format: "data:image/jpeg;base64,..."
        header, base64_image = data['image'].split(',', 1)
        image_bytes = base64.b64decode(base64_image)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # Decode into OpenCV BGR format

        if frame is None:
            logger.warning("Failed to decode frame from client.")
            return

        # Process the frame using the MIZHI system
        processed_frame_bytes, alerts_triggered, threats_detected = mizhi_system.process_frame_for_web(frame)

        if processed_frame_bytes:
            # Encode processed frame back to base64
            encoded_processed_frame = base64.b64encode(processed_frame_bytes).decode('utf-8')
            # Emit processed frame and alerts back to the client
            emit('processed_frame', {
                'image': f'data:image/jpeg;base64,{encoded_processed_frame}',
                'alerts': alerts_triggered,
                'threat_detected': threats_detected
            })
        else:
            logger.error("Failed to get processed frame bytes.")

    except Exception as e:
        logger.error(f"Error processing video frame: {e}", exc_info=True)
        emit('processing_error', {'message': f'Backend processing error: {str(e)}'})

# --- Main execution ---
if __name__ == '__main__':
    # Adjust port as needed, e.g., for cloud deployment
    # Use '0.0.0.0' to make it accessible from outside localhost
    logger.info("Starting Flask-SocketIO server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
