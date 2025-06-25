// --- DOM Element References ---
const loginPage = document.getElementById('login-page');
const dashboardPage = document.getElementById('dashboard-page');
const loginForm = document.getElementById('login-form');
const usernameInput = document.getElementById('username');
const passwordInput = document.getElementById('password');
const loginMessage = document.getElementById('login-message');
const logoutButton = document.getElementById('logout-button');

const videoFeed = document.getElementById('video-feed');
const videoPlaceholder = document.getElementById('video-placeholder');
const startWebcamBtn = document.getElementById('start-webcam-btn');
const uploadVideoBtn = document.getElementById('upload-video-btn');
const videoFileInput = document.getElementById('video-file-input');
const stopSurveillanceBtn = document.getElementById('stop-surveillance-btn');
const simulateAlertBtn = document.getElementById('simulate-alert-btn'); // Still useful for quick UI test without backend
const systemStatusDisplay = document.getElementById('system-status-display');
const alertLog = document.getElementById('alert-log');

// --- Global State Variables ---
let isLoggedIn = false;
let currentStream = null; // To hold the webcam stream
let currentVideoSrc = null; // To hold the object URL for uploaded video (local playback)
let systemStatus = 'Idle';
let alerts = [];

let socket = null; // Socket.IO connection
let frameInterval = null; // Interval for sending frames
const FPS = 15; // Frames per second to send to backend
const FRAME_SEND_INTERVAL = 1000 / FPS;

// --- Utility Functions ---

function updateSystemStatusDisplay() {
    systemStatusDisplay.textContent = systemStatus;
    systemStatusDisplay.className = 'font-bold '; // Reset class
    if (systemStatus === 'Idle') {
        systemStatusDisplay.classList.add('text-gray-400');
        disableButtons(['stop-surveillance-btn']);
        enableButtons(['start-webcam-btn', 'upload-video-btn']);
    } else if (systemStatus === 'Error' || systemStatus === 'Disconnected') {
        systemStatusDisplay.classList.add('text-red-500');
        enableButtons(['start-webcam-btn', 'upload-video-btn', 'stop-surveillance-btn']); // Allow retrying or stopping
    } else if (systemStatus === 'Webcam Active' || systemStatus === 'Video File Loaded' || systemStatus === 'Processing...') {
        systemStatusDisplay.classList.add('text-green-400');
        enableButtons(['stop-surveillance-btn']);
        disableButtons(['start-webcam-btn', 'upload-video-btn']);
    }
}

function enableButtons(buttonIds) {
    buttonIds.forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = false;
    });
}

function disableButtons(buttonIds) {
    buttonIds.forEach(id => {
        const btn = document.getElementById(id);
        if (btn) btn.disabled = true;
    });
}

function addAlert(type, message) {
    const timestamp = new Date().toLocaleTimeString();
    alerts.push({ type, message, timestamp });

    if (alerts.length > 50) {
        alerts = alerts.slice(alerts.length - 50);
    }
    renderAlerts();
}

function renderAlerts() {
    alertLog.innerHTML = '';
    if (alerts.length === 0) {
        alertLog.innerHTML = '<p class="text-gray-400">No alerts or logs yet. System is idle.</p>';
        return;
    }

    const ul = document.createElement('ul');
    alerts.forEach(alert => {
        const li = document.createElement('li');
        let typeClass = 'text-gray-300';
        let typePrefix = '';

        if (alert.type === 'error') {
            typeClass = 'text-red-400';
        } else if (alert.type === 'warning') {
            typeClass = 'text-orange-400';
            typePrefix = '⚠️ WARNING: ';
        } else if (alert.type === 'critical') {
            typeClass = 'text-red-500 font-bold';
            typePrefix = '🚨 CRITICAL: ';
        }

        li.className = `mb-1 ${typeClass}`;
        li.innerHTML = `<span class="font-mono text-xs text-gray-400 mr-2">[${alert.timestamp}]</span>${typePrefix}${alert.message}`;
        ul.appendChild(li);
    });
    alertLog.appendChild(ul);
    alertLog.scrollTop = alertLog.scrollHeight;
}


// --- Socket.IO and Video Processing Functions ---

function initSocketIO() {
    // IMPORTANT: Replace with your backend server's IP or domain if not running on localhost
    socket = io('http://localhost:5000'); // Connect to your Flask-SocketIO backend

    socket.on('connect', () => {
        console.log('Connected to Socket.IO server');
        addAlert('info', 'Connected to backend server.');
        systemStatus = 'Connected';
        updateSystemStatusDisplay();
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from Socket.IO server');
        addAlert('error', 'Disconnected from backend server.');
        systemStatus = 'Disconnected';
        updateSystemStatusDisplay();
        clearInterval(frameInterval); // Stop sending frames if disconnected
    });

    socket.on('processed_frame', (data) => {
        // Display the processed frame received from the backend
        videoFeed.src = data.image;
        videoFeed.srcObject = null; // Clear srcObject as we're now using src URL
        videoFeed.controls = false; // Hide controls for live processed stream
        videoPlaceholder.classList.add('hidden'); // Hide placeholder
        systemStatus = data.threat_detected ? 'Threat Detected' : 'Processing...'; // Update status based on backend
        updateSystemStatusDisplay();

        // Add any alerts received from the backend
        if (data.alerts && data.alerts.length > 0) {
            data.alerts.forEach(alert => {
                addAlert(alert.type, `${alert.threat_level}: ${alert.message} (Confidence: ${alert.confidence})`);
            });
        }
    });

    socket.on('processing_error', (data) => {
        addAlert('error', `Backend processing error: ${data.message}`);
        systemStatus = 'Error';
        updateSystemStatusDisplay();
    });

    socket.on('server_status', (data) => {
        addAlert('info', `Server: ${data.status}`);
    });
}

// Function to send video frames to the backend
function sendVideoFrame() {
    if (videoFeed.paused || videoFeed.ended) {
        // If the local video playback is paused or ended, stop sending frames
        clearInterval(frameInterval);
        frameInterval = null;
        systemStatus = 'Idle';
        updateSystemStatusDisplay();
        return;
    }

    if (!socket || !socket.connected) {
        console.warn('Socket not connected, cannot send frame.');
        return;
    }

    // Create a canvas element to draw the current video frame
    const canvas = document.createElement('canvas');
    canvas.width = videoFeed.videoWidth;
    canvas.height = videoFeed.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoFeed, 0, 0, canvas.width, canvas.height);

    // Get the image data as a JPEG base64 string
    const imageData = canvas.toDataURL('image/jpeg', 0.8); // 0.8 quality for balance

    // Emit the frame data to the backend via Socket.IO
    socket.emit('video_frame', { image: imageData });
}

// Function to start sending frames (for webcam or uploaded video)
function startFrameSending() {
    if (frameInterval) {
        clearInterval(frameInterval);
    }
    frameInterval = setInterval(sendVideoFrame, FRAME_SEND_INTERVAL);
    systemStatus = 'Processing...';
    updateSystemStatusDisplay();
}

// --- View Management Functions ---

function showLoginPage() {
    loginPage.classList.remove('hidden');
    dashboardPage.classList.add('hidden');
    loginMessage.textContent = ''; // Clear login messages
    usernameInput.value = '';
    passwordInput.value = '';
    isLoggedIn = false;
    stopSurveillance(); // Ensure surveillance is stopped when logging out
    if (socket) socket.disconnect(); // Disconnect socket on logout
    socket = null;
    alerts = []; // Clear alerts
    renderAlerts(); // Re-render empty alerts
    systemStatus = 'Idle';
    updateSystemStatusDisplay();
}

function showDashboardPage() {
    loginPage.classList.add('hidden');
    dashboardPage.classList.remove('hidden');
    isLoggedIn = true;
    addAlert('info', 'Logged in successfully.');
    initSocketIO(); // Initialize Socket.IO connection on dashboard entry
    systemStatus = 'Idle'; // Initial status on dashboard
    renderAlerts(); // Ensure alerts are rendered
    updateSystemStatusDisplay(); // Update button states
}

// --- Event Handlers ---

// Login Form Submission
loginForm.addEventListener('submit', (e) => {
    e.preventDefault(); // Prevent default form submission

    const username = usernameInput.value;
    const password = passwordInput.value;

    // Mock authentication: In a real app, this would involve API calls to a backend
    if (username === 'keralapolice' && password === 'mizhi@2024') {
        showDashboardPage();
    } else {
        loginMessage.textContent = 'Invalid username or password.';
        addAlert('error', 'Login failed: Invalid credentials.');
    }
});

// Logout Button Click
logoutButton.addEventListener('click', () => {
    showLoginPage();
});

// Start Webcam Button Click
startWebcamBtn.addEventListener('click', async () => {
    try {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        if (currentVideoSrc) {
            URL.revokeObjectURL(currentVideoSrc);
            currentVideoSrc = null;
        }

        const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        currentStream = mediaStream;
        videoFeed.srcObject = mediaStream;
        videoFeed.play();
        videoFeed.controls = false;
        videoPlaceholder.classList.add('hidden');
        systemStatus = 'Webcam Active';
        addAlert('info', 'Webcam stream started.');
        startFrameSending(); // Start sending frames to backend
    } catch (error) {
        console.error('Error accessing webcam:', error);
        systemStatus = 'Error';
        addAlert('error', `Failed to start webcam: ${error.message}`);
    } finally {
        updateSystemStatusDisplay();
    }
});

// Upload Video Button Click (triggers hidden file input)
uploadVideoBtn.addEventListener('click', () => {
    videoFileInput.click();
});

// Handle Video File Input Change
videoFileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
        if (currentVideoSrc) {
            URL.revokeObjectURL(currentVideoSrc);
        }

        // For uploaded files, we will send them to the backend for processing
        // and then the backend can send back processed frames via WebSocket
        // Or, for simplicity in this example, we'll just play it locally
        // and send frames to backend as if it were a webcam.
        const url = URL.createObjectURL(file);
        currentVideoSrc = url;
        videoFeed.src = url;
        videoFeed.srcObject = null;
        videoFeed.play();
        videoFeed.controls = true;
        videoPlaceholder.classList.add('hidden');
        systemStatus = 'Video File Loaded';
        addAlert('info', `Video file "${file.name}" loaded.`);

        // --- OPTIONAL: Upload the full file to backend ---
        // You could also send the entire file to backend for server-side processing
        // const formData = new FormData();
        // formData.append('video', file);
        // try {
        //     const response = await fetch('http://localhost:5000/upload_video', {
        //         method: 'POST',
        //         body: formData,
        //     });
        //     const result = await response.json();
        //     addAlert('info', `File upload to backend: ${result.message}`);
        // } catch (error) {
        //     addAlert('error', `File upload failed: ${error.message}`);
        // }
        // --- END OPTIONAL ---

        startFrameSending(); // Start sending frames to backend for processing
    } else {
        systemStatus = 'Idle';
        addAlert('info', 'No video file selected.');
    }
    updateSystemStatusDisplay();
});

// Stop Surveillance Button Click
stopSurveillanceBtn.addEventListener('click', stopSurveillance);

function stopSurveillance() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
        videoFeed.srcObject = null;
    }
    if (currentVideoSrc) {
        URL.revokeObjectURL(currentVideoSrc);
        currentVideoSrc = null;
        videoFeed.src = '';
    }
    videoFeed.pause();
    videoFeed.load(); // Reload video to clear current playback
    videoPlaceholder.classList.remove('hidden'); // Show placeholder
    systemStatus = 'Idle';
    addAlert('info', 'Surveillance stopped.');
    clearInterval(frameInterval); // Stop sending frames
    frameInterval = null;
    updateSystemStatusDisplay();
}

// Simulate Alert Button Click (frontend-only simulation)
simulateAlertBtn.addEventListener('click', () => {
    const randomAlertType = Math.random() > 0.5 ? 'warning' : 'critical';
    const randomMessage = randomAlertType === 'warning'
        ? 'Suspicious movement detected near the entrance.'
        : 'Potential weapon detected! Review footage immediately.';
    addAlert(randomAlertType, randomMessage);
});


// --- Initial Setup on Page Load ---
document.addEventListener('DOMContentLoaded', () => {
    showLoginPage(); // Start by showing the login page
    // No need to initSocketIO here, it happens after successful login
});
