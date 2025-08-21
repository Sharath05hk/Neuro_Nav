import React, { useState } from "react";
import "../styles/LandingPage.css";

function LandingPage() {
  const [cameraOn, setCameraOn] = useState(false);

  const toggleCamera = async () => {
    if (!cameraOn) {
      // Start camera in Flask
      await fetch("http://127.0.0.1:5000/start_camera");
      setCameraOn(true);
    } else {
      // Stop camera in Flask
      await fetch("http://127.0.0.1:5000/stop_camera");
      setCameraOn(false);
    }
  };

  return (
    <div className="landing-container">
      <h1 className="title">Neuro-Nav AI&ML Project </h1>

      <div className="camera-box">
        {cameraOn ? (
          <img
            src="http://127.0.0.1:5000/video_feed"
            alt="Webcam Feed"
            className="camera-feed"
          />
        ) : (
          <p className="placeholder">📷 Your webcam feed will appear here...</p>
        )}
      </div>

      <div className="button-row">
        <button onClick={toggleCamera} className="start-btn">
          {cameraOn ? "Stop Camera" : "Start Camera"}
        </button>
      </div>
    </div>
  );
}

export default LandingPage;
