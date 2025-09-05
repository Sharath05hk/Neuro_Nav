import React, { useState } from "react";
import "../styles/LandingPage.css";

function LandingPage() {
  const [cameraOn, setCameraOn] = useState(false);

  // Start project
  const startProject = async () => {
    await fetch("http://127.0.0.1:5000/start_camera");
    setCameraOn(true);
  };

  // Stop project
  const stopProject = async () => {
    await fetch("http://127.0.0.1:5000/stop_camera");
    setCameraOn(false);
  };

  return (
    <div className="landing-container">
      <h1 className="title">Neuro-Nav AI & ML Project</h1>

      <div className="main-content">
        {/* Camera box (left side now) */}
        <div className="camera-box">
          {cameraOn ? (
            <img
              src="http://127.0.0.1:5000/video_feed"
              alt="Webcam Feed"
              className="camera-feed"
            />
          ) : (
            <p className="placeholder">
              📷 Your webcam feed will appear here...
            </p>
          )}
        </div>

        {/* Buttons column (right side) */}
        <div className="button-column">
          <button
            onClick={() => fetch("http://127.0.0.1:5000/open_pictures")}
            className="btn"
          >
            Open Pictures
          </button>
          <button onClick={startProject} className="btn start-btn">
            Start Project
          </button>
          <button onClick={stopProject} className="btn stop-btn">
            Stop Project
          </button>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;
