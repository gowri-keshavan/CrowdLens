// frontend/src/App.jsx
import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("video", file);
    setStatus("Uploading...");

    try {
      const res = await axios.post("http://localhost:5000/upload", formData);
      setStatus(res.data.status);
    } catch (err) {
      console.error(err);
      setStatus("Error uploading video");
    }
  };

  return (
    <div className="app">
      <h1 className="title"> CrowdLens </h1>
      <p>AI-BASED REAL TIME CROWD BEHAVIOUR ANALYSER
</p>
      <p className="subtitle">Upload your video for real-time crowd detection and tracking</p>

      <div className="upload-container">
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button onClick={handleUpload}>Upload & Analyze</button>
      </div>

      {status && <p className="status">{status}</p>}
    </div>
  );
}

export default App;
