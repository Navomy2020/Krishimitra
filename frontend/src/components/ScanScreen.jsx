import { useRef, useState } from "react";
import Icon from "./Icon.jsx";

export default function ScanScreen({ onFile, error }) {
  const cameraRef = useRef(null);
  const galleryRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleInput = (e) => {
    const file = e.target.files?.[0];
    e.target.value = ""; // so choosing the same photo again still fires
    if (file) onFile(file);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) onFile(file);
  };

  return (
    <section className="screen" aria-labelledby="scan-title">
      <h1 id="scan-title" tabIndex={-1}>Check your crop</h1>
      <p className="lede">Photograph one leaf and get a likely diagnosis with the next steps.</p>

      <div
        className={`viewfinder${dragging ? " is-dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <span className="corner corner-tl" />
        <span className="corner corner-tr" />
        <span className="corner corner-bl" />
        <span className="corner corner-br" />
        <svg className="viewfinder-leaf" viewBox="0 0 120 120" aria-hidden="true">
          <path d="M18 98C18 52 46 20 106 14c-3 56-30 90-76 90-4 0-8-.4-12-1.4z" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
          <path d="M22 96C44 66 68 44 96 26" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" />
        </svg>
        <p className="viewfinder-hint">{dragging ? "Drop the photo to check it" : "Fill the frame with one leaf"}</p>
      </div>

      {error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}

      <div className="actions">
        <button type="button" className="btn btn-primary" onClick={() => cameraRef.current?.click()}>
          <Icon name="camera" /> Take a photo
        </button>
        <button type="button" className="btn btn-quiet" onClick={() => galleryRef.current?.click()}>
          <Icon name="image" /> Choose from gallery
        </button>
      </div>
      <input ref={cameraRef} type="file" accept="image/*" capture="environment" hidden onChange={handleInput} />
      <input ref={galleryRef} type="file" accept="image/*" hidden onChange={handleInput} />

      <ul className="tips">
        <li>Use daylight and turn the flash off.</li>
        <li>Keep the spotted or yellowed part in focus.</li>
        <li>Photograph one leaf at a time.</li>
      </ul>
    </section>
  );
}
