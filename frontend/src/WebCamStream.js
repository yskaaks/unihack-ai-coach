import React, { useEffect, useRef } from 'react';

function WebcamStream() {
  const videoRef = useRef(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((error) => {
        console.error('Error accessing the webcam:', error);
      });
  }, []);

  return (
    <div>
      <h1></h1>
      <video ref={videoRef} autoPlay playsInline style={{ width: '100%', maxWidth: '500px' }} />
    </div>
  );
}

export default WebcamStream;
