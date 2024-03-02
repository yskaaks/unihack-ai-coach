import React from 'react';
import { useRef, useEffect, useState } from 'react';
import './App.css';
import WebcamStream from './WebCamStream';

function App() {
  return (
    <div className="App">
     <h1>
      GymBuddy
     </h1>

     <h2 className='subheader'>
        Insert some sorta subtitle LOL
     </h2>

      <WebcamStream />
  
    </div>
  );
}

export default App;
