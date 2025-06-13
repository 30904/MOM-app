import React, { useState, useEffect } from 'react';
import { FaMicrophone } from 'react-icons/fa';

const RecordingInterface = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [timer, setTimer] = useState(0);
  const [meetingTitle, setMeetingTitle] = useState('');
  const [participants, setParticipants] = useState('');

  useEffect(() => {
    let interval;
    if (isRecording) {
      interval = setInterval(() => {
        setTimer((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg animate-componentFadeIn">
      {/* Microphone and Waveform */}
      <div className="flex flex-col items-center mb-4">
        <button
          onClick={() => setIsRecording(!isRecording)}
          className={`p-4 rounded-full ${isRecording ? 'bg-red-500' : 'bg-blue-900'} text-white animate-pulse`}
        >
          <FaMicrophone className="h-8 w-8" />
        </button>
        {isRecording && (
          <div className="mt-2 w-1/2 h-6 bg-gray-200 rounded overflow-hidden">
            <div className="h-full bg-blue-600 animate-waveform"></div>
          </div>
        )}
        <span className="mt-2 text-gray-700 dark:text-gray-300">{formatTime(timer)}</span>
      </div>

      {/* Meeting Info */}
      <div className="space-y-4">
        <input
          type="text"
          placeholder="Meeting Title"
          value={meetingTitle}
          onChange={(e) => setMeetingTitle(e.target.value)}
          className="w-full p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white"
        />
        <input
          type="text"
          placeholder="Participants (comma-separated)"
          value={participants}
          onChange={(e) => setParticipants(e.target.value)}
          className="w-full p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white"
        />
      </div>
    </div>
  );
};

export default RecordingInterface;