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
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-2 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-700">Recording</h3>
      </div>

      {/* Recording Controls */}
      <div className="flex-1 p-2 flex flex-col">
        <div className="flex flex-col items-center mb-2">
          <button
            onClick={() => setIsRecording(!isRecording)}
            className={`p-3 rounded-full ${
              isRecording ? 'bg-red-500' : 'bg-blue-900'
            } text-white hover:opacity-90 transition-opacity`}
          >
            <FaMicrophone className="h-6 w-6" />
          </button>
          {isRecording && (
            <div className="mt-2 w-full sm:w-1/2 h-4 bg-gray-200 rounded overflow-hidden">
              <div className="h-full bg-blue-600 animate-waveform"></div>
            </div>
          )}
          <span className="mt-1 text-sm text-gray-600">{formatTime(timer)}</span>
        </div>

        {/* Meeting Info */}
        <div className="space-y-2">
          <input
            type="text"
            placeholder="Meeting Title"
            value={meetingTitle}
            onChange={(e) => setMeetingTitle(e.target.value)}
            className="w-full p-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
          <input
            type="text"
            placeholder="Participants (comma-separated)"
            value={participants}
            onChange={(e) => setParticipants(e.target.value)}
            className="w-full p-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>
    </div>
  );
};

export default RecordingInterface;