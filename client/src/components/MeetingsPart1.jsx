import React from 'react';
import { Link } from 'react-router-dom';
import RecordingInterface from './RecordingInterface';
import TranscriptionDisplay from './TranscriptionDisplay';

const MeetingsPart1 = ({ transcriptions }) => {
  return (
    <div className="container pt-4 h-[calc(100vh-5rem)] grid grid-rows-[auto_1fr_auto] gap-4 overflow-hidden">
      <h2 className="text-2xl font-bold text-blue-900 tracking-tight">Recording & Transcription</h2>
      <div className="grid grid-rows-2 gap-4 overflow-hidden min-h-0">
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <RecordingInterface />
        </div>
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <TranscriptionDisplay transcriptions={transcriptions} />
        </div>
      </div>
      <div className="flex justify-end">
        <Link
          to="/meetings/part2"
          className="px-4 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 text-lg font-medium"
        >
          Go to Action Items & Summary
        </Link>
      </div>
    </div>
  );
};

export default MeetingsPart1;