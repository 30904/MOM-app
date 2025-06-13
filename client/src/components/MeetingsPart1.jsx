import React from 'react';
import { Link } from 'react-router-dom';
import RecordingInterface from './RecordingInterface';
import TranscriptionDisplay from './TranscriptionDisplay';

const MeetingsPart1 = ({ transcriptions }) => {
  return (
    <div className="h-full flex flex-col">
      <div className="p-2 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-700">Recording & Transcription</h3>
      </div>
      <div className="flex-1 p-4 min-h-0">
        <div className="grid grid-rows-2 gap-4 h-full">
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <RecordingInterface />
          </div>
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <TranscriptionDisplay transcriptions={transcriptions} />
          </div>
        </div>
      </div>
      <div className="p-2 border-t border-gray-200">
        <div className="flex justify-end">
          <Link
            to="/meetings/part2"
            className="px-4 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 text-sm font-medium"
          >
            Go to Action Items & Summary
          </Link>
        </div>
      </div>
    </div>
  );
};

export default MeetingsPart1;