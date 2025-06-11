import React from 'react';

const TranscriptionDisplay = ({ transcriptions }) => {
  return (
    <div className="p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg">
      {/* Search & Filter */}
      <div className="flex justify-between mb-4">
        <input
          type="text"
          placeholder="Search transcriptions..."
          className="p-2 border border-blue-900 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 dark:bg-gray-700 dark:text-white"
        />
        <div className="flex space-x-2">
          <select className="p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white">
            <option>Filter by Speaker</option>
            <option>Speaker 1</option>
            <option>Speaker 2</option>
          </select>
          <select className="p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white">
            <option>Filter by Time</option>
            <option>0:00 - 1:00</option>
            <option>1:00 - 2:00</option>
          </select>
        </div>
      </div>

      {/* Transcription Blocks */}
      <div className="space-y-4">
        {transcriptions.map((transcription, index) => (
          <div key={index} className="flex items-start space-x-4">
            <span className="text-gray-500 text-sm">{transcription.timestamp}</span>
            <div className={`p-3 rounded-lg ${transcription.speaker === 'Speaker 1' ? 'bg-blue-100' : 'bg-pink-100'} flex-1`}>
              <p className="font-semibold">{transcription.speaker}</p>
              <p>{transcription.text}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TranscriptionDisplay;