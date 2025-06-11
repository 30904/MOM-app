import React, { useState, useEffect } from 'react';

const SkeletonLoader = () => (
  <div className="space-y-4">
    {[...Array(3)].map((_, index) => (
      <div key={index} className="flex items-start space-x-4 animate-pulse">
        <div className="w-16 h-4 bg-gray-200 rounded"></div>
        <div className="flex-1 p-3 bg-gray-200 rounded-lg">
          <div className="h-4 bg-gray-300 rounded w-1/4 mb-2"></div>
          <div className="h-4 bg-gray-300 rounded w-3/4"></div>
        </div>
      </div>
    ))}
  </div>
);

const TranscriptionDisplay = ({ transcriptions }) => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading delay
    const timer = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg">
      {/* Search & Filter */}
      <div className="flex justify-between mb-4">
        {isLoading ? (
          <div className="flex justify-between w-full">
            <div className="w-1/3 h-10 bg-gray-200 rounded-lg animate-pulse"></div>
            <div className="flex space-x-2">
              <div className="w-32 h-10 bg-gray-200 rounded-lg animate-pulse"></div>
              <div className="w-32 h-10 bg-gray-200 rounded-lg animate-pulse"></div>
            </div>
          </div>
        ) : (
          <>
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
          </>
        )}
      </div>

      {/* Transcription Blocks */}
      {isLoading ? (
        <SkeletonLoader />
      ) : (
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
      )}
    </div>
  );
};

export default TranscriptionDisplay;