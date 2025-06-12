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
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedSpeaker, setSelectedSpeaker] = useState('');
  const [selectedTimeRange, setSelectedTimeRange] = useState('');
  const [progress, setProgress] = useState(0);

  // Simulate loading delay
  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  // Simulate real-time transcription progress
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => (prev < 100 ? prev + 10 : 100));
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Filter transcriptions based on search, speaker, and time range
  const filteredTranscriptions = transcriptions.filter((transcription) => {
    const matchesSearch = transcription.text.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesSpeaker = selectedSpeaker ? transcription.speaker === selectedSpeaker : true;
    const matchesTime = selectedTimeRange
      ? transcription.timestamp >= selectedTimeRange.split(' - ')[0] &&
        transcription.timestamp <= selectedTimeRange.split(' - ')[1]
      : true;
    return matchesSearch && matchesSpeaker && matchesTime;
  });

  return (
    <div className="p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg animate-componentFadeIn">
      {/* Real-Time Progress Bar */}
      {progress < 100 && (
        <div className="mb-4">
          <p className="text-gray-700 dark:text-gray-300">Transcription in Progress...</p>
          <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
            <div
              className="bg-blue-600 h-2.5 rounded-full"
              style={{ width: `${progress}%`, transition: 'width 0.5s ease-in-out' }}
            ></div>
          </div>
        </div>
      )}

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
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="p-2 border border-blue-900 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600 dark:bg-gray-700 dark:text-white"
            />
            <div className="flex space-x-2">
              <select
                value={selectedSpeaker}
                onChange={(e) => setSelectedSpeaker(e.target.value)}
                className="p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white"
              >
                <option value="">Filter by Speaker</option>
                <option value="Speaker 1">Speaker 1</option>
                <option value="Speaker 2">Speaker 2</option>
              </select>
              <select
                value={selectedTimeRange}
                onChange={(e) => setSelectedTimeRange(e.target.value)}
                className="p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white"
              >
                <option value="">Filter by Time</option>
                <option value="00:00 - 01:00">0:00 - 1:00</option>
                <option value="01:00 - 02:00">1:00 - 2:00</option>
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
          {filteredTranscriptions.map((transcription, index) => (
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