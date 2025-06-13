import React, { useState, useEffect } from 'react';

const SkeletonLoader = () => (
  <div className="space-y-2">
    {[...Array(3)].map((_, index) => (
      <div key={index} className="flex items-start space-x-2 animate-pulse">
        <div className="w-12 h-3 bg-gray-200 rounded"></div>
        <div className="flex-1 p-2 bg-gray-200 rounded-lg">
          <div className="h-3 bg-gray-300 rounded w-1/4 mb-1"></div>
          <div className="h-3 bg-gray-300 rounded w-3/4"></div>
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
    <div className="h-full flex flex-col">
      {/* Header with Progress */}
      <div className="p-2 border-b border-gray-200">
        <div className="flex justify-between items-center mb-2">
          <h3 className="text-sm font-medium text-gray-700">Transcription</h3>
          {progress < 100 && (
            <span className="text-xs text-gray-500">{progress}%</span>
          )}
        </div>
        {progress < 100 && (
          <div className="w-full bg-gray-200 rounded-full h-1.5">
            <div
              className="bg-blue-600 h-1.5 rounded-full"
              style={{ width: `${progress}%`, transition: 'width 0.5s ease-in-out' }}
            ></div>
          </div>
        )}
      </div>

      {/* Search & Filter */}
      <div className="p-2 border-b border-gray-200">
        {isLoading ? (
          <div className="flex justify-between w-full">
            <div className="w-1/3 h-8 bg-gray-200 rounded animate-pulse"></div>
            <div className="flex space-x-1">
              <div className="w-24 h-8 bg-gray-200 rounded animate-pulse"></div>
              <div className="w-24 h-8 bg-gray-200 rounded animate-pulse"></div>
            </div>
          </div>
        ) : (
          <div className="flex justify-between items-center space-x-2">
            <input
              type="text"
              placeholder="Search..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1 p-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
            <select
              value={selectedSpeaker}
              onChange={(e) => setSelectedSpeaker(e.target.value)}
              className="p-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="">All Speakers</option>
              <option value="Speaker 1">Speaker 1</option>
              <option value="Speaker 2">Speaker 2</option>
            </select>
            <select
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(e.target.value)}
              className="p-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="">All Time</option>
              <option value="00:00 - 01:00">0:00 - 1:00</option>
              <option value="01:00 - 02:00">1:00 - 2:00</option>
            </select>
          </div>
        )}
      </div>

      {/* Transcription Blocks */}
      <div className="flex-1 p-2 overflow-y-auto max-h-[calc(100vh-25rem)]">
        {isLoading ? (
          <SkeletonLoader />
        ) : (
          <div className="space-y-2">
            {filteredTranscriptions.map((transcription, index) => (
              <div key={index} className="flex items-start space-x-2">
                <span className="text-xs text-gray-500 whitespace-nowrap">{transcription.timestamp}</span>
                <div className={`p-2 rounded-lg ${transcription.speaker === 'Speaker 1' ? 'bg-blue-50' : 'bg-pink-50'} flex-1`}>
                  <p className="text-xs font-medium text-gray-700">{transcription.speaker}</p>
                  <p className="text-sm text-gray-600">{transcription.text}</p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default TranscriptionDisplay;