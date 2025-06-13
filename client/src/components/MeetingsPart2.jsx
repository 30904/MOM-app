import React from 'react';
import { Link } from 'react-router-dom';
import ActionItemsSection from './ActionItemsSection';
import TranslationFeatures from './TranslationFeatures';
import SummarySection from './SummarySection';

const MeetingsPart2 = ({ summaries }) => {
  return (
    <div className="container pt-4 h-[calc(100vh-5rem)] grid grid-rows-[auto_1fr_auto] gap-4 overflow-hidden">
      <h2 className="text-2xl font-bold text-blue-900 tracking-tight">Action Items & Summary</h2>
      <div className="grid grid-rows-2 gap-4 overflow-hidden min-h-0">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 overflow-hidden min-h-0">
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <ActionItemsSection />
          </div>
          <div className="bg-white rounded-lg shadow-md overflow-hidden">
            <TranslationFeatures />
          </div>
        </div>
        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <SummarySection summaries={summaries} />
        </div>
      </div>
      <div className="flex justify-end">
        <Link
          to="/meetings/part1"
          className="px-4 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 text-lg font-medium"
        >
          Back to Recording & Transcription
        </Link>
      </div>
    </div>
  );
};

export default MeetingsPart2;