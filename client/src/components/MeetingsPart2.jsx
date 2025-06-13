import React from 'react';
import { Link } from 'react-router-dom';
import ActionItemsSection from './ActionItemsSection';
import TranslationFeatures from './TranslationFeatures';
import SummarySection from './SummarySection';

const MeetingsPart2 = ({ summaries }) => {
  return (
    <div className="h-full flex flex-col">
      <div className="p-2 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-700">Action Items & Summary</h3>
      </div>
      <div className="flex-1 p-4 min-h-0">
        <div className="grid grid-rows-2 gap-4 h-full">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
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
      </div>
      <div className="p-2 border-t border-gray-200">
        <div className="flex justify-end">
          <Link
            to="/meetings/part1"
            className="px-4 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 text-sm font-medium"
          >
            Back to Recording & Transcription
          </Link>
        </div>
      </div>
    </div>
  );
};

export default MeetingsPart2;