import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import RecordingInterface from './components/RecordingInterface';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import ActionItemsSection from './components/ActionItemsSection';
import TranslationFeatures from './components/TranslationFeatures';
import SummarySection from './components/SummarySection';

const App = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  const transcriptions = [
    { timestamp: '00:01', speaker: 'Speaker 1', text: 'Let’s start the meeting.' },
    { timestamp: '00:03', speaker: 'Speaker 2', text: 'Sure, I’ll take notes.' },
  ];
  const summaries = [
    { title: 'Meeting Summary', type: 'summary', points: ['Discussed project timeline', 'Assigned tasks'] },
    { title: 'Action Items', type: 'action', points: ['Piyush: Prepare slides', 'Arnav: Schedule follow-up'] },
  ];

  return (
    <Router>
      <div className={`flex min-h-screen ${isDarkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
        <Sidebar />
        <div className="flex-1 ml-64">
          <Header toggleTheme={toggleTheme} isDarkMode={isDarkMode} />
          <Routes>
            <Route path="/" element={<div className="p-4">Dashboard Content</div>} />
            <Route path="/meetings" element={
              <div className="p-4 space-y-4">
                <RecordingInterface />
                <TranscriptionDisplay transcriptions={transcriptions} />
                <ActionItemsSection />
                <TranslationFeatures />
                <SummarySection summaries={summaries} />
              </div>
            } />
            <Route path="/reports" element={<div className="p-4">Reports Content</div>} />
            <Route path="/settings" element={<div className="p-4">Settings Page</div>} />
            <Route path="/login" element={<div className="p-4">Login Page</div>} />
            <Route path="/upload" element={<div className="p-4">Smart Upload Page</div>} />
            <Route path="/history" element={<div className="p-4">History Page</div>} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;