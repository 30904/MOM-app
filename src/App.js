import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import RecordingInterface from './components/RecordingInterface';
import TranscriptionDisplay from './components/TranscriptionDisplay';
import ActionItemsSection from './components/ActionItemsSection';
import TranslationFeatures from './components/TranslationFeatures';
import SummarySection from './components/SummarySection';
import Reports from './components/Reports';
import SmartUpload from './components/SmartUpload';
import History from './components/History';
import Settings from './components/Settings';

const App = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  // Sample data for testing
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
      <div className={`flex min-h-screen ${isDarkMode ? 'bg-gray-800 dark:bg-gray-900' : 'bg-gray-50'}`}>
        <Sidebar />
        <div className="flex-1">
          <Header toggleTheme={toggleTheme} isDarkMode={isDarkMode} />
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/meetings" element={
              <div className="container py-6 space-y-6">
                <RecordingInterface />
                <TranscriptionDisplay transcriptions={transcriptions} />
                <ActionItemsSection />
                <TranslationFeatures />
                <SummarySection summaries={summaries} />
              </div>
            } />
            <Route path="/reports" element={<Reports />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/login" element={<div className="p-4">Login Page</div>} />
            <Route path="/upload" element={<SmartUpload />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;