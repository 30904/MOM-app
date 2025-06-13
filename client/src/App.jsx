import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import MeetingsPart1 from './components/MeetingsPart1';
import MeetingsPart2 from './components/MeetingsPart2';
import Reports from './components/Reports';
import Upload from './components/Upload';
import History from './components/History';
import Settings from './components/Settings';
import Login from './components/Login';

const App = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const toggleTheme = () => setIsDarkMode(!isDarkMode);

  // Check login state on app load
  useEffect(() => {
    const loggedIn = localStorage.getItem('isLoggedIn') === 'true';
    setIsLoggedIn(loggedIn);
  }, []);

  const handleLogin = () => {
    setIsLoggedIn(true);
    localStorage.setItem('isLoggedIn', 'true');
  };

  const handleLogout = () => {
    setIsLoggedIn(false);
    localStorage.setItem('isLoggedIn', 'false');
  };

  // Sample data for testing
  const transcriptions = [
    { timestamp: '00:01', speaker: 'Speaker 1', text: "Let's start the meeting." },
    { timestamp: '00:03', speaker: 'Speaker 2', text: "Sure, I'll take notes." },
  ];
  const summaries = [
    { title: 'Meeting Summary', type: 'summary', points: ['Discussed project timeline', 'Assigned tasks'] },
    { title: 'Action Items', type: 'action', points: ['Piyush: Prepare slides', 'Arnav: Schedule follow-up'] },
  ];

  return (
    <Router>
      <div className={`h-screen flex ${isDarkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
        {/* Conditionally render Sidebar only if logged in */}
        {isLoggedIn && <Sidebar />}
        <div className="flex-1 flex flex-col min-h-0">
          {/* Conditionally render Header only if logged in */}
          {isLoggedIn && <Header toggleTheme={toggleTheme} isDarkMode={isDarkMode} onLogout={handleLogout} />}
          <main className="flex-1 min-h-0">
            <Routes>
              <Route
                path="/login"
                element={
                  isLoggedIn ? (
                    <Navigate to="/" />
                  ) : (
                    <Login onLogin={handleLogin} />
                  )
                }
              />
              <Route
                path="/"
                element={
                  isLoggedIn ? (
                    <Dashboard />
                  ) : (
                    <Navigate to="/login" />
                  )
                }
              />
              <Route
                path="/meetings/part1"
                element={
                  isLoggedIn ? (
                    <MeetingsPart1 transcriptions={transcriptions} />
                  ) : (
                    <Navigate to="/login" />
                  )
                }
              />
              <Route
                path="/meetings/part2"
                element={
                  isLoggedIn ? (
                    <MeetingsPart2 summaries={summaries} />
                  ) : (
                    <Navigate to="/login" />
                  )
                }
              />
              <Route
                path="/reports"
                element={
                  isLoggedIn ? (
                    <Reports />
                  ) : (
                    <Navigate to="/login" />
                  )
                }
              />
              <Route
                path="/settings"
                element={
                  isLoggedIn ? (
                    <Settings />
                  ) : (
                    <Navigate to="/login" />
                  )
                }
              />
              <Route
                path="/upload"
                element={
                  isLoggedIn ? (
                    <Upload />
                  ) : (
                    <Navigate to="/login" />
                  )
                }
              />
              <Route
                path="/history"
                element={
                  isLoggedIn ? (
                    <History />
                  ) : (
                    <Navigate to="/login" />
                  )
                }
              />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
};

export default App;