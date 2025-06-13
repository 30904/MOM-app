import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FaUserCircle, FaChevronDown, FaBars } from 'react-icons/fa';

const Header = ({ toggleTheme, isDarkMode, onLogout }) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [isNavOpen, setIsNavOpen] = useState(false);
  const [isMeetingsOpen, setIsMeetingsOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
    navigate('/login');
    setIsDropdownOpen(false);
    setIsNavOpen(false);
  };

  return (
    <header className={`flex flex-col sm:flex-row justify-between items-center p-4 sm:p-6 ${isDarkMode ? 'bg-gray-900' : 'bg-gray-100'} shadow-lg`}>
      {/* Logo & Mobile Nav Toggle */}
      <div className="flex justify-between items-center w-full sm:w-auto">
        <h1 className="text-2xl font-bold text-blue-900 tracking-tight">MOM</h1>
        <button className="sm:hidden text-blue-900" onClick={() => setIsNavOpen(!isNavOpen)}>
          <FaBars className="h-6 w-6" />
        </button>
      </div>

      {/* Navigation Bar */}
      <nav className={`flex-col sm:flex-row sm:flex sm:space-x-8 space-y-3 sm:space-y-0 mt-4 sm:mt-0 w-full sm:w-auto ${isNavOpen ? 'flex' : 'hidden sm:flex'}`}>
        <Link to="/" className="text-blue-900 hover:text-blue-600 font-medium text-lg transition-colors duration-200" onClick={() => setIsNavOpen(false)}>Dashboard</Link>
        <div className="relative">
          <button 
            onClick={() => setIsMeetingsOpen(!isMeetingsOpen)} 
            className="text-blue-900 hover:text-blue-600 font-medium text-lg transition-colors duration-200 flex items-center space-x-1"
          >
            <span>Meetings</span>
            <FaChevronDown className={`h-4 w-4 transition-transform duration-200 ${isMeetingsOpen ? 'rotate-180' : ''}`} />
          </button>
          {isMeetingsOpen && (
            <div className="absolute left-0 mt-2 w-48 bg-white dark:bg-gray-800 shadow-lg rounded-md z-10">
              <Link
                to="/meetings/part1"
                className="block px-4 py-2 text-blue-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700"
                onClick={() => {
                  setIsNavOpen(false);
                  setIsMeetingsOpen(false);
                }}
              >
                Recording & Transcription
              </Link>
              <Link
                to="/meetings/part2"
                className="block px-4 py-2 text-blue-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700"
                onClick={() => {
                  setIsNavOpen(false);
                  setIsMeetingsOpen(false);
                }}
              >
                Action Items & Summary
              </Link>
            </div>
          )}
        </div>
        <Link to="/reports" className="text-blue-900 hover:text-blue-600 font-medium text-lg transition-colors duration-200" onClick={() => setIsNavOpen(false)}>Reports</Link>
      </nav>

      {/* User Profile & Theme Toggle */}
      <div className="flex items-center space-x-4 mt-4 sm:mt-0">
        <button onClick={toggleTheme} className="px-4 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200">
          {isDarkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
        <div className="relative">
          <button onClick={() => setIsDropdownOpen(!isDropdownOpen)} className="flex items-center space-x-2">
            <FaUserCircle className="h-8 w-8 text-blue-900" />
            <span className="text-blue-900 font-medium">Arnav</span>
            <FaChevronDown className="h-5 w-5 text-blue-900" />
          </button>
          {isDropdownOpen && (
            <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 shadow-lg rounded-md z-10">
              <Link to="/settings" className="block px-4 py-2 text-blue-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700" onClick={() => setIsDropdownOpen(false)}>Settings</Link>
              <button onClick={handleLogout} className="block w-full text-left px-4 py-2 text-blue-900 dark:text-white hover:bg-gray-100 dark:hover:bg-gray-700">Logout</button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;