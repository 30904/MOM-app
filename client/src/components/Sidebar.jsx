import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FaTachometerAlt, FaUpload, FaHistory, FaCog, FaFileAlt, FaBars, FaChevronDown } from 'react-icons/fa';

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMeetingsOpen, setIsMeetingsOpen] = useState(false);

  return (
    <>
      {/* Mobile Toggle Button */}
      <button
        className="sm:hidden p-4 text-blue-900"
        onClick={() => setIsOpen(!isOpen)}
      >
        <FaBars />
      </button>

      {/* Sidebar */}
      <aside
        className={`fixed top-0 left-0 h-screen bg-blue-900 text-white p-6 transition-transform duration-300 w-64 z-50 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        } sm:translate-x-0 sm:static sm:w-64 sm:block shadow-lg`}
      >
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-2xl font-bold tracking-tight">MOM</h2>
          <button className="sm:hidden text-white" onClick={() => setIsOpen(false)}>
            <FaBars />
          </button>
        </div>
        <nav className="space-y-3">
          <Link
            to="/"
            className="flex items-center space-x-3 p-3 hover:bg-blue-700 rounded-lg transition-colors duration-200"
            onClick={() => setIsOpen(false)}
          >
            <FaTachometerAlt className="h-5 w-5" />
            <span className="text-lg">Dashboard</span>
          </Link>
          <div>
            <button
              onClick={() => setIsMeetingsOpen(!isMeetingsOpen)}
              className="flex items-center space-x-3 p-3 w-full text-left hover:bg-blue-700 rounded-lg transition-colors duration-200"
            >
              <FaFileAlt className="h-5 w-5" />
              <span className="text-lg">Meetings</span>
              <FaChevronDown className={`h-4 w-4 ml-auto transition-transform ${isMeetingsOpen ? 'rotate-180' : ''}`} />
            </button>
            {isMeetingsOpen && (
              <div className="space-y-1 pl-8">
                <Link
                  to="/meetings/part1"
                  className="flex items-center space-x-3 p-3 hover:bg-blue-700 rounded-lg transition-colors duration-200"
                  onClick={() => setIsOpen(false)}
                >
                  <span className="text-base">Recording & Transcription</span>
                </Link>
                <Link
                  to="/meetings/part2"
                  className="flex items-center space-x-3 p-3 hover:bg-blue-700 rounded-lg transition-colors duration-200"
                  onClick={() => setIsOpen(false)}
                >
                  <span className="text-base">Action Items & Summary</span>
                </Link>
              </div>
            )}
          </div>
          <Link
            to="/upload"
            className="flex items-center space-x-3 p-3 hover:bg-blue-700 rounded-lg transition-colors duration-200"
            onClick={() => setIsOpen(false)}
          >
            <FaUpload className="h-5 w-5" />
            <span className="text-lg">Upload</span>
          </Link>
          <Link
            to="/history"
            className="flex items-center space-x-3 p-3 hover:bg-blue-700 rounded-lg transition-colors duration-200"
            onClick={() => setIsOpen(false)}
          >
            <FaHistory className="h-5 w-5" />
            <span className="text-lg">History</span>
          </Link>
          <Link
            to="/settings"
            className="flex items-center space-x-3 p-3 hover:bg-blue-700 rounded-lg transition-colors duration-200"
            onClick={() => setIsOpen(false)}
          >
            <FaCog className="h-5 w-5" />
            <span className="text-lg">Settings</span>
          </Link>
        </nav>
      </aside>

      {/* Overlay for Mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black opacity-50 sm:hidden z-40"
          onClick={() => setIsOpen(false)}
        ></div>
      )}
    </>
  );
};

export default Sidebar;