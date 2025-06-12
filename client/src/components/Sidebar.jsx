import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { FaTachometerAlt, FaUpload, FaHistory, FaCog, FaFileAlt, FaBars } from 'react-icons/fa';

const Sidebar = () => {
  const [isOpen, setIsOpen] = useState(false);

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
          <h2 className="text-2xl font-bold tracking-tight">MOM App</h2>
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
          <Link
            to="/meetings"
            className="flex items-center space-x-3 p-3 hover:bg-blue-700 rounded-lg transition-colors duration-200"
            onClick={() => setIsOpen(false)}
          >
            <FaFileAlt className="h-5 w-5" />
            <span className="text-lg">Meetings</span>
          </Link>
          <Link
            to="/upload"
            className="flex items-center space-x-3 p-3 hover:bg-blue-700 rounded-lg transition-colors duration-200"
            onClick={() => setIsOpen(false)}
          >
            <FaUpload className="h-5 w-5" />
            <span className="text-lg">Smart Upload</span>
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