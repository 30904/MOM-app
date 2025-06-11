import React from 'react';
import { Link } from 'react-router-dom';
import { FaTachometerAlt, FaUpload, FaHistory, FaCog, FaFileAlt } from 'react-icons/fa';

const Sidebar = () => {
  return (
    <aside className="w-64 bg-blue-900 text-white h-screen p-4 fixed">
      <h2 className="text-xl font-bold mb-4">MOM App</h2>
      <nav className="space-y-2">
        <Link to="/" className="flex items-center space-x-2 p-2 hover:bg-blue-700 rounded">
          <FaTachometerAlt />
          <span>Dashboard</span>
        </Link>
        <Link to="/meetings" className="flex items-center space-x-2 p-2 hover:bg-blue-700 rounded">
          <FaFileAlt />
          <span>Meetings</span>
        </Link>
        <Link to="/upload" className="flex items-center space-x-2 p-2 hover:bg-blue-700 rounded">
          <FaUpload />
          <span>Smart Upload</span>
        </Link>
        <Link to="/history" className="flex items-center space-x-2 p-2 hover:bg-blue-700 rounded">
          <FaHistory />
          <span>History</span>
        </Link>
        <Link to="/settings" className="flex items-center space-x-2 p-2 hover:bg-blue-700 rounded">
          <FaCog />
          <span>Settings</span>
        </Link>
      </nav>
    </aside>
  );
};

export default Sidebar;