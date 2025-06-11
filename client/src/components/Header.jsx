import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FaUserCircle, FaChevronDown } from 'react-icons/fa';

const Header = ({ toggleTheme, isDarkMode }) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogout = () => {
    navigate('/login');
  };

  return (
    <header className={`flex justify-between items-center p-4 ${isDarkMode ? 'bg-gray-900' : 'bg-gray-100'} shadow-md`}>
      {/* Logo & Branding */}
      <div className="flex items-center">
        <h1 className="text-2xl font-bold text-blue-900">MOM App</h1>
      </div>

      {/* Navigation Bar */}
      <nav className="flex space-x-6">
        <Link to="/" className="text-blue-900 hover:text-blue-600">Dashboard</Link>
        <Link to="/meetings" className="text-blue-900 hover:text-blue-600">Meetings</Link>
        <Link to="/reports" className="text-blue-900 hover:text-blue-600">Reports</Link>
      </nav>

      {/* User Profile & Theme Toggle */}
      <div className="flex items-center space-x-4">
        <button onClick={toggleTheme} className="px-3 py-1 bg-blue-900 text-white rounded">
          {isDarkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
        <div className="relative">
          <button onClick={() => setIsDropdownOpen(!isDropdownOpen)} className="flex items-center space-x-2">
            <FaUserCircle className="h-8 w-8 text-blue-900" />
            <span className="text-blue-900">Piyush</span>
            <FaChevronDown className="h-5 w-5 text-blue-900" />
          </button>
          {isDropdownOpen && (
            <div className="absolute right-0 mt-2 w-48 bg-white shadow-lg rounded-md z-10">
              <Link to="/settings" className="block px-4 py-2 text-blue-900 hover:bg-gray-100">Settings</Link>
              <button onClick={handleLogout} className="block w-full text-left px-4 py-2 text-blue-900 hover:bg-gray-100">Logout</button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};

export default Header;