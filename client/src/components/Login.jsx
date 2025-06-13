import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const Login = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errors, setErrors] = useState({});
  const navigate = useNavigate();

  const validateForm = () => {
    const newErrors = {};
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!email || !emailRegex.test(email)) {
      newErrors.email = 'Please enter a valid email address.';
    }
    if (!password || password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters long.';
    }
    return newErrors;
  };

  const handleSubmit = () => {
    const validationErrors = validateForm();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }
    onLogin(); // Update login state in App.jsx
    navigate('/');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-800 py-6 px-4 sm:px-6 lg:px-8">
      <div className="card max-w-md w-full">
        <h2 className="text-3xl font-bold text-blue-900 tracking-tight mb-6">Sign In</h2>
        <div className="space-y-6">
          <div>
            <label className="block text-gray-700 dark:text-gray-300 text-lg font-medium mb-2">
              Email Address
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full p-3 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
              placeholder="Enter your email"
            />
            {errors.email && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.email}</p>
            )}
          </div>
          <div>
            <label className="block text-gray-700 dark:text-gray-300 text-lg font-medium mb-2">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full p-3 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white focus:ring-2 focus:ring-blue-600 focus:outline-none"
              placeholder="Enter your password"
            />
            {errors.password && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.password}</p>
            )}
          </div>
          <button
            onClick={handleSubmit}
            className="w-full px-4 py-3 bg-blue-900 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 text-lg font-medium"
          >
            Sign In
          </button>
        </div>
      </div>
    </div>
  );
};

export default Login;