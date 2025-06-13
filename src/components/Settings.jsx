import React from 'react';

const Settings = () => {
  return (
    <div className="p-4 space-y-4 animate-componentFadeIn">
      <h2 className="text-2xl font-bold text-blue-900">Settings</h2>
      <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-4">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">User Preferences</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-gray-700 dark:text-gray-300">Email Notifications</label>
            <select className="w-full p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white">
              <option>Enabled</option>
              <option>Disabled</option>
            </select>
          </div>
          <div>
            <label className="block text-gray-700 dark:text-gray-300">Default Language</label>
            <select className="w-full p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white">
              <option>English</option>
              <option>Marathi</option>
              <option>Spanish</option>
            </select>
          </div>
          <button className="px-4 py-2 bg-blue-900 text-white rounded">Save Changes</button>
        </div>
      </div>
    </div>
  );
};

export default Settings;