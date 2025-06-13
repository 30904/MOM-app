import React from 'react';
import { FaSearch } from 'react-icons/fa';

const History = () => {
  const pastMeetings = [
    { id: 1, title: 'Team Sync', date: '2025-06-05', duration: '45 mins' },
    { id: 2, title: 'Client Review', date: '2025-06-08', duration: '1 hr' },
  ];

  return (
    <div className="p-4 space-y-4 animate-componentFadeIn">
      <h2 className="text-2xl font-bold text-blue-900">Meeting History</h2>
      <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-4">
        <div className="flex items-center mb-4">
          <FaSearch className="text-blue-900 mr-2" />
          <input
            type="text"
            placeholder="Search meetings..."
            className="w-full p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white"
          />
        </div>
        <ul className="space-y-2">
          {pastMeetings.map((meeting) => (
            <li key={meeting.id} className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
              <p className="font-medium">{meeting.title}</p>
              <p className="text-sm text-gray-600 dark:text-gray-300">
                {meeting.date} • {meeting.duration}
              </p>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default History;