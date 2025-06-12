import React from 'react';
import { FaCalendarAlt, FaTasks, FaUsers } from 'react-icons/fa';

const Dashboard = () => {
  const stats = {
    totalMeetings: 12,
    tasksCompleted: 8,
    teamMembers: 5,
  };

  const recentMeetings = [
    { id: 1, title: 'Project Kickoff', date: '2025-06-10', time: '10:00 AM' },
    { id: 2, title: 'Sprint Planning', date: '2025-06-11', time: '2:00 PM' },
  ];

  return (
    <div className="p-4 space-y-4 animate-componentFadeIn">
      {/* Welcome Message */}
      <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-4">
        <h2 className="text-2xl font-bold text-blue-900">Welcome, Piyush!</h2>
        <p className="text-gray-600 dark:text-gray-300">Here’s an overview of your meetings and tasks.</p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-blue-100 dark:bg-gray-700 p-4 rounded-lg shadow-md flex items-center space-x-4">
          <FaUsers className="text-blue-900 text-3xl" />
          <div>
            <h3 className="text-lg font-semibold text-blue-900">Team Members</h3>
            <p className="text-gray-600 dark:text-gray-300">{stats.teamMembers}</p>
          </div>
        </div>
        <div className="bg-blue-100 dark:bg-gray-700 p-4 rounded-lg shadow-md flex items-center space-x-4">
          <FaCalendarAlt className="text-blue-900 text-3xl" />
          <div>
            <h3 className="text-lg font-semibold text-blue-900">Total Meetings</h3>
            <p className="text-gray-600 dark:text-gray-300">{stats.totalMeetings}</p>
          </div>
        </div>
        <div className="bg-blue-100 dark:bg-gray-700 p-4 rounded-lg shadow-md flex items-center space-x-4">
          <FaTasks className="text-blue-900 text-3xl" />
          <div>
            <h3 className="text-lg font-semibold text-blue-900">Tasks Completed</h3>
            <p className="text-gray-600 dark:text-gray-300">{stats.tasksCompleted}</p>
          </div>
        </div>
      </div>

      {/* Recent Meetings and Calendar */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Recent Meetings */}
        <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-4">
          <h3 className="text-lg font-semibold text-blue-900 mb-2">Recent Meetings</h3>
          <ul className="space-y-2">
            {recentMeetings.map((meeting) => (
              <li key={meeting.id} className="p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
                <p className="font-medium">{meeting.title}</p>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  {meeting.date} at {meeting.time}
                </p>
              </li>
            ))}
          </ul>
        </div>

        {/* Calendar Placeholder */}
        <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-4">
          <h3 className="text-lg font-semibold text-blue-900 mb-2">Calendar Overview</h3>
          <div className="h-40 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">Calendar Widget (To be implemented)</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
