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
    <div className="container py-6 space-y-6 animate-componentFadeIn">
      {/* Welcome Message */}
      <div className="card">
        <h2 className="text-3xl font-bold text-blue-900 tracking-tight">Welcome, Piyush!</h2>
        <p className="text-gray-600 dark:text-gray-300 mt-1">Here’s an overview of your meetings and tasks.</p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="card flex items-center space-x-4">
          <FaUsers className="text-blue-900 text-4xl" />
          <div>
            <h3 className="text-xl font-semibold text-blue-900">Team Members</h3>
            <p className="text-gray-600 dark:text-gray-300 text-lg">{stats.teamMembers}</p>
          </div>
        </div>
        <div className="card flex items-center space-x-4">
          <FaCalendarAlt className="text-blue-900 text-4xl" />
          <div>
            <h3 className="text-xl font-semibold text-blue-900">Total Meetings</h3>
            <p className="text-gray-600 dark:text-gray-300 text-lg">{stats.totalMeetings}</p>
          </div>
        </div>
        <div className="card flex items-center space-x-4">
          <FaTasks className="text-blue-900 text-4xl" />
          <div>
            <h3 className="text-xl font-semibold text-blue-900">Tasks Completed</h3>
            <p className="text-gray-600 dark:text-gray-300 text-lg">{stats.tasksCompleted}</p>
          </div>
        </div>
      </div>

      {/* Recent Meetings and Calendar */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Meetings */}
        <div className="card">
          <h3 className="text-xl font-semibold text-blue-900 mb-4">Recent Meetings</h3>
          <ul className="space-y-3">
            {recentMeetings.map((meeting) => (
              <li key={meeting.id} className="p-3 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-200">
                <p className="font-medium text-gray-900 dark:text-gray-100">{meeting.title}</p>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  {meeting.date} at {meeting.time}
                </p>
              </li>
            ))}
          </ul>
        </div>

        {/* Calendar Placeholder */}
        <div className="card">
          <h3 className="text-xl font-semibold text-blue-900 mb-4">Calendar Overview</h3>
          <div className="h-48 bg-gray-200 dark:bg-gray-700 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">Calendar Widget (To be implemented)</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;