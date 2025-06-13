import React, { useState } from 'react';
import { FaCalendarAlt, FaTasks, FaUsers } from 'react-icons/fa';
import Calendar from 'react-calendar';
import 'react-calendar/dist/Calendar.css';
import '../calendar.css';

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

  const [selectedDate, setSelectedDate] = useState(new Date(2025, 5, 13));

  return (
    <div className="container px-6 py-6 space-y-6 animate-componentFadeIn">
      {/* Welcome Message */}
      <div className="bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md">
        <h2 className="text-3xl font-bold text-blue-900 dark:text-blue-300 tracking-tight">Welcome, Arnav!</h2>
        <p className="text-gray-600 dark:text-gray-300 mt-1">Here’s an overview of your meetings and tasks.</p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg shadow-md flex items-center space-x-4">
          <FaUsers className="text-blue-900 dark:text-blue-300 text-4xl" />
          <div>
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300">Team Members</h3>
            <p className="text-gray-600 dark:text-gray-300 text-lg">{stats.teamMembers}</p>
          </div>
        </div>
        <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg shadow-md flex items-center space-x-4">
          <FaCalendarAlt className="text-blue-900 dark:text-blue-300 text-4xl" />
          <div>
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300">Total Meetings</h3>
            <p className="text-gray-600 dark:text-Gray-300 text-lg">{stats.totalMeetings}</p>
          </div>
        </div>
        <div className="bg-blue-100 dark:bg-blue-900 p-4 rounded-lg shadow-md flex items-center space-x-4">
          <FaTasks className="text-blue-900 dark:text-blue-300 text-4xl" />
          <div>
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300">Tasks Completed</h3>
            <p className="text-gray-600 dark:text-gray-300 text-lg">{stats.tasksCompleted}</p>
          </div>
        </div>
      </div>

      {/* Recent Meetings and Calendar */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Meetings */}
        <div className="bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300">Recent Meetings</h3>
            <button
              className="text-blue-600 dark:text-blue-400 hover:underline text-sm"
              onClick={() => alert('This will show all meetings in the future!')}
            >
              View All
            </button>
          </div>
          <ul className="space-y-3">
            {recentMeetings.map((meeting) => (
              <li
                key={meeting.id}
                className="p-3 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors duration-200"
              >
                <p className="font-medium text-gray-900 dark:text-gray-100">{meeting.title}</p>
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  {meeting.date} at {meeting.time}
                </p>
              </li>
            ))}
          </ul>
        </div>

        {/* Calendar Overview */}
        <div className="bg-white dark:bg-gray-700 p-4 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-300 mb-4">Calendar Overview</h3>
          <div className="w-full max-w-md mx-auto">
            <Calendar
              className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 shadow-sm"
              calendarType="gregory"
              value={selectedDate}
              onChange={(date) => {
                setSelectedDate(date);
                alert(`Selected date: ${date.toDateString()}`);
              }}
              tileClassName={({ date }) => {
                const meetingDates = recentMeetings.map((meeting) => new Date(meeting.date).toDateString());
                if (meetingDates.includes(date.toDateString())) {
                  return 'bg-blue-200 dark:bg-blue-700 rounded-full';
                }
                return null;
              }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;