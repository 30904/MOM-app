import React, { useState } from 'react';
import { FaCalendarAlt, FaTasks, FaUsers, FaChartLine, FaEllipsisV, FaPlus } from 'react-icons/fa';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = () => {
  const [selectedTimeRange, setSelectedTimeRange] = useState('week');

  const stats = {
    totalMeetings: 12,
    tasksCompleted: 8,
    teamMembers: 5,
    upcomingMeetings: 3,
  };

  const recentMeetings = [
    { id: 1, title: 'Project Kickoff', date: '2025-06-10', time: '10:00 AM', status: 'upcoming', participants: 8 },
    { id: 2, title: 'Sprint Planning', date: '2025-06-11', time: '2:00 PM', status: 'upcoming', participants: 6 },
    { id: 3, title: 'Client Review', date: '2025-06-09', time: '11:00 AM', status: 'completed', participants: 4 },
  ];

  const meetingData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'Meetings',
        data: [3, 2, 4, 1, 5, 0, 2],
        borderColor: 'rgb(30, 58, 138)',
        backgroundColor: 'rgba(30, 58, 138, 0.5)',
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          stepSize: 1,
        },
      },
    },
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-2 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h3 className="text-sm font-medium text-gray-700">Dashboard</h3>
          <div className="flex items-center space-x-2">
            <select
              value={selectedTimeRange}
              onChange={(e) => setSelectedTimeRange(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="week">This Week</option>
              <option value="month">This Month</option>
              <option value="year">This Year</option>
            </select>
          </div>
        </div>
      </div>

      <div className="flex-1 p-4 min-h-0">
        {/* Quick Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Total Meetings</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.totalMeetings}</p>
              </div>
              <FaCalendarAlt className="text-blue-900 text-xl" />
            </div>
            <div className="mt-2">
              <span className="text-xs text-green-600">↑ 12% from last week</span>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Tasks Completed</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.tasksCompleted}</p>
              </div>
              <FaTasks className="text-blue-900 text-xl" />
            </div>
            <div className="mt-2">
              <span className="text-xs text-green-600">↑ 8% from last week</span>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Team Members</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.teamMembers}</p>
              </div>
              <FaUsers className="text-blue-900 text-xl" />
            </div>
            <div className="mt-2">
              <span className="text-xs text-gray-500">Active now</span>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-500">Upcoming</p>
                <p className="text-2xl font-semibold text-gray-900">{stats.upcomingMeetings}</p>
              </div>
              <FaChartLine className="text-blue-900 text-xl" />
            </div>
            <div className="mt-2">
              <span className="text-xs text-blue-600">Next: Project Kickoff</span>
            </div>
          </div>
        </div>

        {/* Charts and Recent Meetings */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Meeting Activity Chart */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-medium text-gray-700">Meeting Activity</h3>
              <button className="text-gray-400 hover:text-gray-600">
                <FaEllipsisV />
              </button>
            </div>
            <div className="h-48">
              <Line data={meetingData} options={chartOptions} />
            </div>
          </div>

          {/* Recent Meetings */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-sm font-medium text-gray-700">Recent Meetings</h3>
              <button className="text-blue-900 hover:text-blue-700">
                <FaPlus className="text-sm" />
              </button>
            </div>
            <div className="space-y-3">
              {recentMeetings.map((meeting) => (
                <div
                  key={meeting.id}
                  className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium text-gray-900">{meeting.title}</p>
                      <p className="text-xs text-gray-500">
                        {meeting.date} at {meeting.time}
                      </p>
                    </div>
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${
                        meeting.status === 'upcoming'
                          ? 'bg-blue-100 text-blue-700'
                          : 'bg-green-100 text-green-700'
                      }`}
                    >
                      {meeting.status}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center text-xs text-gray-500">
                    <FaUsers className="mr-1" />
                    {meeting.participants} participants
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;