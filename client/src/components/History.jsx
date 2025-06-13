import React, { useState } from 'react';
import { FaSearch, FaFilter, FaSort, FaCalendarAlt, FaClock, FaUsers, FaDownload, FaShare } from 'react-icons/fa';

const History = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('date');
  const [filterStatus, setFilterStatus] = useState('all');

  const pastMeetings = [
    {
      id: 1,
      title: 'Team Sync',
      date: '2025-06-05',
      duration: '45 mins',
      participants: 8,
      status: 'completed',
      type: 'internal',
      summary: 'Discussed Q2 goals and project timelines',
    },
    {
      id: 2,
      title: 'Client Review',
      date: '2025-06-08',
      duration: '1 hr',
      participants: 12,
      status: 'completed',
      type: 'client',
      summary: 'Presented project progress and gathered feedback',
    },
    {
      id: 3,
      title: 'Sprint Planning',
      date: '2025-06-10',
      duration: '1.5 hrs',
      participants: 6,
      status: 'completed',
      type: 'internal',
      summary: 'Planned sprint goals and assigned tasks',
    },
  ];

  const filteredMeetings = pastMeetings
    .filter((meeting) => {
      const matchesSearch = meeting.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        meeting.summary.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesFilter = filterStatus === 'all' || meeting.status === filterStatus;
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      if (sortBy === 'date') return new Date(b.date) - new Date(a.date);
      if (sortBy === 'duration') return b.duration.localeCompare(a.duration);
      if (sortBy === 'participants') return b.participants - a.participants;
      return 0;
    });

  return (
    <div className="h-full flex flex-col">
      <div className="p-2 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h3 className="text-sm font-medium text-gray-700">Meeting History</h3>
          <div className="flex items-center space-x-2">
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="all">All Status</option>
              <option value="completed">Completed</option>
              <option value="cancelled">Cancelled</option>
            </select>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="date">Sort by Date</option>
              <option value="duration">Sort by Duration</option>
              <option value="participants">Sort by Participants</option>
            </select>
          </div>
        </div>
      </div>

      <div className="flex-1 p-4 min-h-0">
        <div className="mb-4">
          <div className="relative">
            <input
              type="text"
              placeholder="Search meetings..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full p-2 pl-10 border border-gray-300 rounded-lg focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
            <FaSearch className="absolute left-3 top-3 text-gray-400" />
          </div>
        </div>

        <div className="space-y-3">
          {filteredMeetings.map((meeting) => (
            <div
              key={meeting.id}
              className="bg-white rounded-lg shadow p-3 hover:shadow-md transition-shadow duration-200"
            >
              <div className="flex justify-between items-start mb-2">
                <div>
                  <h3 className="text-base font-semibold text-gray-900">{meeting.title}</h3>
                  <p className="text-sm text-gray-500">{meeting.summary}</p>
                </div>
                <span
                  className={`text-xs px-2 py-1 rounded-full ${
                    meeting.type === 'internal'
                      ? 'bg-blue-100 text-blue-700'
                      : 'bg-purple-100 text-purple-700'
                  }`}
                >
                  {meeting.type}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-3 mt-3">
                <div className="flex items-center text-sm text-gray-600">
                  <FaCalendarAlt className="mr-2 text-gray-400" />
                  {meeting.date}
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <FaClock className="mr-2 text-gray-400" />
                  {meeting.duration}
                </div>
                <div className="flex items-center text-sm text-gray-600">
                  <FaUsers className="mr-2 text-gray-400" />
                  {meeting.participants} participants
                </div>
              </div>

              <div className="mt-3 flex justify-end space-x-2">
                <button className="p-1.5 text-gray-600 hover:text-blue-600 transition-colors duration-200">
                  <FaDownload />
                </button>
                <button className="p-1.5 text-gray-600 hover:text-blue-600 transition-colors duration-200">
                  <FaShare />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default History;