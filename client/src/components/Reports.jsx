import React from 'react';

const Reports = () => {
  const sampleReports = [
    { id: 1, title: 'Q1 Meeting Summary', date: '2025-03-15', status: 'Completed' },
    { id: 2, title: 'Q2 Planning Report', date: '2025-06-10', status: 'Draft' },
  ];

  return (
    <div className="container py-6 space-y-6 animate-componentFadeIn">
      <h2 className="text-3xl font-bold text-blue-900 tracking-tight">Reports</h2>
      <div className="card">
        <table className="w-full text-left">
          <thead>
            <tr className="bg-gray-100 dark:bg-gray-700">
              <th className="p-3 text-blue-900 font-semibold">Title</th>
              <th className="p-3 text-blue-900 font-semibold">Date</th>
              <th className="p-3 text-blue-900 font-semibold">Status</th>
            </tr>
          </thead>
          <tbody>
            {sampleReports.map((report) => (
              <tr key={report.id} className="border-b dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors duration-200">
                <td className="p-3">{report.title}</td>
                <td className="p-3">{report.date}</td>
                <td className="p-3">
                  <span className={`px-2 py-1 rounded ${report.status === 'Completed' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'}`}>
                    {report.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Reports;