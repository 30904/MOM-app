import React, { useState } from 'react';
import { FaChevronDown, FaChevronUp, FaSearch, FaCheck } from 'react-icons/fa';

const SummarySection = ({ summaries }) => {
  const [expanded, setExpanded] = useState({});

  const toggleSection = (index) => {
    setExpanded((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  const handleExport = (format) => {
    console.log(`Exporting as ${format}`);
  };

  return (
    <div className="p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg">
      {/* Export Buttons */}
      <div className="flex justify-end mb-4 space-x-2">
        <button onClick={() => handleExport('PDF')} className="px-3 py-1 bg-blue-900 text-white rounded">PDF</button>
        <button onClick={() => handleExport('DOCX')} className="px-3 py-1 bg-blue-900 text-white rounded">DOCX</button>
        <button onClick={() => handleExport('TXT')} className="px-3 py-1 bg-blue-900 text-white rounded">TXT</button>
      </div>

      {/* Summary Sections */}
      {summaries.map((summary, index) => (
        <div key={index} className="mb-2">
          <button
            onClick={() => toggleSection(index)}
            className="w-full flex justify-between items-center p-3 bg-gray-100 dark:bg-gray-700 rounded-lg"
          >
            <span className="flex items-center space-x-2">
              {summary.type === 'summary' ? <FaSearch className="text-blue-900" /> : <FaCheck className="text-green-600" />}
              <span>{summary.title}</span>
            </span>
            {expanded[index] ? <FaChevronUp /> : <FaChevronDown />}
          </button>
          {expanded[index] && (
            <ul className="p-3 list-disc list-inside">
              {summary.points.map((point, i) => (
                <li key={i} className="text-gray-700 dark:text-gray-300">{point}</li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
};

export default SummarySection;