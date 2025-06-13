import React, { useState, useEffect } from 'react';
import { FaChevronDown, FaChevronUp, FaSearch, FaCheck, FaFilePdf, FaFileWord, FaFileAlt } from 'react-icons/fa';

const SummarySkeletonLoader = () => (
  <div className="space-y-2">
    {[...Array(2)].map((_, index) => (
      <div key={index} className="animate-pulse">
        <div className="w-full h-12 bg-gray-200 rounded-lg mb-2"></div>
      </div>
    ))}
  </div>
);

const SummarySection = ({ summaries }) => {
  const [expanded, setExpanded] = useState({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading delay
    const timer = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  const toggleSection = (index) => {
    setExpanded((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  const handleExport = (format) => {
    console.log(`Exporting as ${format}`);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Meeting Summary</h3>
      </div>
      
      <div className="flex-1 p-4 overflow-hidden">
        <div className="flex flex-col h-full">
          {/* Export Buttons */}
          <div className="flex justify-end mb-4 space-x-2">
            {isLoading ? (
              <div className="flex space-x-2">
                <div className="w-16 h-8 bg-gray-200 rounded animate-pulse"></div>
                <div className="w-16 h-8 bg-gray-200 rounded animate-pulse"></div>
                <div className="w-16 h-8 bg-gray-200 rounded animate-pulse"></div>
              </div>
            ) : (
              <>
                <button onClick={() => handleExport('PDF')} className="inline-flex items-center px-3 py-1.5 bg-blue-900 text-white rounded text-sm hover:bg-blue-700 transition-colors">
                  <FaFilePdf className="mr-1.5" /> PDF
                </button>
                <button onClick={() => handleExport('DOCX')} className="inline-flex items-center px-3 py-1.5 bg-blue-900 text-white rounded text-sm hover:bg-blue-700 transition-colors">
                  <FaFileWord className="mr-1.5" /> DOCX
                </button>
                <button onClick={() => handleExport('TXT')} className="inline-flex items-center px-3 py-1.5 bg-blue-900 text-white rounded text-sm hover:bg-blue-700 transition-colors">
                  <FaFileAlt className="mr-1.5" /> TXT
                </button>
              </>
            )}
          </div>

          {/* Summary Sections */}
          <div className="flex-1 overflow-y-auto min-h-0">
            {isLoading ? (
              <SummarySkeletonLoader />
            ) : (
              <div className="space-y-2">
                {summaries.map((summary, index) => (
                  <div key={index}>
                    <button
                      onClick={() => toggleSection(index)}
                      className="w-full flex justify-between items-center p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                    >
                      <span className="flex items-center space-x-2">
                        {summary.type === 'summary' ? (
                          <FaSearch className="text-blue-900" />
                        ) : (
                          <FaCheck className="text-green-600" />
                        )}
                        <span className="text-sm font-medium text-gray-700">{summary.title}</span>
                      </span>
                      {expanded[index] ? (
                        <FaChevronUp className="text-gray-400" />
                      ) : (
                        <FaChevronDown className="text-gray-400" />
                      )}
                    </button>
                    {expanded[index] && (
                      <div className="mt-2 p-3 bg-white rounded-lg border border-gray-200">
                        <ul className="space-y-1.5">
                          {summary.points.map((point, i) => (
                            <li key={i} className="text-sm text-gray-600">{point}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SummarySection;