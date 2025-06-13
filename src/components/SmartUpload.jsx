import React from 'react';
import { FaUpload } from 'react-icons/fa';

const SmartUpload = () => {
  return (
    <div className="p-4 space-y-4 animate-componentFadeIn">
      <h2 className="text-2xl font-bold text-blue-900">Smart Upload</h2>
      <div className="bg-white dark:bg-gray-800 shadow-md rounded-lg p-4">
        <div className="border-2 border-dashed border-blue-900 rounded-lg p-6 text-center">
          <FaUpload className="mx-auto text-blue-900 text-4xl mb-2" />
          <p className="text-gray-600 dark:text-gray-300">Drag and drop your audio files here, or</p>
          <button className="mt-2 px-4 py-2 bg-blue-900 text-white rounded">Browse Files</button>
        </div>
      </div>
    </div>
  );
};

export default SmartUpload;