import React, { useState } from 'react';

const Upload = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (file) {
      alert(`File ${file.name} uploaded successfully!`);
    } else {
      alert('Please select a file to upload.');
    }
  };

  return (
    <div className="container py-6 space-y-6 animate-componentFadeIn">
      <h2 className="text-3xl font-bold text-blue-900 tracking-tight">Upload</h2>
      <div className="card">
        <input
          type="file"
          onChange={handleFileChange}
          className="mb-4 p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white"
        />
        <button
          onClick={handleUpload}
          className="px-4 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200"
        >
          Upload File
        </button>
        {file && (
          <p className="mt-4 text-gray-700 dark:text-gray-300">
            Selected file: {file.name}
          </p>
        )}
      </div>
    </div>
  );
};

export default Upload;