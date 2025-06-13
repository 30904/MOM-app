import React, { useState, useCallback, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import { FaCloudUploadAlt, FaFile, FaSpinner } from 'react-icons/fa';

const Upload = () => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles?.length > 0) {
      setFile(acceptedFiles[0]);
    }
  }, []);

  const dropzoneOptions = useMemo(() => ({
    onDrop,
    accept: {
      'audio/*': ['.mp3', '.wav', '.m4a'],
      'video/*': ['.mp4', '.mov', '.avi'],
    },
    maxFiles: 1,
  }), [onDrop]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone(dropzoneOptions);

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(0);

    // Simulate upload progress
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          return 100;
        }
        return prev + 10;
      });
    }, 500);

    // Simulate API call
    try {
      // TODO: Replace with actual API call
      await new Promise((resolve) => setTimeout(resolve, 5000));
      alert(`File ${file.name} uploaded successfully!`);
    } catch (error) {
      alert('Upload failed. Please try again.');
    } finally {
      clearInterval(interval);
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-2 border-b border-gray-200">
        <h3 className="text-sm font-medium text-gray-700">Upload Recording</h3>
      </div>

      <div className="flex-1 p-4">
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors duration-200 ${
            isDragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }`}
        >
          <input {...getInputProps()} />
          <FaCloudUploadAlt className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-600">
            {isDragActive
              ? 'Drop the file here'
              : 'Drag and drop a file here, or click to select'}
          </p>
          <p className="mt-1 text-xs text-gray-500">
            Supported formats: MP3, WAV, M4A, MP4, MOV, AVI
          </p>
        </div>

        {file && (
          <div className="mt-4 p-3 bg-white rounded-lg border border-gray-200">
            <div className="flex items-center space-x-3">
              <FaFile className="h-8 w-8 text-blue-500" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {file.name}
                </p>
                <p className="text-xs text-gray-500">
                  {formatFileSize(file.size)}
                </p>
              </div>
              <button
                onClick={() => setFile(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>

            {isUploading && (
              <div className="mt-3">
                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div
                    className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
                <p className="mt-1 text-xs text-gray-500 text-right">
                  {uploadProgress}%
                </p>
              </div>
            )}
          </div>
        )}

        <div className="mt-4 flex justify-end">
          <button
            onClick={handleUpload}
            disabled={!file || isUploading}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center space-x-2 ${
              !file || isUploading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-900 text-white hover:bg-blue-700'
            }`}
          >
            {isUploading ? (
              <>
                <FaSpinner className="animate-spin" />
                <span>Uploading...</span>
              </>
            ) : (
              <span>Upload File</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Upload;