import React from 'react';
import { FaCopy } from 'react-icons/fa';

const TranslationFeatures = () => {
  const originalText = 'Hello, this is a meeting summary.';
  const translatedText = 'नमस्ते, ही एक बैठक सारांश आहे.';
  const [language, setLanguage] = React.useState('Marathi');

  const languages = [
    { name: 'Marathi', flag: '🇮🇳' },
    { name: 'Spanish', flag: '🇪🇸' },
    { name: 'French', flag: '🇫🇷' },
  ];

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  return (
    <div className="p-4 bg-white dark:bg-gray-800 shadow-md rounded-lg animate-componentFadeIn">
      {/* Language Options */}
      <div className="flex items-center mb-4">
        <label className="mr-2 text-gray-700 dark:text-gray-300">Select Language:</label>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          className="p-2 border border-blue-900 rounded-lg dark:bg-gray-700 dark:text-white"
        >
          {languages.map((lang) => (
            <option key={lang.name} value={lang.name}>
              {lang.flag} {lang.name}
            </option>
          ))}
        </select>
        <button className="ml-2 px-3 py-1 bg-blue-900 text-white rounded">Auto-Detect</button>
      </div>

      {/* Dual View */}
      <div className="flex space-x-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold mb-2">Original Text</h3>
          <div className="p-3 bg-gray-100 dark:bg-gray-700 rounded-lg relative">
            <p>{originalText}</p>
            <button
              onClick={() => copyToClipboard(originalText)}
              className="absolute top-2 right-2 text-blue-900 hover:text-blue-600"
            >
              <FaCopy />
            </button>
          </div>
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold mb-2">Translated Text</h3>
          <div className="p-3 bg-gray-100 dark:bg-gray-700 rounded-lg relative">
            <p>{translatedText}</p>
            <button
              onClick={() => copyToClipboard(translatedText)}
              className="absolute top-2 right-2 text-blue-900 hover:text-blue-600"
            >
              <FaCopy />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranslationFeatures;