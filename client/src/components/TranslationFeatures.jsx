import React, { useState } from 'react';
import { FaCopy } from 'react-icons/fa';

const TranslationFeatures = () => {
  const originalText = 'Hello, this is a meeting summary.';
  const translations = {
    English: 'Hello, this is a meeting summary.',
    Japanese: 'こんにちは、これは会議の要約です。',
    Hindi: 'नमस्ते, यह एक बैठक सारांश है।',
    Korean: '안녕하세요, 이것은 회의 요약입니다.',
    Spanish: 'Hola, este es un resumen de la reunión.',
    French: 'Bonjour, ceci est un résumé de la réunion.',
  };
  const [language, setLanguage] = useState('English');

  const languages = [
    { name: 'English', flag: '🇬🇧' },
    { name: 'Japanese', flag: '🇯🇵' },
    { name: 'Hindi', flag: '🇮🇳' },
    { name: 'Korean', flag: '🇰🇷' },
    { name: 'Spanish', flag: '🇪🇸' },
    { name: 'French', flag: '🇫🇷' },
  ];

  const translatedText = translations[language];

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Translation</h3>
      </div>
      <div className="flex-1 p-4 overflow-hidden">
        <div className="flex flex-col h-full">
          {/* Language Options */}
          <div className="flex items-center space-x-3 mb-4">
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="flex-1 p-2 border border-gray-200 rounded-lg bg-white text-gray-700 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {languages.map((lang) => (
                <option key={lang.name} value={lang.name}>
                  {lang.flag} {lang.name}
                </option>
              ))}
            </select>
            <button className="px-4 py-2 bg-blue-900 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200 text-sm">
              Auto-Detect
            </button>
          </div>

          {/* Dual View */}
          <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-4 min-h-0">
            <div className="flex flex-col">
              <h4 className="text-sm font-medium text-gray-600 mb-2">Original Text</h4>
              <div className="flex-1 p-3 bg-gray-50 rounded-lg relative">
                <p className="text-sm text-gray-700">{originalText}</p>
                <button
                  onClick={() => copyToClipboard(originalText)}
                  className="absolute top-2 right-2 text-gray-400 hover:text-gray-600 transition-colors duration-200"
                >
                  <FaCopy className="h-4 w-4" />
                </button>
              </div>
            </div>
            <div className="flex flex-col">
              <h4 className="text-sm font-medium text-gray-600 mb-2">Translated Text</h4>
              <div className="flex-1 p-3 bg-gray-50 rounded-lg relative">
                <p className="text-sm text-gray-700">{translatedText}</p>
                <button
                  onClick={() => copyToClipboard(translatedText)}
                  className="absolute top-2 right-2 text-gray-400 hover:text-gray-600 transition-colors duration-200"
                >
                  <FaCopy className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TranslationFeatures;