import React, { useState } from "react";

interface ResultItem {
  filename: string;
  mean_length: number;
  nagg_rate: number;
}

const CDTAnalyzer: React.FC = () => {
  const [ctrlFile, setCtrlFile] = useState<File | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [results, setResults] = useState<ResultItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleCtrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) setCtrlFile(e.target.files[0]);
  };

  const handleFilesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) setFiles(e.target.files);
  };

  const handleAnalyze = async () => {
    if (!ctrlFile || !files?.length) {
      alert("Please select files");
      return;
    }
    setIsLoading(true);
    
    // Mock API call - replace with actual implementation
    setTimeout(() => {
      const mockResults = [
        { filename: "sample1.csv", mean_length: 45.23, nagg_rate: 0.12 },
        { filename: "sample2.csv", mean_length: 52.67, nagg_rate: 0.08 },
        { filename: "sample3.csv", mean_length: 38.91, nagg_rate: 0.15 },
      ];
      setResults(mockResults);
      setIsLoading(false);
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        {/* Header */}
        <div className="mb-8">
          <nav className="text-sm text-gray-600 mb-4">
            <a href="/" className="hover:text-gray-900">Top</a>
            <span className="mx-2">/</span>
            <span className="text-gray-900">CDT</span>
          </nav>
          <h1 className="text-3xl font-bold text-gray-900">CDT Analyzer</h1>
        </div>

        {/* File Selection */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
          <div className="space-y-4 md:space-y-0 md:space-x-4 md:flex md:items-center">
            <div className="flex-1">
              <label className="block">
                <div className="bg-black text-white px-6 py-3 rounded-lg cursor-pointer hover:bg-gray-800 transition-colors text-center font-medium">
                  {ctrlFile ? ctrlFile.name : 'Select Control CSV'}
                </div>
                <input 
                  type="file" 
                  accept=".csv" 
                  className="hidden" 
                  onChange={handleCtrlChange} 
                />
              </label>
            </div>
            
            <div className="flex-1">
              <label className="block">
                <div className="bg-black text-white px-6 py-3 rounded-lg cursor-pointer hover:bg-gray-800 transition-colors text-center font-medium">
                  {files ? `${files.length} file(s) selected` : 'Select CSV files'}
                </div>
                <input 
                  type="file" 
                  accept=".csv" 
                  multiple 
                  className="hidden" 
                  onChange={handleFilesChange} 
                />
              </label>
            </div>
            
            <div className="flex-1">
              <button
                onClick={handleAnalyze}
                disabled={isLoading}
                className="w-full bg-gray-900 text-white px-6 py-3 rounded-lg font-medium hover:bg-black transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>
          </div>
        </div>

        {/* Loading Overlay */}
        {isLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 flex items-center space-x-3">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-gray-900"></div>
              <span className="text-gray-700">Processing files...</span>
            </div>
          </div>
        )}

        {/* Results Table */}
        {results.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200">
              <h2 className="text-lg font-semibold text-gray-900">Analysis Results</h2>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Filename
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Mean Length
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Nagg Rate
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {results.map((result, idx) => (
                    <tr key={idx} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {result.filename}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {result.mean_length.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {(result.nagg_rate * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CDTAnalyzer;