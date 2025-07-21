import React, { useState } from "react";
import axios from "axios";
import { settings } from "../settings";
import {
  Upload, Settings, Image as ImageIcon, ChevronRight, Activity,
  Zap, Eye, BarChart3, Menu, X, Play, Download,
  Info, Clock, Cpu, LucideIcon
} from "lucide-react";

// 型定義
interface Parameter {
  name: string;
  label: string;
  type: 'number' | 'select' | 'boolean';
  min?: number;
  max?: number;
  default: number | boolean;
  step?: number;
  options?: number[];
  description: string;
}

interface Algorithm {
  id: string;
  name: string;
  category: string;
  description: string;
  parameters: Parameter[];
}

interface Result {
  id: string;
  algorithm: string;
  imageSrc: string;
  metadata: {
    parameters: Record<string, number | boolean>;
    processingTime: number;
    fileSize: number;
    dimensions: { width: number; height: number };
  };
  timestamp: Date;
}

// アルゴリズム定義
const ALGORITHMS: Record<string, Algorithm> = {
  canny: {
    id: 'canny',
    name: 'Canny Edge Detection',
    category: 'Edge Detection',
    description: 'Multi-stage algorithm for optimal edge detection with noise reduction.',
    parameters: [
      { name: 'threshold1', label: 'Lower Threshold', type: 'number', min: 0, max: 255, default: 100, description: 'Weak edge suppression threshold' },
      { name: 'threshold2', label: 'Upper Threshold', type: 'number', min: 0, max: 255, default: 200, description: 'Strong edge detection threshold' },
      { name: 'apertureSize', label: 'Aperture Size', type: 'select', options: [3, 5, 7], default: 3, description: 'Sobel kernel size' },
    ]
  },
  sobel: {
    id: 'sobel',
    name: 'Sobel Filter',
    category: 'Edge Detection',
    description: 'Gradient-based edge detection using Sobel operators.',
    parameters: [
      { name: 'ksize', label: 'Kernel Size', type: 'select', options: [3, 5, 7], default: 3, description: 'Size of the Sobel kernel' },
      { name: 'dx', label: 'X Derivative', type: 'number', min: 0, max: 2, default: 1, description: 'Order of X derivative' },
      { name: 'dy', label: 'Y Derivative', type: 'number', min: 0, max: 2, default: 1, description: 'Order of Y derivative' },
    ]
  },
  gaussian: {
    id: 'gaussian',
    name: 'Gaussian Blur',
    category: 'Noise Reduction',
    description: 'Smoothing filter using Gaussian kernel for noise reduction.',
    parameters: [
      { name: 'ksize', label: 'Kernel Size', type: 'number', min: 1, max: 31, default: 5, description: 'Size of the Gaussian kernel (odd numbers only)' },
      { name: 'sigmaX', label: 'Sigma X', type: 'number', min: 0.1, max: 10, default: 1.0, step: 0.1, description: 'Standard deviation in X direction' },
      { name: 'sigmaY', label: 'Sigma Y', type: 'number', min: 0.1, max: 10, default: 1.0, step: 0.1, description: 'Standard deviation in Y direction' },
    ]
  },
  histogram: {
    id: 'histogram',
    name: 'Histogram Analysis',
    category: 'Image Analysis',
    description: 'Analyze image histogram and compute statistical properties.',
    parameters: [
      { name: 'bins', label: 'Number of Bins', type: 'number', min: 8, max: 256, default: 256, description: 'Number of histogram bins' },
      { name: 'normalize', label: 'Normalize', type: 'boolean', default: false, description: 'Normalize histogram values' },
    ]
  },
  cell_contour: {
    id: 'cell_contour',
    name: 'Cell Contour Detection',
    category: 'Cell Analysis',
    description: 'Detect cell contours using thresholding similar to the Cell extraction pipeline.',
    parameters: [
      { name: 'threshold', label: 'Binary Threshold', type: 'number', min: 0, max: 255, default: 130, description: 'Threshold for binarization' },
      { name: 'min_area', label: 'Min Area', type: 'number', min: 10, max: 10000, default: 300, description: 'Minimum contour area' },
    ]
  }
};

// カテゴリー別のアイコン
const CATEGORY_ICONS: Record<string, LucideIcon> = {
  'Edge Detection': Zap,
  'Noise Reduction': Eye,
  'Image Analysis': BarChart3,
  'Cell Analysis': Cpu,
};

const url_prefix = settings.url_prefix;

const ImagePlayground: React.FC = () => {
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<keyof typeof ALGORITHMS>('canny');
  const [parameters, setParameters] = useState<Record<string, Record<string, number | boolean>>>(() => {
    const initialParams: Record<string, Record<string, number | boolean>> = {};
    Object.values(ALGORITHMS).forEach(alg => {
      initialParams[alg.id] = {};
      alg.parameters.forEach(param => {
        initialParams[alg.id][param.name] = param.default;
      });
    });
    return initialParams;
  });
  
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<Result[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const currentAlgorithm = ALGORITHMS[selectedAlgorithm];
  const CategoryIcon = CATEGORY_ICONS[currentAlgorithm.category] || Settings;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setFile(e.target.files[0]);
    }
  };

  const handleParameterChange = (paramName: string, value: number | boolean) => {
    setParameters(prev => ({
      ...prev,
      [selectedAlgorithm]: {
        ...prev[selectedAlgorithm],
        [paramName]: value
      }
    }));
  };

  const getImageDimensions = (imageUrl: string): Promise<{ width: number; height: number }> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve({ width: img.width, height: img.height });
      img.onerror = reject;
      img.src = imageUrl;
    });
  };

  const handleProcess = async () => {
    if (!file) return;
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const params = parameters[selectedAlgorithm];
      const query = new URLSearchParams();
      Object.entries(params).forEach(([key, value]) => query.append(key, String(value)));

      const start = performance.now();
      const response = await axios.post(
        `${url_prefix}/image_playground/${selectedAlgorithm}?${query.toString()}`,
        formData,
        { responseType: "blob" }
      );
      const processingTime = performance.now() - start;
      const blob = response.data as Blob;
      const imageUrl = URL.createObjectURL(blob);
      const dim = await getImageDimensions(imageUrl);

      const newResult: Result = {
        id: Date.now().toString(),
        algorithm: selectedAlgorithm,
        imageSrc: imageUrl,
        metadata: {
          parameters: params,
          processingTime,
          fileSize: blob.size,
          dimensions: dim
        },
        timestamp: new Date()
      };

      setResults(prev => [newResult, ...prev]);
    } catch (err) {
      console.error(err);
      alert("Failed to process image");
    } finally {
      setIsLoading(false);
    }
  };

  const renderParameterInput = (param: Parameter) => {
    const value = parameters[selectedAlgorithm][param.name];
    
    const inputStyle: React.CSSProperties = {
      width: '100%',
      backgroundColor: '#ffffff',
      border: '1px solid #d1d5db',
      borderRadius: '8px',
      padding: '8px 12px',
      color: '#374151',
      outline: 'none',
      fontSize: '14px'
    };
    
    const handleFocus = (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => {
      (e.target as HTMLElement).style.borderColor = '#3b82f6';
    };
    
    const handleBlur = (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => {
      (e.target as HTMLElement).style.borderColor = '#d1d5db';
    };
    
    switch (param.type) {
      case 'select':
        return (
          <select
            value={value as number}
            onChange={(e) => handleParameterChange(param.name, parseInt(e.target.value))}
            style={inputStyle}
            onFocus={handleFocus}
            onBlur={handleBlur}
          >
            {param.options?.map((option: number) => (
              <option key={option} value={option} style={{ backgroundColor: '#ffffff' }}>{option}</option>
            ))}
          </select>
        );
      
      case 'boolean':
        return (
          <input
            type="checkbox"
            checked={value as boolean}
            onChange={(e) => handleParameterChange(param.name, e.target.checked)}
            style={{
              width: '16px',
              height: '16px',
              accentColor: '#3b82f6',
              backgroundColor: '#ffffff',
              border: '1px solid #d1d5db',
              borderRadius: '4px'
            }}
          />
        );
      
      default:
        return (
          <input
            type="number"
            value={value as number}
            onChange={(e) => handleParameterChange(param.name, param.step ? parseFloat(e.target.value) : parseInt(e.target.value))}
            min={param.min}
            max={param.max}
            step={param.step || 1}
            style={inputStyle}
            onFocus={handleFocus}
            onBlur={handleBlur}
          />
        );
    }
  };

  const cardStyle: React.CSSProperties = {
    background: '#ffffff',
    border: '1px solid #e5e7eb',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 10px 20px rgba(0, 0, 0, 0.1)'
  };

  const buttonStyle: React.CSSProperties = {
    width: '100%',
    padding: '16px 24px',
    borderRadius: '12px',
    fontSize: '18px',
    fontWeight: '500',
    transition: 'all 0.3s ease',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    border: 'none',
    cursor: 'pointer'
  };

  return (
    <div
      style={{
        display: 'flex',
        width: '100%',
        fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif'
      }}
    >
      {/* Sidebar */}
      <div style={{
        width: sidebarOpen ? '320px' : '0',
        transition: 'all 0.3s ease',
        overflow: 'hidden'
      }}>
        <div style={{
          height: '100%',
          backgroundColor: 'rgba(0, 0, 0, 0.05)',
          borderRight: '1px solid #e5e7eb',
          padding: '16px'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '24px'
          }}>
            <h3 style={{
              fontSize: '18px',
              fontWeight: '500',
              color: '#374151',
              margin: 0
            }}>Algorithms</h3>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              style={{
                padding: '4px',
                color: '#9ca3af',
                backgroundColor: 'transparent',
                border: 'none',
                cursor: 'pointer'
              }}
              onMouseEnter={(e) => (e.target as HTMLElement).style.color = '#d1d5db'}
              onMouseLeave={(e) => (e.target as HTMLElement).style.color = '#9ca3af'}
            >
              <X size={20} />
            </button>
          </div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {Object.entries(
              Object.values(ALGORITHMS).reduce((acc: Record<string, Algorithm[]>, alg) => {
                if (!acc[alg.category]) acc[alg.category] = [];
                acc[alg.category].push(alg);
                return acc;
              }, {})
            ).map(([category, algorithms]) => (
              <div key={category}>
                <h4 style={{
                  fontSize: '14px',
                  fontWeight: '500',
                  color: '#9ca3af',
                  marginBottom: '8px',
                  margin: 0
                }}>{category}</h4>
                {algorithms.map(alg => {
                  const Icon = CATEGORY_ICONS[alg.category];
                  const isSelected = selectedAlgorithm === alg.id;
                  return (
                    <button
                      key={alg.id}
                      onClick={() => setSelectedAlgorithm(alg.id as keyof typeof ALGORITHMS)}
                      style={{
                        width: '100%',
                        textAlign: 'left',
                        padding: '12px',
                        borderRadius: '8px',
                        transition: 'all 0.3s ease',
                        backgroundColor: isSelected ? 'rgba(59, 130, 246, 0.2)' : '#ffffff',
                        border: isSelected ? '1px solid rgba(59, 130, 246, 0.5)' : '1px solid #e5e7eb',
                        color: isSelected ? '#2563eb' : '#374151',
                        cursor: 'pointer'
                      }}
                      onMouseEnter={(e) => {
                        if (!isSelected) {
                          (e.target as HTMLElement).style.backgroundColor = '#f3f4f6';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isSelected) {
                          (e.target as HTMLElement).style.backgroundColor = '#ffffff';
                        }
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <Icon size={16} style={{ marginRight: '8px' }} />
                        <span style={{ fontSize: '14px', fontWeight: '500' }}>{alg.name}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, overflowY: 'auto' }}>
        <div style={{
          maxWidth: '1200px',
          margin: '0 auto',
          padding: '32px 24px'
        }}>
          {/* Header */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: '32px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              {!sidebarOpen && (
                <button
                  onClick={() => setSidebarOpen(true)}
                  style={{
                    padding: '8px',
                    color: '#9ca3af',
                    backgroundColor: 'transparent',
                    border: 'none',
                    cursor: 'pointer',
                    marginRight: '16px'
                  }}
                  onMouseEnter={(e) => (e.target as HTMLElement).style.color = '#d1d5db'}
                  onMouseLeave={(e) => (e.target as HTMLElement).style.color = '#9ca3af'}
                >
                  <Menu size={24} />
                </button>
              )}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                  <Activity size={32} style={{ color: '#60a5fa', marginRight: '12px' }} />
                  <h1 style={{
                    fontSize: '32px',
                    fontWeight: '300',
                    color: '#374151',
                    letterSpacing: '0.5px',
                    margin: 0
                  }}>
                    Computer Vision PlayGround
                  </h1>
                </div>
                <p style={{
                  color: '#9ca3af',
                  fontSize: '18px',
                  fontWeight: '300',
                  margin: 0
                }}>
                  Advanced Image Processing & Analysis Platform
                </p>
              </div>
            </div>
          </div>

          {/* Breadcrumbs */}
          <nav style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            fontSize: '14px',
            marginBottom: '32px'
          }}>
            <a href="/" style={{
              color: '#4ade80',
              textDecoration: 'none',
              transition: 'color 0.3s ease'
            }}
            onMouseEnter={(e) => (e.target as HTMLElement).style.color = '#22c55e'}
            onMouseLeave={(e) => (e.target as HTMLElement).style.color = '#4ade80'}
            >
              Research Home
            </a>
            <ChevronRight size={16} style={{ color: '#6b7280' }} />
            <span style={{ color: '#9ca3af' }}>{currentAlgorithm.category}</span>
            <ChevronRight size={16} style={{ color: '#6b7280' }} />
            <span style={{ color: '#374151' }}>{currentAlgorithm.name}</span>
          </nav>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))',
            gap: '24px'
          }}>
            {/* Configuration Panel */}
            <div style={{ gridColumn: 'span 2', display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Algorithm Info */}
              <div style={cardStyle}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  marginBottom: '16px'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <CategoryIcon size={20} style={{ color: '#60a5fa', marginRight: '8px' }} />
                    <h2 style={{
                      fontSize: '20px',
                      fontWeight: '500',
                      color: '#374151',
                      margin: 0
                    }}>{currentAlgorithm.name}</h2>
                  </div>
                  <span style={{
                    backgroundColor: '#f3f4f6',
                    color: '#374151',
                    padding: '6px 12px',
                    borderRadius: '20px',
                    fontSize: '14px'
                  }}>
                    {currentAlgorithm.category}
                  </span>
                </div>
                <p style={{
                  color: '#9ca3af',
                  lineHeight: '1.6',
                  margin: 0
                }}>
                  {currentAlgorithm.description}
                </p>
              </div>

              {/* File Upload */}
              <div style={cardStyle}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '16px'
                }}>
                  <Upload size={20} style={{ color: '#60a5fa', marginRight: '8px' }} />
                  <h2 style={{
                    fontSize: '20px',
                    fontWeight: '500',
                    color: '#374151',
                    margin: 0
                  }}>Input Image</h2>
                </div>
                
                <div style={{
                  border: '2px dashed #d1d5db',
                  borderRadius: '8px',
                  padding: '32px',
                  textAlign: 'center',
                  backgroundColor: '#f9fafb'
                }}>
                  <input 
                    type="file" 
                    accept="image/*" 
                    onChange={handleFileChange}
                    style={{
                      display: 'block',
                      width: '100%',
                      color: '#374151',
                      backgroundColor: '#ffffff',
                      border: '1px solid #d1d5db',
                      borderRadius: '8px',
                      padding: '12px'
                    }}
                  />
                  {file && (
                    <div style={{
                      marginTop: '12px',
                      color: '#4ade80',
                      fontSize: '14px'
                    }}>
                      <p style={{ fontWeight: '500', margin: '0 0 4px 0' }}>✓ Selected: {file.name}</p>
                      <p style={{ color: '#6b7280', margin: 0 }}>Size: {(file.size / 1024).toFixed(1)} KB</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Parameters */}
              <div style={cardStyle}>
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  marginBottom: '24px'
                }}>
                  <Settings size={20} style={{ color: '#60a5fa', marginRight: '8px' }} />
                  <h2 style={{
                    fontSize: '20px',
                    fontWeight: '500',
                    color: '#374151',
                    margin: 0
                  }}>Parameters</h2>
                </div>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {currentAlgorithm.parameters.map((param: Parameter) => (
                    <div key={param.name}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '8px'
                      }}>
                        <label style={{
                          color: '#374151',
                          fontWeight: '500',
                          fontSize: '14px'
                        }}>{param.label}</label>
                        <div style={{
                          position: 'relative',
                          display: 'inline-block'
                        }} title={param.description}>
                          <Info 
                            size={16} 
                            style={{ 
                              color: '#6b7280', 
                              cursor: 'help' 
                            }}
                          />
                        </div>
                      </div>
                      {renderParameterInput(param)}
                      {param.min !== undefined && param.max !== undefined && (
                        <p style={{
                          color: '#6b7280',
                          fontSize: '12px',
                          marginTop: '4px',
                          margin: '4px 0 0 0'
                        }}>
                          Range: {param.min} - {param.max}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Process Button */}
              <button 
                onClick={handleProcess} 
                disabled={isLoading || !file}
                style={{
                  ...buttonStyle,
                  backgroundColor: isLoading || !file ? '#d1d5db' : '#3b82f6',
                  color: isLoading || !file ? '#9ca3af' : '#ffffff',
                  cursor: isLoading || !file ? 'not-allowed' : 'pointer',
                  background: isLoading || !file ? '#d1d5db' : 'linear-gradient(45deg, #3b82f6 30%, #2563eb 90%)'
                }}
                onMouseEnter={(e) => {
                  if (!isLoading && file) {
                    const target = e.target as HTMLElement;
                    target.style.background = 'linear-gradient(45deg, #2563eb 30%, #1d4ed8 90%)';
                    target.style.transform = 'translateY(-2px)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isLoading && file) {
                    const target = e.target as HTMLElement;
                    target.style.background = 'linear-gradient(45deg, #3b82f6 30%, #2563eb 90%)';
                    target.style.transform = 'translateY(0)';
                  }
                }}
              >
                {isLoading ? (
                  <>
                    <div style={{
                      width: '20px',
                      height: '20px',
                      border: '2px solid transparent',
                      borderTop: '2px solid #ffffff',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite',
                      marginRight: '8px'
                    }}></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <Play size={20} style={{ marginRight: '8px' }} />
                    Apply {currentAlgorithm.name}
                  </>
                )}
              </button>
            </div>

            {/* Results Panel */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <h3 style={{
                fontSize: '18px',
                fontWeight: '500',
                color: '#f3f4f6',
                display: 'flex',
                alignItems: 'center',
                margin: 0
              }}>
                <BarChart3 size={20} style={{ marginRight: '8px', color: '#60a5fa' }} />
                Results ({results.length})
              </h3>
              
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '16px',
                maxHeight: '800px',
                overflowY: 'auto'
              }}>
                {results.map(result => (
                  <div key={result.id} style={{
                    ...cardStyle,
                    padding: '16px'
                  }}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      marginBottom: '12px'
                    }}>
                      <span style={{
                        fontSize: '14px',
                        fontWeight: '500',
                        color: '#374151'
                      }}>
                        {ALGORITHMS[result.algorithm].name}
                      </span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <button style={{
                          padding: '4px',
                          color: '#9ca3af',
                          backgroundColor: 'transparent',
                          border: 'none',
                          cursor: 'pointer'
                        }}
                        onMouseEnter={(e) => (e.target as HTMLElement).style.color = '#d1d5db'}
                        onMouseLeave={(e) => (e.target as HTMLElement).style.color = '#9ca3af'}
                        >
                          <Download size={16} />
                        </button>
                      </div>
                    </div>
                    
                    <div style={{
                      borderRadius: '8px',
                      overflow: 'hidden',
                      border: '1px solid #d1d5db',
                      marginBottom: '12px'
                    }}>
                      <img 
                        src={result.imageSrc} 
                        alt="Result" 
                        style={{ width: '100%', height: 'auto', display: 'block' }}
                      />
                    </div>
                    
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '8px',
                      fontSize: '12px',
                      color: '#9ca3af'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <Clock size={12} style={{ marginRight: '4px' }} />
                        {result.metadata.processingTime.toFixed(0)}ms
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center' }}>
                        <Cpu size={12} style={{ marginRight: '4px' }} />
                        {result.metadata.dimensions.width}×{result.metadata.dimensions.height}
                      </div>
                      <div style={{ color: '#6b7280' }}>
                        {result.timestamp.toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
                
                {results.length === 0 && (
                  <div style={{
                    textAlign: 'center',
                    padding: '32px',
                    color: '#6b7280'
                  }}>
                    <ImageIcon size={48} style={{
                      margin: '0 auto 12px auto', 
                      opacity: 0.5,
                      display: 'block'
                    }} />
                    <p style={{ margin: '0 0 4px 0' }}>No results yet</p>
                    <p style={{ fontSize: '14px', margin: 0 }}>Process an image to see results here</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Footer */}
          <div style={{
            marginTop: '48px',
            paddingTop: '24px',
            borderTop: '1px solid #e5e7eb'
          }}>
            <p style={{
              textAlign: 'center',
              color: '#6b7280',
              fontSize: '14px',
              margin: 0
            }}>
              Computer Vision Research Laboratory © 2025
            </p>
          </div>
        </div>
      </div>

      {/* CSS Animation for loading spinner */}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default ImagePlayground;