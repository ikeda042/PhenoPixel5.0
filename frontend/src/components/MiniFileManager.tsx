import { useEffect, useState, type DragEvent } from 'react';
import axios from 'axios';
import { settings } from '../settings';

// API設定
const API_URL = settings.url_prefix;

// SVGアイコンコンポーネント
const icons = {
  Upload: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M14.5 3h-5v7l-3-3m0 6h12l-6-6h-3z"/>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 5 17 10"/>
      <line x1="12" y1="5" x2="12" y2="15"/>
    </svg>
  ),
  Download: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
      <polyline points="7 10 12 15 17 10"/>
      <line x1="12" y1="15" x2="12" y2="3"/>
    </svg>
  ),
  Trash2: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <polyline points="3 6 5 6 21 6"/>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
      <line x1="10" y1="11" x2="10" y2="17"/>
      <line x1="14" y1="11" x2="14" y2="17"/>
    </svg>
  ),
  FileText: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="16" y1="13" x2="8" y2="13"/>
      <line x1="16" y1="17" x2="8" y2="17"/>
      <polyline points="10 9 9 9 8 9"/>
    </svg>
  ),
  Image: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
      <circle cx="8.5" cy="8.5" r="1.5"/>
      <polyline points="21 15 16 10 5 21"/>
    </svg>
  ),
  Video: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <polygon points="23 7 16 12 23 17 23 7"/>
      <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
    </svg>
  ),
  Music: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M9 18V5l12-2v13"/>
      <circle cx="6" cy="18" r="3"/>
      <circle cx="18" cy="16" r="3"/>
    </svg>
  ),
  Archive: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <polyline points="21 8 21 21 3 21 3 8"/>
      <rect x="1" y="3" width="22" height="5"/>
      <line x1="10" y1="12" x2="14" y2="12"/>
    </svg>
  ),
  File: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
  ),
  Grid3x3: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <rect x="3" y="3" width="7" height="7"/>
      <rect x="14" y="3" width="7" height="7"/>
      <rect x="14" y="14" width="7" height="7"/>
      <rect x="3" y="14" width="7" height="7"/>
    </svg>
  ),
  List: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <line x1="8" y1="6" x2="21" y2="6"/>
      <line x1="8" y1="12" x2="21" y2="12"/>
      <line x1="8" y1="18" x2="21" y2="18"/>
      <line x1="3" y1="6" x2="3.01" y2="6"/>
      <line x1="3" y1="12" x2="3.01" y2="12"/>
      <line x1="3" y1="18" x2="3.01" y2="18"/>
    </svg>
  ),
  Search: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <circle cx="11" cy="11" r="8"/>
      <path d="m21 21-4.35-4.35"/>
    </svg>
  ),
  Plus: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <line x1="12" y1="5" x2="12" y2="19"/>
      <line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  ),
  Folder: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2z"/>
    </svg>
  ),
  HardDrive: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <line x1="22" y1="12" x2="2" y2="12"/>
      <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>
      <line x1="6" y1="16" x2="6.01" y2="16"/>
      <line x1="10" y1="16" x2="10.01" y2="16"/>
    </svg>
  ),
  FileSpreadsheet: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <path d="M8 13h8M8 17h8M8 9h2"/>
    </svg>
  ),
  Presentation: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M2 3h20v14H2z"/>
      <path d="M8 21h8"/>
      <path d="M12 17v4"/>
      <path d="m7 8 3 3-3 3"/>
      <path d="M13 8h3"/>
    </svg>
  ),
  FileCode: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <polyline points="10 13 8 15 10 17"/>
      <polyline points="14 13 16 15 14 17"/>
    </svg>
  ),
  Settings: ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={style}>
      <circle cx="12" cy="12" r="3"/>
      <path d="M12 1v6m0 6v6m11-7h-6m-6 0H1m11-7h6m-6 0H1"/>
    </svg>
  ),
};

// Utility
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

// API 関数
interface RawFileInfo {
  name: string;
  size: number;
  modified: string;
}

async function listFiles(): Promise<FileInfo[]> {
  const response = await axios.get<RawFileInfo[]>(`${API_URL}/files`);
  return response.data.map((f) => ({
    name: f.name,
    type: f.name.split('.').pop(),
    size: formatBytes(f.size),
    modified: new Date(f.modified).toLocaleDateString('ja-JP'),
  }));
}

async function uploadFile(file: File): Promise<void> {
  const formData = new FormData();
  formData.append('file', file);
  await axios.post(`${API_URL}/upload`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
}

async function downloadFile(filename: string) {
  const response = await axios.get(`${API_URL}/download/${encodeURIComponent(filename)}`, {
    responseType: 'blob'
  });
  const url = window.URL.createObjectURL(new Blob([response.data]));
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
}

async function deleteFile(filename: string): Promise<void> {
  await axios.delete(`${API_URL}/delete/${encodeURIComponent(filename)}`);
}

interface FileInfo {
  name: string;
  size: string;
  modified: string;
  type?: string;
}

function useResponsive() {
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 1200,
    height: typeof window !== 'undefined' ? window.innerHeight : 800,
  });

  useEffect(() => {
    function handleResize() {
      setWindowSize({ width: window.innerWidth, height: window.innerHeight });
    }
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const isMobile = windowSize.width < 768;
  const isTablet = windowSize.width < 1024 && windowSize.width >= 768;
  return { windowSize, isMobile, isTablet };
}

function MiniFileManager() {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [selected, setSelected] = useState<File | null>(null);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchTerm, setSearchTerm] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [showMobileSearch, setShowMobileSearch] = useState(false);
  const { isMobile, isTablet } = useResponsive();

  const refresh = async () => {
    try {
      const infos = await listFiles();
      setFiles(infos);
    } catch (error) {
      console.error('Failed to fetch files:', error);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (isMobile && viewMode === 'grid') {
      setViewMode('list');
    }
  }, [isMobile, viewMode]);

  const handleUpload = async () => {
    if (selected) {
      setUploading(true);
      try {
        await uploadFile(selected);
        setSelected(null);
        await refresh();
      } catch (error) {
        console.error('Upload failed:', error);
      } finally {
        setUploading(false);
      }
    }
  };

  const handleDelete = async (name: string) => {
    try {
      await deleteFile(name);
      await refresh();
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    if (droppedFiles.length > 0) {
      setSelected(droppedFiles[0]);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const getFileIcon = (type: string) => {
    const iconSize = isMobile ? '20px' : '24px';
    const iconStyle = { width: iconSize, height: iconSize };
    
    switch (type?.toLowerCase()) {
      case 'pdf':
        return <icons.FileText style={{ ...iconStyle, color: '#dc2626' }} />;
      case 'doc':
      case 'docx':
        return <icons.FileText style={{ ...iconStyle, color: '#2563eb' }} />;
      case 'txt':
      case 'rtf':
        return <icons.FileText style={{ ...iconStyle, color: '#6b7280' }} />;
      case 'xls':
      case 'xlsx':
      case 'csv':
        return <icons.FileSpreadsheet style={{ ...iconStyle, color: '#059669' }} />;
      case 'ppt':
      case 'pptx':
        return <icons.Presentation style={{ ...iconStyle, color: '#dc2626' }} />;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
      case 'svg':
      case 'webp':
      case 'bmp':
      case 'tiff':
        return <icons.Image style={{ ...iconStyle, color: '#0061ff' }} />;
      case 'mp4':
      case 'avi':
      case 'mov':
      case 'mkv':
      case 'wmv':
      case 'flv':
      case 'webm':
        return <icons.Video style={{ ...iconStyle, color: '#7c3aed' }} />;
      case 'mp3':
      case 'wav':
      case 'flac':
      case 'aac':
      case 'ogg':
      case 'm4a':
        return <icons.Music style={{ ...iconStyle, color: '#059669' }} />;
      case 'zip':
      case 'rar':
      case '7z':
      case 'tar':
      case 'gz':
      case 'bz2':
        return <icons.Archive style={{ ...iconStyle, color: '#d97706' }} />;
      case 'js':
      case 'ts':
      case 'jsx':
      case 'tsx':
      case 'html':
      case 'css':
      case 'py':
      case 'java':
      case 'cpp':
      case 'c':
      case 'php':
      case 'rb':
      case 'go':
      case 'rs':
      case 'swift':
      case 'kt':
        return <icons.FileCode style={{ ...iconStyle, color: '#f59e0b' }} />;
      case 'json':
      case 'xml':
      case 'yaml':
      case 'yml':
      case 'toml':
      case 'ini':
      case 'cfg':
        return <icons.Settings style={{ ...iconStyle, color: '#8b5cf6' }} />;
      default:
        return <icons.File style={{ ...iconStyle, color: '#6b7280' }} />;
    }
  };

  const filteredFiles = files.filter((file) =>
    file.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getGridColumns = () => {
    if (isMobile) return 'repeat(auto-fill, minmax(150px, 1fr))';
    if (isTablet) return 'repeat(auto-fill, minmax(180px, 1fr))';
    return 'repeat(auto-fill, minmax(220px, 1fr))';
  };

  const DROPBOX_BLUE = '#0061ff';
  const BORDER_COLOR = '#e5e5e5';
  const TEXT_PRIMARY = '#1e1919';
  const TEXT_SECONDARY = '#637282';
  const BACKGROUND_GRAY = '#f7f9fa';

  const headerStyle: React.CSSProperties = {
    backgroundColor: '#ffffff',
    borderBottom: `1px solid ${BORDER_COLOR}`,
    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.04)',
  };

  const containerStyle: React.CSSProperties = {
    minHeight: '100vh',
    backgroundColor: BACKGROUND_GRAY,
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif',
  };

  const uploadAreaStyle: React.CSSProperties = {
    border: `2px dashed ${isDragging ? DROPBOX_BLUE : '#c6c6c6'}`,
    borderRadius: '8px',
    padding: isMobile ? '24px 16px' : '40px 20px',
    textAlign: 'center',
    backgroundColor: isDragging ? '#f0f7ff' : '#ffffff',
    transition: 'all 0.2s ease',
    marginBottom: '24px',
  };

  const cardStyle: React.CSSProperties = {
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    border: `1px solid ${BORDER_COLOR}`,
    transition: 'all 0.2s ease',
    overflow: 'hidden',
  };

  const buttonStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    padding: isMobile ? '12px 16px' : '10px 16px',
    backgroundColor: DROPBOX_BLUE,
    color: '#ffffff',
    borderRadius: '6px',
    border: 'none',
    cursor: 'pointer',
    fontSize: '14px',
    fontWeight: '500',
    transition: 'all 0.2s ease',
  };

  const iconButtonStyle: React.CSSProperties = {
    padding: isMobile ? '8px' : '6px',
    backgroundColor: 'transparent',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    color: TEXT_SECONDARY,
    transition: 'all 0.2s ease',
  };

  const searchInputStyle: React.CSSProperties = {
    paddingLeft: '36px',
    paddingRight: '12px',
    paddingTop: '8px',
    paddingBottom: '8px',
    border: `1px solid ${BORDER_COLOR}`,
    borderRadius: '6px',
    backgroundColor: '#ffffff',
    color: TEXT_PRIMARY,
    fontSize: '14px',
    outline: 'none',
    width: isMobile ? '100%' : '280px',
  };

  return (
    <div style={containerStyle}>
      <div style={headerStyle}>
        <div style={{ width: '100%', padding: isMobile ? '0 16px' : '0 24px' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              height: isMobile ? '56px' : '60px',
              flexWrap: isMobile ? 'wrap' : 'nowrap',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <div
                style={{
                  width: isMobile ? '28px' : '32px',
                  height: isMobile ? '28px' : '32px',
                  backgroundColor: DROPBOX_BLUE,
                  borderRadius: '4px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <icons.HardDrive
                  style={{
                    width: isMobile ? '16px' : '18px',
                    height: isMobile ? '16px' : '18px',
                    color: '#ffffff',
                  }}
                />
              </div>
              <h1
                style={{
                  fontSize: isMobile ? '18px' : '20px',
                  fontWeight: '500',
                  color: TEXT_PRIMARY,
                  margin: 0,
                }}
              >
                ファイル
              </h1>
            </div>
            {!isMobile && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
                <div style={{ position: 'relative' }}>
                  <icons.Search
                    style={{
                      position: 'absolute',
                      left: '12px',
                      top: '50%',
                      transform: 'translateY(-50%)',
                      color: TEXT_SECONDARY,
                      width: '16px',
                      height: '16px',
                    }}
                  />
                  <input
                    type="text"
                    placeholder="ファイルを検索"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    style={searchInputStyle}
                  />
                </div>
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    backgroundColor: BACKGROUND_GRAY,
                    borderRadius: '6px',
                    padding: '2px',
                    border: `1px solid ${BORDER_COLOR}`,
                  }}
                >
                  <button
                    onClick={() => setViewMode('grid')}
                    style={{
                      padding: '6px',
                      borderRadius: '4px',
                      border: 'none',
                      backgroundColor: viewMode === 'grid' ? '#ffffff' : 'transparent',
                      color: TEXT_SECONDARY,
                      cursor: 'pointer',
                      boxShadow:
                        viewMode === 'grid' ? '0 1px 2px rgba(0, 0, 0, 0.1)' : 'none',
                    }}
                  >
                    <icons.Grid3x3 style={{ width: '16px', height: '16px' }} />
                  </button>
                  <button
                    onClick={() => setViewMode('list')}
                    style={{
                      padding: '6px',
                      borderRadius: '4px',
                      border: 'none',
                      backgroundColor: viewMode === 'list' ? '#ffffff' : 'transparent',
                      color: TEXT_SECONDARY,
                      cursor: 'pointer',
                      boxShadow:
                        viewMode === 'list' ? '0 1px 2px rgba(0, 0, 0, 0.1)' : 'none',
                    }}
                  >
                    <icons.List style={{ width: '16px', height: '16px' }} />
                  </button>
                </div>
              </div>
            )}
            {isMobile && (
              <button
                onClick={() => setShowMobileSearch(!showMobileSearch)}
                style={{
                  padding: '8px',
                  backgroundColor: 'transparent',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  color: TEXT_SECONDARY,
                }}
              >
                <icons.Search style={{ width: '20px', height: '20px' }} />
              </button>
            )}
          </div>
          {isMobile && showMobileSearch && (
            <div style={{ paddingBottom: '16px', position: 'relative' }}>
              <icons.Search
                style={{
                  position: 'absolute',
                  left: '12px',
                  top: '50%',
                  transform: 'translateY(-50%)',
                  color: TEXT_SECONDARY,
                  width: '16px',
                  height: '16px',
                }}
              />
              <input
                type="text"
                placeholder="ファイルを検索"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                style={searchInputStyle}
              />
            </div>
          )}
        </div>
      </div>
      <div style={{ width: '100%', padding: isMobile ? '16px' : '24px' }}>
        <div
          style={uploadAreaStyle}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
            <icons.Upload
              style={{
                width: isMobile ? '32px' : '40px',
                height: isMobile ? '32px' : '40px',
                color: TEXT_SECONDARY,
              }}
            />
            <div>
              <p
                style={{
                  fontSize: isMobile ? '14px' : '16px',
                  fontWeight: '400',
                  color: TEXT_PRIMARY,
                  margin: '0 0 8px 0',
                  textAlign: 'center',
                }}
              >
                ファイルをドラッグ&ドロップするか
              </p>
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <label style={{ cursor: 'pointer' }}>
                  <span
                    style={buttonStyle}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#0052cc';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = DROPBOX_BLUE;
                    }}
                  >
                    <icons.Plus style={{ width: '16px', height: '16px', marginRight: '6px' }} />
                    ファイルを選択
                  </span>
                  <input
                    type="file"
                    style={{ display: 'none' }}
                    onChange={(e) => setSelected(e.target.files ? e.target.files[0] : null)}
                  />
                </label>
              </div>
            </div>
            {selected && (
              <div
                style={{
                  backgroundColor: BACKGROUND_GRAY,
                  borderRadius: '6px',
                  padding: '16px',
                  maxWidth: '400px',
                  width: '100%',
                  border: `1px solid ${BORDER_COLOR}`,
                }}
              >
                <p style={{ fontSize: '14px', color: TEXT_SECONDARY, margin: '0 0 4px 0' }}>選択されたファイル:</p>
                <p
                  style={{
                    fontWeight: '500',
                    color: TEXT_PRIMARY,
                    margin: '0 0 12px 0',
                    wordBreak: 'break-all',
                    fontSize: isMobile ? '13px' : '14px',
                  }}
                >
                  {selected.name}
                </p>
                <button
                  onClick={handleUpload}
                  disabled={uploading}
                  style={{
                    ...buttonStyle,
                    width: '100%',
                    justifyContent: 'center',
                    opacity: uploading ? 0.6 : 1,
                    cursor: uploading ? 'not-allowed' : 'pointer',
                  }}
                  onMouseEnter={(e) => {
                    if (!uploading) e.currentTarget.style.backgroundColor = '#0052cc';
                  }}
                  onMouseLeave={(e) => {
                    if (!uploading) e.currentTarget.style.backgroundColor = DROPBOX_BLUE;
                  }}
                >
                  {uploading ? 'アップロード中...' : 'アップロード'}
                </button>
              </div>
            )}
          </div>
        </div>
        {viewMode === 'grid' && !isMobile ? (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: getGridColumns(),
              gap: isMobile ? '12px' : '16px',
            }}
          >
            {filteredFiles.map((file, index) => (
              <div
                key={index}
                style={{ ...cardStyle, cursor: 'pointer' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = 'none';
                }}
              >
                <div style={{ padding: isMobile ? '12px' : '16px' }}>
                  <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '12px' }}>
                    {getFileIcon(file.type || '')}
                  </div>
                  <h3
                    style={{
                      fontWeight: '500',
                      color: TEXT_PRIMARY,
                      fontSize: isMobile ? '13px' : '14px',
                      margin: '0 0 4px 0',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                    title={file.name}
                  >
                    {file.name}
                  </h3>
                  <p style={{ fontSize: '12px', color: TEXT_SECONDARY, margin: '0' }}>{file.size}</p>
                  <p style={{ fontSize: '12px', color: TEXT_SECONDARY, margin: '4px 0 0 0' }}>{file.modified}</p>
                </div>
                <div
                  style={{
                    borderTop: `1px solid ${BORDER_COLOR}`,
                    backgroundColor: BACKGROUND_GRAY,
                    padding: isMobile ? '6px 12px' : '8px 16px',
                    display: 'flex',
                    justifyContent: 'flex-end',
                    gap: '4px',
                  }}
                >
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      downloadFile(file.name);
                    }}
                    style={iconButtonStyle}
                    title="ダウンロード"
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = DROPBOX_BLUE;
                      e.currentTarget.style.backgroundColor = '#f0f7ff';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = TEXT_SECONDARY;
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    <icons.Download style={{ width: '16px', height: '16px' }} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(file.name);
                    }}
                    style={iconButtonStyle}
                    title="削除"
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = '#dc2626';
                      e.currentTarget.style.backgroundColor = '#fef2f2';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = TEXT_SECONDARY;
                      e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    <icons.Trash2 style={{ width: '16px', height: '16px' }} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={cardStyle}>
            {!isMobile && (
              <div
                style={{
                  padding: '12px 20px',
                  borderBottom: `1px solid ${BORDER_COLOR}`,
                  backgroundColor: BACKGROUND_GRAY,
                }}
              >
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: isTablet ? '2fr 1fr 80px' : '2fr 1fr 1fr 100px',
                    gap: '16px',
                    fontSize: '12px',
                    fontWeight: '600',
                    color: TEXT_SECONDARY,
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                  }}
                >
                  <div>名前</div>
                  <div>サイズ</div>
                  {!isTablet && <div>更新日時</div>}
                  <div></div>
                </div>
              </div>
            )}
            <div>
              {filteredFiles.map((file, index) => (
                <div
                  key={index}
                  style={{
                    padding: isMobile ? '16px' : '12px 20px',
                    borderBottom:
                      index < filteredFiles.length - 1 ? `1px solid ${BORDER_COLOR}` : 'none',
                    transition: 'background-color 0.2s ease',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = BACKGROUND_GRAY;
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  {isMobile ? (
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                      <div style={{ flexShrink: 0, marginTop: '2px' }}>{getFileIcon(file.type || '')}</div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div
                          style={{
                            fontWeight: '500',
                            color: TEXT_PRIMARY,
                            fontSize: '14px',
                            marginBottom: '4px',
                            wordBreak: 'break-all',
                          }}
                        >
                          {file.name}
                        </div>
                        <div style={{ fontSize: '13px', color: TEXT_SECONDARY, marginBottom: '8px' }}>
                          {file.size} • {file.modified}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: '4px', flexShrink: 0 }}>
                        <button
                          onClick={() => downloadFile(file.name)}
                          style={iconButtonStyle}
                          title="ダウンロード"
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = DROPBOX_BLUE;
                            e.currentTarget.style.backgroundColor = '#f0f7ff';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.color = TEXT_SECONDARY;
                            e.currentTarget.style.backgroundColor = 'transparent';
                          }}
                        >
                          <icons.Download style={{ width: '16px', height: '16px' }} />
                        </button>
                        <button
                          onClick={() => handleDelete(file.name)}
                          style={iconButtonStyle}
                          title="削除"
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = '#dc2626';
                            e.currentTarget.style.backgroundColor = '#fef2f2';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.color = TEXT_SECONDARY;
                            e.currentTarget.style.backgroundColor = 'transparent';
                          }}
                        >
                          <icons.Trash2 style={{ width: '16px', height: '16px' }} />
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div
                      style={{
                        display: 'grid',
                        gridTemplateColumns: isTablet ? '2fr 1fr 80px' : '2fr 1fr 1fr 100px',
                        gap: '16px',
                        alignItems: 'center',
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        {getFileIcon(file.type || '')}
                        <span
                          style={{
                            fontWeight: '400',
                            color: TEXT_PRIMARY,
                            fontSize: '14px',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {file.name}
                        </span>
                      </div>
                      <div style={{ fontSize: '14px', color: TEXT_SECONDARY }}>{file.size}</div>
                      {!isTablet && <div style={{ fontSize: '14px', color: TEXT_SECONDARY }}>{file.modified}</div>}
                      <div style={{ display: 'flex', gap: '4px', justifyContent: 'flex-end' }}>
                        <button
                          onClick={() => downloadFile(file.name)}
                          style={iconButtonStyle}
                          title="ダウンロード"
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = DROPBOX_BLUE;
                            e.currentTarget.style.backgroundColor = '#f0f7ff';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.color = TEXT_SECONDARY;
                            e.currentTarget.style.backgroundColor = 'transparent';
                          }}
                        >
                          <icons.Download style={{ width: '16px', height: '16px' }} />
                        </button>
                        <button
                          onClick={() => handleDelete(file.name)}
                          style={iconButtonStyle}
                          title="削除"
                          onMouseEnter={(e) => {
                            e.currentTarget.style.color = '#dc2626';
                            e.currentTarget.style.backgroundColor = '#fef2f2';
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.color = TEXT_SECONDARY;
                            e.currentTarget.style.backgroundColor = 'transparent';
                          }}
                        >
                          <icons.Trash2 style={{ width: '16px', height: '16px' }} />
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        {filteredFiles.length === 0 && (
          <div
            style={{
              textAlign: 'center',
              padding: isMobile ? '40px 20px' : '60px 20px',
              ...cardStyle,
            }}
          >
            <icons.Folder
              style={{
                width: isMobile ? '40px' : '48px',
                height: isMobile ? '40px' : '48px',
                color: TEXT_SECONDARY,
                margin: '0 auto 16px',
              }}
            />
            <p style={{ color: TEXT_SECONDARY, margin: 0, fontSize: isMobile ? '14px' : '16px' }}>
              {searchTerm ? '検索条件に一致するファイルが見つかりません' : 'ファイルがありません'}
            </p>
            {!searchTerm && (
              <p style={{ color: TEXT_SECONDARY, margin: '8px 0 0 0', fontSize: isMobile ? '13px' : '14px' }}>
                上のエリアにファイルをドラッグ&ドロップしてアップロードできます
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default MiniFileManager;