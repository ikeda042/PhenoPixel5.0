import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Box, CircularProgress, Button } from '@mui/material';
import { settings } from '../settings';
import DownloadIcon from '@mui/icons-material/Download';

interface ImageFetcherProps {
    dbName: string;
    label: string;
    cellId: string;
    degree?: number;
}
const url_prefix = settings.url_prefix;

const HeatmapEngine: React.FC<ImageFetcherProps> = ({ dbName, label, cellId, degree = 3 }) => {
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [bulkLoading, setBulkLoading] = useState<boolean>(false);

    useEffect(() => {
        const fetchImageData = async () => {
            setLoading(true);
            try {
                const response = await axios.get(`${url_prefix}/cells/${dbName}/${label}/${cellId}/heatmap`, { responseType: 'blob' });
                const imageBlobUrl = URL.createObjectURL(response.data);
                setImageUrl(imageBlobUrl);
            } catch (error) {
                console.error('Failed to fetch image data:', error);
                setImageUrl(null);
            } finally {
                setLoading(false);
            }
        };

        fetchImageData();
    }, [dbName, label, cellId, degree]);

    const handleDownloadCsv = async () => {
        try {
            const response = await axios.get(`${url_prefix}/cells/${dbName}/${label}/${cellId}/heatmap/csv`, { responseType: 'blob' });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `${dbName}_peak_path.csv`);
            document.body.appendChild(link);
            link.click();
            link?.parentNode?.removeChild(link);
        } catch (error) {
            console.error('Failed to download CSV:', error);
        }
    };

    const handleBulkDownloadCsv = async () => {
        setBulkLoading(true);
        try {
            const response = await axios.get(`${url_prefix}/cells/${dbName}/${label}/${cellId}/heatmap/bulk/csv`, { responseType: 'blob' });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            const cleanedDbName = dbName.replace(/\.db$/, '');
            link.setAttribute('download', `${cleanedDbName}_bulk_peak_paths.csv`);
            document.body.appendChild(link);
            link.click();
            link?.parentNode?.removeChild(link);
        } catch (error) {
            console.error('Failed to download bulk CSV:', error);
        } finally {
            setBulkLoading(false);
        }
    };

    if (loading) {
        return <Box display="flex" justifyContent="center"><CircularProgress /></Box>;
    }

    if (!imageUrl) {
        return <Box>No image available</Box>;
    }

    return (
        <Box display="flex" flexDirection="column" alignItems="right">
            <img src={imageUrl} alt="Cell" style={{ maxWidth: '100%', height: 'auto' }} />
            <Button
                variant="contained"
                onClick={handleDownloadCsv}
                sx={{
                    color: 'black',
                    backgroundColor: '#ffffff',
                    '&:hover': {
                        backgroundColor: '#e0e0e0',
                    },
                    marginTop: 1,
                }}
                startIcon={<DownloadIcon />}
            >
                Download CSV
            </Button>
            <Button
                variant="contained"
                onClick={handleBulkDownloadCsv}
                sx={{
                    color: 'black',
                    backgroundColor: '#ffffff',
                    '&:hover': {
                        backgroundColor: '#e0e0e0',
                    },
                    marginTop: 1,
                }}
                startIcon={<DownloadIcon />}
                disabled={bulkLoading}
            >
                {bulkLoading ? <CircularProgress size={24} /> : 'Download CSV (bulk)'}
            </Button>
        </Box>
    );
};

export default HeatmapEngine;
