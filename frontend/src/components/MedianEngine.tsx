import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Box, CircularProgress, Button } from '@mui/material';
import { settings } from '../settings';
import DownloadIcon from '@mui/icons-material/Download';

interface ImageFetcherProps {
    dbName: string;
    label: string;
    cellId: string;
}
const url_prefix = settings.url_prefix;

const MedianEngine: React.FC<ImageFetcherProps> = ({ dbName, label, cellId }) => {
    const [imageUrl, setImageUrl] = useState<string | null>(null);
    const [loading, setLoading] = useState<boolean>(false);

    useEffect(() => {
        const fetchImageData = async () => {
            setLoading(true);
            try {
                const response = await axios.get(`${url_prefix}/cells/${dbName}/${label}/${cellId}/median_fluo_intensities`, { responseType: 'blob' });
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
    }, [dbName, label, cellId]);

    const handleDownloadCsv = async () => {
        try {
            const response = await axios.get(`${url_prefix}/cells/${dbName}/${label}/median_fluo_intensities/csv`, { responseType: 'blob' });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `${dbName}_median_fluo_intensities.csv`);
            document.body.appendChild(link);
            link.click();
            link?.parentNode?.removeChild(link);
        } catch (error) {
            console.error('Failed to download CSV:', error);
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
                    color: 'text.primary',
                    backgroundColor: 'background.paper',
                    '&:hover': {
                        backgroundColor: 'action.hover',
                    },
                    marginTop: 1,
                }}
                startIcon={<DownloadIcon />}
            >
                Download CSV
            </Button>
        </Box>
    );
};

export default MedianEngine;