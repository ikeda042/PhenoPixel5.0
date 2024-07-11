import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Box, CircularProgress, Button } from '@mui/material';
import { settings } from '../settings';

interface ImageFetcherProps {
    dbName: string;
    label: string;
    cellId: string;
}
const url_prefix = settings.url_prefix;

const SK326Engine: React.FC<ImageFetcherProps> = ({ dbName, label, cellId }) => {
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
            const response = await axios.get(`${url_prefix}/cells/${dbName}/${label}/mean_fluo_intensities/csv`, { responseType: 'blob' });
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `${label}_mean_fluo_intensities.csv`);
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
        <Box display="flex" flexDirection="column" alignItems="center">
            <Box>
                <img src={imageUrl} alt="Cell" style={{ maxWidth: '100%', height: 'auto' }} />
            </Box>
            <Button variant="contained" color="primary" onClick={handleDownloadCsv}>
                Download CSV
            </Button>
        </Box>
    );
};

export default SK326Engine;