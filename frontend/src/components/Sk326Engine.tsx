import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Box, CircularProgress } from '@mui/material';
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

    if (loading) {
        return <Box display="flex" justifyContent="center"><CircularProgress /></Box>;
    }

    if (!imageUrl) {
        return <Box>No image available</Box>;
    }

    return (
        <Box display="flex" justifyContent="center">
            <img src={imageUrl} alt="Cell" style={{ maxWidth: '100%', height: 'auto' }} />
        </Box>
    );
};

export default SK326Engine;