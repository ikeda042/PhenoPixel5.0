import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
    Box,
    CircularProgress,
    Button,
    ToggleButtonGroup,
    ToggleButton,
    Typography,
} from '@mui/material';
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
    const [selectedChannel, setSelectedChannel] = useState<'fluo1' | 'fluo2'>('fluo1');
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchImageData = async () => {
            setLoading(true);
            try {
                setImageUrl(null);
                setError(null);
                const response = await axios.get(
                    `${url_prefix}/cells/${dbName}/${label}/${cellId}/median_fluo_intensities`,
                    {
                        responseType: 'blob',
                        params: { img_type: selectedChannel },
                    }
                );
                const imageBlobUrl = URL.createObjectURL(response.data);
                setImageUrl(imageBlobUrl);
            } catch (error) {
                console.error('Failed to fetch image data:', error);
                setImageUrl(null);
                setError('No image available for the selected channel.');
            } finally {
                setLoading(false);
            }
        };

        fetchImageData();
    }, [dbName, label, cellId, selectedChannel]);

    useEffect(() => {
        return () => {
            if (imageUrl) {
                URL.revokeObjectURL(imageUrl);
            }
        };
    }, [imageUrl]);

    const handleChannelChange = (
        _event: React.MouseEvent<HTMLElement>,
        newChannel: 'fluo1' | 'fluo2' | null
    ) => {
        if (newChannel) {
            setSelectedChannel(newChannel);
        }
    };

    const handleDownloadCsv = async () => {
        try {
            const response = await axios.get(
                `${url_prefix}/cells/${dbName}/${label}/median_fluo_intensities/csv`,
                {
                    responseType: 'blob',
                    params: { img_type: selectedChannel },
                }
            );
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `${dbName}_median_${selectedChannel}_fluo_intensities.csv`);
            document.body.appendChild(link);
            link.click();
            link?.parentNode?.removeChild(link);
        } catch (error) {
            console.error('Failed to download CSV:', error);
        }
    };

    return (
        <Box display="flex" flexDirection="column" alignItems="flex-end">
            <Box display="flex" flexDirection="column" alignItems="flex-end" mb={2}>
                <Typography variant="subtitle2" sx={{ mb: 1 }}>
                    Fluorescence channel
                </Typography>
                <ToggleButtonGroup
                    size="small"
                    exclusive
                    value={selectedChannel}
                    onChange={handleChannelChange}
                >
                    <ToggleButton value="fluo1">Fluo 1</ToggleButton>
                    <ToggleButton value="fluo2">Fluo 2</ToggleButton>
                </ToggleButtonGroup>
            </Box>
            {loading ? (
                <Box display="flex" justifyContent="center" width="100%">
                    <CircularProgress />
                </Box>
            ) : imageUrl ? (
                <img src={imageUrl} alt="Cell" style={{ maxWidth: '100%', height: 'auto' }} />
            ) : (
                <Box>{error ?? 'No image available'}</Box>
            )}
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
                disabled={!imageUrl || loading}
            >
                Download CSV
            </Button>
        </Box>
    );
};

export default MedianEngine;
