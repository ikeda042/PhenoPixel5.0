import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Box, CircularProgress, Button, FormControl, InputLabel, Select, MenuItem } from '@mui/material';
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
    const [hasFluo2, setHasFluo2] = useState<boolean>(false);
    const [channel, setChannel] = useState<'fluo1' | 'fluo2'>('fluo1');

    useEffect(() => {
        const checkFluo2 = async () => {
            try {
                const res = await axios.get(`${url_prefix}/databases/${dbName}/has-fluo2`);
                setHasFluo2(res.data.has_fluo2);
            } catch (e) {
                console.error('Failed to check fluo2 existence', e);
            }
        };
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

        checkFluo2();
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
            const channelParam = channel === 'fluo2' ? 2 : 1;
            const response = await axios.get(`${url_prefix}/cells/${dbName}/${label}/${cellId}/heatmap/bulk/csv?channel=${channelParam}`, { responseType: 'blob' });
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
                sx={{ mt: 1 }}
                startIcon={<DownloadIcon />}
            >
                Download CSV
            </Button>
            {hasFluo2 && (
                <FormControl sx={{ mt: 1 }} size="small">
                    <InputLabel id="channel-select-label">Channel</InputLabel>
                    <Select
                        labelId="channel-select-label"
                        value={channel}
                        label="Channel"
                        onChange={(e) => setChannel(e.target.value as 'fluo1' | 'fluo2')}
                    >
                        <MenuItem value="fluo1">fluo1</MenuItem>
                        <MenuItem value="fluo2">fluo2</MenuItem>
                    </Select>
                </FormControl>
            )}
            <Button
                variant="contained"
                onClick={handleBulkDownloadCsv}
                sx={{ mt: 1 }}
                startIcon={<DownloadIcon />}
                disabled={bulkLoading}
            >
                {bulkLoading ? <CircularProgress size={24} /> : 'All paths csv(queued)'}
            </Button>
        </Box>
    );
};

export default HeatmapEngine;
