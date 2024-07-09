import React from 'react';
import { CircularProgress, Box } from '@mui/material';

const Spinner: React.FC = () => {
    return (
        <Box display="flex" justifyContent="center" alignItems="center" height="100%">
            <CircularProgress />
        </Box>
    );
};

export default Spinner;