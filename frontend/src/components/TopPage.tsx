import React, { useEffect, useState } from "react";
import { Box, Button, Container, Typography } from "@mui/material";
import DatabaseIcon from '@mui/icons-material/Storage';
import ScienceIcon from '@mui/icons-material/Science';
import { useNavigate } from "react-router-dom";
import { settings } from "../settings";

const TopPage: React.FC = () => {
    const navigate = useNavigate();
    const [backendReady, setBackendReady] = useState(false);

    useEffect(() => {
        const checkBackend = async () => {
            try {
                const response = await fetch(`${settings.url_prefix}/healthcheck`);
                if (response.status === 200) {
                    setBackendReady(true);
                }
            } catch (error) {
                console.error("Error checking backend health:", error);
            }
        };

        checkBackend();
    }, []);

    const handleNavigate = (path: string) => {
        navigate(path);
    };

    return (
        <Container>
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" gap={2} height="100vh">
                {backendReady && (
                    <Typography variant="h6" color="green">
                        Backend ready: {settings.url_prefix}
                    </Typography>
                )}
                <Button
                    variant="contained"
                    component="span"
                    startIcon={<DatabaseIcon />}
                    onClick={() => handleNavigate('/dbconsole')}
                    sx={{
                        backgroundColor: 'white',
                        color: 'black',
                        width: '100%',
                        height: '56px',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    Data analyses
                </Button>
                <Button
                    variant="contained"
                    component="span"
                    startIcon={<ScienceIcon />}
                    onClick={() => handleNavigate('/nd2files')}
                    sx={{
                        backgroundColor: 'white',
                        color: 'black',
                        width: '100%',
                        height: '56px',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    Cell extraction
                </Button>
            </Box>
        </Container>
    );
};

export default TopPage;