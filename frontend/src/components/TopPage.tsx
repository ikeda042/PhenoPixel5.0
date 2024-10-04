import React, { useEffect, useState } from "react";
import { Box, Button, Container, Typography } from "@mui/material";
import DatabaseIcon from '@mui/icons-material/Storage';
import ScienceIcon from '@mui/icons-material/Science';
import TerminalIcon from '@mui/icons-material/Terminal';
import { useNavigate } from "react-router-dom";
import { settings } from "../settings";
import GitHubIcon from '@mui/icons-material/GitHub';
import BarChartIcon from '@mui/icons-material/BarChart';
import DisplaySettingsIcon from '@mui/icons-material/DisplaySettings';

const TopPage: React.FC = () => {
    const navigate = useNavigate();
    const [backendStatus, setBackendStatus] = useState<string | null>(null);

    useEffect(() => {
        const checkBackend = async () => {
            try {
                const response = await fetch(`${settings.url_prefix}/healthcheck`);
                if (response.status === 200) {
                    setBackendStatus("ready");
                } else {
                    setBackendStatus("not working");
                }
            } catch (error) {
                console.error("Error checking backend health:", error);
                setBackendStatus("not working");
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
                {backendStatus && (
                    <Typography variant="h6" color={backendStatus === "ready" ? "green" : "red"}>
                        Backend {backendStatus}: {settings.url_prefix}
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
                        textTransform: 'none',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    Data Analyses
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
                        textTransform: 'none',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    Cell Extraction
                </Button>
                <Button
                    variant="contained"
                    component="span"
                    startIcon={<DisplaySettingsIcon />}
                    onClick={() => handleNavigate('/tl-engine')}
                    sx={{
                        backgroundColor: 'white',
                        color: 'black',
                        width: '100%',
                        height: '56px',
                        textTransform: 'none',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    X100TLengine
                </Button>
                <Button
                    variant="contained"
                    component="span"
                    startIcon={<BarChartIcon />}
                    onClick={() => handleNavigate('/graphengine')}
                    sx={{
                        backgroundColor: 'white',
                        color: 'black',
                        width: '100%',
                        height: '56px',
                        textTransform: 'none',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    GraphEngine
                </Button>
                <Button
                    variant="contained"
                    component="span"
                    startIcon={<TerminalIcon />}
                    onClick={() => window.open(`${settings.url_prefix}/docs`, '_blank')}
                    sx={{
                        backgroundColor: 'white',
                        color: 'black',
                        width: '100%',
                        height: '56px',
                        textTransform: 'none',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    Swagger UI
                </Button>
                <Button
                    variant="contained"
                    component="span"
                    startIcon={<GitHubIcon />}
                    onClick={() => window.open(`https://github.com/ikeda042/PhenoPixel5.0`, '_blank')}
                    sx={{
                        backgroundColor: 'white',
                        color: 'black',
                        width: '100%',
                        height: '56px',
                        textTransform: 'none',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    Github
                </Button>
            </Box>
        </Container>
    );
};

export default TopPage;