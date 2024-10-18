import React, { useEffect, useState } from "react";
import { Box, Card, CardContent, Grid, Typography, Container } from "@mui/material";
import DatabaseIcon from '@mui/icons-material/Storage';
import ScienceIcon from '@mui/icons-material/Science';
import TerminalIcon from '@mui/icons-material/Terminal';
import { useNavigate } from "react-router-dom";
import { settings } from "../settings";
import GitHubIcon from '@mui/icons-material/GitHub';
import BarChartIcon from '@mui/icons-material/BarChart';
import DisplaySettingsIcon from '@mui/icons-material/DisplaySettings';
import Inventory2Icon from '@mui/icons-material/Inventory2';
import SettingsEthernetIcon from '@mui/icons-material/SettingsEthernet';

const TopPage: React.FC = () => {
    const navigate = useNavigate();
    const [backendStatus, setBackendStatus] = useState<string | null>(null);
    const [dropboxStatus, setDropboxStatus] = useState<boolean | null>(null);
    const [internetStatus, setInternetStatus] = useState<boolean | null>(null); // New state for internet status

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

        const checkDropboxConnection = async () => {
            try {
                const response = await fetch(`${settings.url_prefix}/dropbox/connection_check`);
                const data = await response.json();
                if (response.status === 200 && data.status) {
                    setDropboxStatus(true);
                } else {
                    setDropboxStatus(false);
                }
            } catch (error) {
                console.error("Error checking Dropbox connection:", error);
                setDropboxStatus(false);
            }
        };

        const checkInternetConnection = async () => { // New function to check internet status
            try {
                const response = await fetch(`${settings.url_prefix}/internet-connection`);
                const data = await response.json();
                if (response.status === 200 && data.status) {
                    setInternetStatus(true);
                } else {
                    setInternetStatus(false);
                }
            } catch (error) {
                console.error("Error checking internet connection:", error);
                setInternetStatus(false);
            }
        };

        checkBackend();
        checkDropboxConnection();
        checkInternetConnection(); // Call the new function
    }, []);

    const handleNavigate = (path: string) => {
        navigate(path);
    };

    const menuItems = [
        {
            title: "Data Analyses",
            icon: <DatabaseIcon />,
            path: '/dbconsole',
            description: "Label cells / manage databases."
        },
        {
            title: "Cell Extraction",
            icon: <ScienceIcon />,
            path: '/nd2files',
            description: "Extract cells from ND2 files."
        },
        {
            title: "Results",
            icon: <Inventory2Icon />,
            path: '/results',
            description: "Results for the queued jobs."
        },
        {
            title: "X100TLengine",
            icon: <DisplaySettingsIcon />,
            path: '/tl-engine',
            description: "Process nd2 timelapse files.(beta)"
        },
        {
            title: "GraphEngine",
            icon: <BarChartIcon />,
            path: '/graphengine',
            description: "Create graphs from the data."
        },
        {
            title: "Swagger UI",
            icon: <TerminalIcon />,
            path: `${settings.url_prefix}/docs`,
            description: "Test the API endpoints.",
            external: true
        },
        {
            title: "Github",
            icon: <GitHubIcon />,
            path: 'https://github.com/ikeda042/PhenoPixel5.0',
            description: "Project documentation.",
            external: true
        },
        {
            title: "System Status",
            icon: <SettingsEthernetIcon />,
            path: '#',
            description: (
                <>
                    <Typography variant="body2" color={backendStatus === "ready" ? "green" : "red"}>
                        Backend: {backendStatus || "unknown"} ({settings.url_prefix})
                    </Typography>
                    <Typography variant="body2" color={dropboxStatus !== null && dropboxStatus ? "green" : "red"}>
                        Dropbox: {dropboxStatus !== null ? (dropboxStatus ? "Connected" : "Not connected") : "unknown"}
                    </Typography>
                    <Typography variant="body2" color={internetStatus !== null && internetStatus ? "green" : "red"}>
                        Internet: {internetStatus !== null ? (internetStatus ? "Connected" : "Not connected") : "unknown"}
                    </Typography>
                </>
            ),
            external: false
        },
    ];

    return (
        <Container>
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height="100vh">
                <Grid container spacing={2} justifyContent="center">
                    {menuItems.map((item, index) => (
                        <Grid item xs={12} sm={6} md={3} key={index}>
                            <Card
                                onClick={() => item.external ? window.open(item.path, '_blank') : handleNavigate(item.path)}
                                sx={{
                                    cursor: 'pointer',
                                    textAlign: 'center',
                                    height: '200px',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    justifyContent: 'center',
                                    boxShadow: 6,
                                    transition: 'box-shadow 0.3s ease-in-out',
                                    '&:hover': {
                                        backgroundColor: 'lightgrey',
                                        boxShadow: 10
                                    }
                                }}
                            >
                                <CardContent>
                                    {item.icon}
                                    <Typography variant="h6" mt={2}>
                                        {item.title}
                                    </Typography>
                                    <Typography variant="body2" mt={1} color="textSecondary">
                                        {typeof item.description === "string" ? item.description : item.description}
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            </Box>
        </Container>
    );
};

export default TopPage;
