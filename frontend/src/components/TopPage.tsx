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

const TopPage: React.FC = () => {
    const navigate = useNavigate();
    const [backendStatus, setBackendStatus] = useState<string | null>(null);
    const [dropboxStatus, setDropboxStatus] = useState<boolean | null>(null);

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

        checkBackend();
        checkDropboxConnection();
    }, []);

    const handleNavigate = (path: string) => {
        navigate(path);
    };

    const menuItems = [
        { title: "Data Analyses", icon: <DatabaseIcon />, path: '/dbconsole' },
        { title: "Results", icon: <Inventory2Icon />, path: '/results' },
        { title: "Cell Extraction", icon: <ScienceIcon />, path: '/nd2files' },
        { title: "X100TLengine", icon: <DisplaySettingsIcon />, path: '/tl-engine' },
        { title: "GraphEngine", icon: <BarChartIcon />, path: '/graphengine' },
        { title: "Swagger UI", icon: <TerminalIcon />, path: `${settings.url_prefix}/docs`, external: true },
        { title: "Github", icon: <GitHubIcon />, path: 'https://github.com/ikeda042/PhenoPixel5.0', external: true },
    ];

    return (
        <Container>
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" height="120vh">
                {backendStatus && (
                    <Typography variant="h6" color={backendStatus === "ready" ? "green" : "red"}>
                        Backend {backendStatus}: {settings.url_prefix}
                    </Typography>
                )}
                {dropboxStatus !== null && (
                    <Typography variant="h6" color={dropboxStatus ? "green" : "red"}>
                        Dropbox {dropboxStatus ? "Connected" : "Not connected"}
                    </Typography>
                )}
                <Grid container spacing={2} justifyContent="center">
                    {menuItems.map((item, index) => (
                        <Grid item xs={12} sm={6} md={3} key={index}>
                            <Card
                                onClick={() => item.external ? window.open(item.path, '_blank') : handleNavigate(item.path)}
                                sx={{ cursor: 'pointer', textAlign: 'center', height: '150px', display: 'flex', flexDirection: 'column', justifyContent: 'center', '&:hover': { backgroundColor: 'lightgrey' } }}
                            >
                                <CardContent>
                                    {item.icon}
                                    <Typography variant="h6" mt={2}>
                                        {item.title}
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
