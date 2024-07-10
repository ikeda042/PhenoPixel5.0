import React, { useEffect, useState } from "react";
import { Box, Typography, Container } from "@mui/material";
import axios from "axios";

interface ListDBResponse {
    databases: string[];
}

const TopPage: React.FC = () => {
    const [databases, setDatabases] = useState<string[]>([]);

    useEffect(() => {
        const fetchDatabases = async () => {
            try {
                const response = await axios.get<ListDBResponse>("/database");
                setDatabases(response.data.databases);
            } catch (error) {
                console.error("Failed to fetch databases", error);
            }
        };

        fetchDatabases();
    }, []);

    return (
        <Container>
            <Typography variant="h4" component="h1" gutterBottom>
                Database Names
            </Typography>
            <Box>
                {databases.map((database, index) => (
                    <Typography key={index}>{database}</Typography>
                ))}
            </Box>
        </Container>
    );
};

export default TopPage;