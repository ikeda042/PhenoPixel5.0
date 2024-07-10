import React, { useEffect, useState } from "react";
import { Box, Typography, Container, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, IconButton } from "@mui/material";
import axios from "axios";
import { settings } from "../settings";
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import { useNavigate } from "react-router-dom";

interface ListDBResponse {
    databases: string[];
}

const url_prefix = settings.url_prefix;

const TopPage: React.FC = () => {
    const [databases, setDatabases] = useState<string[]>([]);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchDatabases = async () => {
            try {
                const response = await axios.get<ListDBResponse>(`${url_prefix}/databases`);
                setDatabases(response.data.databases);
            } catch (error) {
                console.error("Failed to fetch databases", error);
            }
        };

        fetchDatabases();
    }, []);

    const handleNavigate = (dbName: string) => {
        navigate(`/databases/?db_name=${dbName}`);
    };

    return (
        <Container>
            <Typography variant="h4" component="h1" gutterBottom>
                Database Names
            </Typography>
            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Database Name</TableCell>
                            <TableCell align="right">Go</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {databases.map((database, index) => (
                            <TableRow key={index}>
                                <TableCell component="th" scope="row">
                                    {database}
                                </TableCell>
                                <TableCell align="right">
                                    <IconButton onClick={() => handleNavigate(database)}>
                                        <Typography >Access database </Typography>
                                        <NavigateNextIcon />
                                    </IconButton>
                                </TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Container>
    );
};

export default TopPage;