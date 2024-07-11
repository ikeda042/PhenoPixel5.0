import React, { useEffect, useState } from "react";
import { Box, Typography, Container, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, IconButton, TextField } from "@mui/material";
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
    const [searchQuery, setSearchQuery] = useState("");
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

    const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setSearchQuery(event.target.value);
    };

    const filteredDatabases = databases.filter(database => database.toLowerCase().includes(searchQuery.toLowerCase()));

    return (
        <Container>
            <Box display="flex" flexDirection="row" alignItems="center">
                <Typography variant="h4" component="h1" gutterBottom>
                    Databases
                </Typography>
            </Box>
            <Box mb={2}>
                <TextField
                    fullWidth
                    label="Search Database"
                    variant="outlined"
                    value={searchQuery}
                    onChange={handleSearchChange}
                />
            </Box>
            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Database Name</TableCell>
                            <TableCell align="right">Go</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {filteredDatabases.map((database, index) => (
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