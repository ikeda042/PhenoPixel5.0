import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper } from '@material-ui/core';
import { settings } from '../settings';

interface CellMorphologyTableProps {
    cellId: string;
}

const CellMorphologyTable: React.FC<CellMorphologyTableProps> = ({ cellId }) => {
    const [cellMorphology, setCellMorphology] = useState(null);

    useEffect(() => {
        const fetchCellMorphology = async () => {
            try {
                const response = await axios.get(`${settings.api_url}/cells/${cellId}/morphology`);
                setCellMorphology(response.data);
            } catch (error) {
                console.error("Error fetching cell morphology data:", error);
            }
        };

        fetchCellMorphology();
    }, [cellId]);

    if (!cellMorphology) {
        return <div>Loading...</div>;
    }

    return (
        <TableContainer component={Paper}>
            <Table>
                <TableHead>
                    <TableRow>
                        <TableCell>Parameter</TableCell>
                        <TableCell align="right">Value</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {Object.entries(cellMorphology).map(([key, value]) => (
                        <TableRow key={key}>
                            <TableCell component="th" scope="row">
                                {key}
                            </TableCell>
                            <TableCell align="right">{value}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

export default CellMorphologyTable;