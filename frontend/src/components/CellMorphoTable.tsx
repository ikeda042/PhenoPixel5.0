import React, { useEffect, useState } from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Box } from '@material-ui/core';
import { settings } from '../settings';

interface CellMorphologyTableProps {
    cellId: string;
    db_name: string;
    polyfitDegree: number;
}
interface CellMorphologyData {
    [key: string]: number;
}

const url_prefix = settings.url_prefix;

const CellMorphologyTable: React.FC<CellMorphologyTableProps> = ({ cellId, db_name, polyfitDegree }) => {
    const [cellMorphology, setCellMorphology] = useState<CellMorphologyData | null>(null);

    const parameterDisplayNameMapping: { [key: string]: string } = {
        area: "Area(µm^2)",
        volume: "Volume(µm^3)",
        width: "Width(µm)",
        length: "Length(µm)",
        mean_fluo_intensity: "Mean Fluorescence Intensity",
        mean_ph_intensity: "Mean PH Intensity",
        mean_fluo_intensity_normalized: "Mean Fluorescence Intensity (Normalized)",
        mean_ph_intensity_normalized: "Mean PH Intensity (Normalized)",
        median_fluo_intensity: "Median Fluorescence Intensity",
        median_ph_intensity: "Median PH Intensity",
        median_fluo_intensity_normalized: "Median Fluorescence Intensity (Normalized)",
        median_ph_intensity_normalized: "Median PH Intensity (Normalized)"
    };

    useEffect(() => {
        const fetchCellMorphologyData = async () => {
            const apiUrl = `${url_prefix}/cells/${cellId}/${db_name}/morphology?degree=${polyfitDegree}`;

            try {
                const response = await fetch(apiUrl);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data: CellMorphologyData = await response.json();
                setCellMorphology(data);
            } catch (error) {
                console.error("Failed to fetch cell morphology data:", error);
            }
        };

        fetchCellMorphologyData();
    }, [cellId, db_name, polyfitDegree]);

    if (!cellMorphology) {
        return <div>Loading...</div>;
    }

    return (
        <Box maxWidth="sm">
            <TableContainer component={Paper}>
                <Table size="small">
                    <TableHead>
                        <TableRow>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {Object.entries(cellMorphology).map(([key, value]) => (
                            <TableRow key={key}>
                                <TableCell component="th" scope="row">
                                    {parameterDisplayNameMapping[key] || key}
                                </TableCell>
                                <TableCell align="right">{String(value)}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </TableContainer>
        </Box>
    );
};

export default CellMorphologyTable;