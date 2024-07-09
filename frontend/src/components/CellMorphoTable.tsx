import React, { useEffect, useState } from 'react';
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Box } from '@material-ui/core';

interface CellMorphologyTableProps {
    cellId: string;
}
interface CellMorphologyData {
    [key: string]: number;
}
const CellMorphologyTable: React.FC<CellMorphologyTableProps> = ({ cellId }) => {
    const [cellMorphology, setCellMorphology] = useState<CellMorphologyData | null>(null);


    const parameterDisplayNameMapping: { [key: string]: string } = {
        area: 'Area(px^2)',
        volume: 'Volume(px^3)',
        width: 'Width(px)',
        length: 'Length(px)',
        mean_fluo_intensity: 'Mean Fluorescence Intensity',
        mean_ph_intensity: 'Mean PH Intensity',
        mean_fluo_intensity_normalized: 'Mean Fluorescence Intensity (Normalized)',
        mean_ph_intensity_normalized: 'Mean PH Intensity (Normalized)',
        median_fluo_intensity: 'Median Fluorescence Intensity',
        median_ph_intensity: 'Median PH Intensity',
        median_fluo_intensity_normalized: 'Median Fluorescence Intensity (Normalized)',
        median_ph_intensity_normalized: 'Median PH Intensity (Normalized)',
    };

    useEffect(() => {
        const generateDummyData = () => {
            const dummyData: CellMorphologyData = Object.keys(parameterDisplayNameMapping).reduce((acc, key) => {
                acc[key] = Math.floor(Math.random() * 100);
                return acc;
            }, {} as CellMorphologyData);
            setCellMorphology(dummyData);
        };

        generateDummyData();
    }, [cellId]);

    if (!cellMorphology) {
        return <div>Loading...</div>;
    }

    return (
        <Box maxWidth="sm">
            <TableContainer component={Paper}>
                <Table size="small">
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