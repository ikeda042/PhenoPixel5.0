// src/components/CellImageGrid.tsx

import React, { useEffect, useState } from "react";
import axios from "axios";
import { Select, MenuItem, FormControl, InputLabel, Grid, Box } from "@mui/material";

const CellImageGrid: React.FC = () => {
    const [cellIds, setCellIds] = useState<string[]>([]);
    const [images, setImages] = useState<{ [key: string]: string }>({});
    const [mode, setMode] = useState<string>("ph");

    useEffect(() => {
        const fetchCellIds = async () => {
            const response = await axios.get("/cells");
            setCellIds(response.data);
        };

        fetchCellIds();
    }, []);

    useEffect(() => {
        const fetchImages = async () => {
            const newImages: { [key: string]: string } = {};
            for (const cellId of cellIds) {
                const response = await axios.get(`/cells/${cellId}/${mode}_image`);
                newImages[cellId] = response.data.image_url;
            }
            setImages(newImages);
        };

        if (cellIds.length > 0) {
            fetchImages();
        }
    }, [cellIds, mode]);

    return (
        <Box>
            <FormControl fullWidth>
                <InputLabel id="mode-select-label">Mode</InputLabel>
                <Select
                    labelId="mode-select-label"
                    value={mode}
                    onChange={(e) => setMode(e.target.value)}
                >
                    <MenuItem value="ph">PH</MenuItem>
                    <MenuItem value="fluo">Fluo</MenuItem>
                </Select>
            </FormControl>
            <Grid container spacing={2} style={{ marginTop: 20 }}>
                {cellIds.map((cellId) => (
                    <Grid item xs={12} sm={6} md={4} lg={3} key={cellId}>
                        <img src={images[cellId]} alt={`Cell ${cellId}`} style={{ width: "100%" }} />
                    </Grid>
                ))}
            </Grid>
        </Box>
    );
};

export default CellImageGrid;
