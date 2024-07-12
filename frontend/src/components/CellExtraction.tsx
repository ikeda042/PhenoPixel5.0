import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import {
    Box, Grid, Typography, TextField, Button, MenuItem, Select, FormControl, InputLabel, CircularProgress
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const Extraction: React.FC = () => {
    const [searchParams] = useSearchParams();
    const fileName = searchParams.get("file_name") || "";
    const [mode, setMode] = useState("dual");
    const [param1, setParam1] = useState(100);
    const [imageSize, setImageSize] = useState(200);
    const [isLoading, setIsLoading] = useState(false);
    const [extractedImages, setExtractedImages] = useState<string[]>([]);

    const handleExtractCells = async () => {
        setIsLoading(true);
        try {
            const response = await axios.get(`${url_prefix}/cell_extraction/${fileName}/${mode}`, {
                params: {
                    param1,
                    image_size: imageSize,
                },
            });
            setExtractedImages(response.data.images);
        } catch (error) {
            console.error("Failed to extract cells", error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Box>
            <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                    <FormControl fullWidth margin="normal">
                        <InputLabel>Mode</InputLabel>
                        <Select
                            value={mode}
                            onChange={(e) => setMode(e.target.value)}
                        >
                            <MenuItem value="single_layer">Single Layer</MenuItem>
                            <MenuItem value="dual_layer">Dual Layer</MenuItem>
                            <MenuItem value="triple_layer">Triple Layer</MenuItem>
                        </Select>
                    </FormControl>
                    <TextField
                        label="Param1"
                        type="number"
                        fullWidth
                        margin="normal"
                        value={param1}
                        onChange={(e) => setParam1(Number(e.target.value))}
                    />
                    <TextField
                        label="Image Size"
                        type="number"
                        fullWidth
                        margin="normal"
                        value={imageSize}
                        onChange={(e) => setImageSize(Number(e.target.value))}
                    />
                    <Button
                        variant="contained"
                        color="primary"
                        fullWidth
                        onClick={handleExtractCells}
                        disabled={isLoading}
                        sx={{
                            backgroundColor: 'black',
                            color: 'white',
                            width: '100%',
                            height: '56px',
                            '&:hover': {
                                backgroundColor: 'grey'
                            }
                        }}
                    >
                        {isLoading ? <CircularProgress size={24} /> : "Extract Cells"}
                    </Button>
                </Grid>
                <Grid item xs={12} md={8}>
                    <Box display="flex" flexWrap="wrap" gap={2}>
                        {extractedImages.map((image, index) => (
                            <Box key={index} component="img" src={image} alt={`Extracted cell ${index}`} />
                        ))}
                    </Box>
                </Grid>
            </Grid>
        </Box>
    );
};

export default Extraction;
