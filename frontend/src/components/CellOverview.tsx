import React, { useEffect, useState } from "react";
import axios from "axios";
import { Select, MenuItem, FormControl, InputLabel, Grid, Box, Button, Typography, TextField, FormControlLabel, Checkbox } from "@mui/material";

const CellImageGrid: React.FC = () => {
    const [cellIds, setCellIds] = useState<string[]>([]);
    const [images, setImages] = useState<{ [key: string]: { ph: string, fluo: string } }>({});
    const [mode, setMode] = useState<string>("ph");
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [drawContour, setDrawContour] = useState<boolean>(false);
    const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
    const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);

    useEffect(() => {
        const fetchCellIds = async () => {
            const response = await axios.get("http://localhost:8000/cells");
            const ids = response.data.map((cell: { cell_id: string }) => cell.cell_id);
            setCellIds(ids);
        };

        fetchCellIds();
    }, []);

    useEffect(() => {
        const fetchImages = async (cellId: string) => {
            try {
                const fetchImage = async (url: string) => {
                    console.log(`Fetching image from URL: ${url}`);
                    const response = await axios.get(url, { responseType: 'blob' });
                    const imageUrl = URL.createObjectURL(response.data);
                    return imageUrl;
                };

                const phImage = await fetchImage(`http://localhost:8000/cells/${cellId}/ph_image?draw_contour=${drawContour}&draw_scale_bar=${drawScaleBar}`);
                const fluoImage = await fetchImage(`http://localhost:8000/cells/${cellId}/fluo_image?draw_contour=${drawContour}&draw_scale_bar=${drawScaleBar}&brightness_factor=${brightnessFactor}`);

                setImages((prevImages) => ({
                    ...prevImages,
                    [cellId]: { ph: phImage, fluo: fluoImage }
                }));
            } catch (error) {
                console.error("Error fetching images:", error);
            }
        };

        if (cellIds.length > 0 && !images[cellIds[currentIndex]]) {
            fetchImages(cellIds[currentIndex]);
        }
    }, [cellIds, currentIndex, drawContour, drawScaleBar, brightnessFactor]);

    const handleNext = () => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % cellIds.length);
    };

    const handlePrev = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1 + cellIds.length) % cellIds.length);
    };

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
            <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
                <Button variant="contained" color="primary" onClick={handlePrev} disabled={cellIds.length === 0}>
                    Prev
                </Button>
                <Typography variant="h6">
                    {cellIds.length > 0 ? `Cell ${currentIndex + 1} of ${cellIds.length}` : "Loading..."}
                </Typography>
                <Button variant="contained" color="primary" onClick={handleNext} disabled={cellIds.length === 0}>
                    Next
                </Button>
            </Box>
            <Box mt={2}>
                <FormControlLabel
                    control={<Checkbox checked={drawContour} onChange={(e) => setDrawContour(e.target.checked)} />}
                    label="Draw Contour"
                />
                <FormControlLabel
                    control={<Checkbox checked={drawScaleBar} onChange={(e) => setDrawScaleBar(e.target.checked)} />}
                    label="Draw Scale Bar"
                />
                <TextField
                    label="Brightness Factor"
                    type="number"
                    value={brightnessFactor}
                    onChange={(e) => setBrightnessFactor(parseFloat(e.target.value))}
                    InputProps={{
                        inputProps: { min: 0.1, step: 0.1 },
                        onWheel: (e) => e.currentTarget.blur() // Disable mouse wheel adjustments
                    }}
                />
            </Box>
            <Grid container spacing={2} style={{ marginTop: 20 }}>
                <Grid item xs={6}>
                    {images[cellIds[currentIndex]] ? (
                        <img src={images[cellIds[currentIndex]].ph} alt={`Cell ${cellIds[currentIndex]} PH`} style={{ width: "100%" }} />
                    ) : (
                        <div>Loading PH...</div>
                    )}
                </Grid>
                <Grid item xs={6}>
                    {images[cellIds[currentIndex]] ? (
                        <img src={images[cellIds[currentIndex]].fluo} alt={`Cell ${cellIds[currentIndex]} Fluo`} style={{ width: "100%" }} />
                    ) : (
                        <div>Loading Fluo...</div>
                    )}
                </Grid>
            </Grid>
        </Box>
    );
};

export default CellImageGrid;
