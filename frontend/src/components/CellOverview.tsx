import React, { useEffect, useState } from "react";
import axios from "axios";
import { Select, MenuItem, FormControl, InputLabel, Grid, Box, Button, Typography, TextField, FormControlLabel, Checkbox } from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";

import { settings } from "../settings";


const CellImageGrid: React.FC = () => {
    const [cellIds, setCellIds] = useState<string[]>([]);
    const [images, setImages] = useState<{ [key: string]: { ph: string, fluo: string } }>({});
    const [label, setLabel] = useState<string>("N/A");
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [drawContour, setDrawContour] = useState<boolean>(false);
    const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
    const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);

    useEffect(() => {
        const fetchCellIds = async () => {
            const response = await axios.get(`${settings.api_url}/cells`, { params: { label } });
            const ids = response.data.map((cell: { cell_id: string }) => cell.cell_id);
            setCellIds(ids);
        };

        fetchCellIds();
    }, [label]);

    useEffect(() => {
        const fetchImages = async (cellId: string) => {
            try {
                const fetchImage = async (url: string) => {
                    console.log(`Fetching image from URL: ${url}`);
                    const response = await axios.get(url, { responseType: 'blob' });
                    const imageUrl = URL.createObjectURL(response.data);
                    return imageUrl;
                };

                const phImage = await fetchImage(`${settings.api_url}/cells/${cellId}/ph_image?draw_contour=${drawContour}&draw_scale_bar=${drawScaleBar}&brightness_factor=${brightnessFactor}`);
                const fluoImage = await fetchImage(`${settings.api_url}/cells/${cellId}/fluo_image?draw_contour=${drawContour}&draw_scale_bar=${drawScaleBar}&brightness_factor=${brightnessFactor}`);

                return { ph: phImage, fluo: fluoImage };
            } catch (error) {
                console.error("Error fetching images:", error);
                return null;
            }
        };

        if (cellIds.length > 0) {
            fetchImages(cellIds[currentIndex]).then(newImages => {
                if (newImages) {
                    setImages((prevImages) => ({
                        ...prevImages,
                        [cellIds[currentIndex]]: newImages
                    }));
                }
            });
        }
    }, [cellIds, currentIndex, drawContour, drawScaleBar, brightnessFactor]);

    const handleNext = () => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % cellIds.length);
    };

    const handlePrev = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1 + cellIds.length) % cellIds.length);
    };

    const handleLabelChange = (event: SelectChangeEvent<string>) => {
        setLabel(event.target.value);
    };

    const handleContourChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setDrawContour(e.target.checked);
    };

    const handleScaleBarChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setDrawScaleBar(e.target.checked);
    };

    const handleBrightnessChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setBrightnessFactor(parseFloat(e.target.value));
    };


    const handleWheel = (event: React.WheelEvent<HTMLDivElement>) => {
        event.currentTarget.blur();
        event.preventDefault();
    };


    return (
        <Box>
            <FormControl fullWidth>
                <InputLabel id="label-select-label">Label</InputLabel>
                <Select
                    labelId="label-select-label"
                    value={label}
                    onChange={handleLabelChange}
                >
                    <MenuItem value="N/A">N/A</MenuItem>
                    <MenuItem value="1">1</MenuItem>
                    <MenuItem value="2">2</MenuItem>
                    <MenuItem value="3">3</MenuItem>
                </Select>
            </FormControl>
            <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
                <Button variant="contained" color="primary" onClick={handlePrev} disabled={cellIds.length === 0} style={{ backgroundColor: "black" }}>
                    Prev
                </Button>
                <Typography variant="h6">
                    {cellIds.length > 0 ? `Cell ${currentIndex + 1} of ${cellIds.length}` : "Loading..."}
                </Typography>
                <Button variant="contained" color="primary" onClick={handleNext} disabled={cellIds.length === 0} style={{ backgroundColor: "black" }}>
                    Next
                </Button>
            </Box>
            <Box mt={2}>
                <FormControlLabel
                    control={<Checkbox checked={drawContour} onChange={handleContourChange} style={{ color: "black" }} />}
                    label="Draw Contour"
                    style={{ color: "black" }}
                />
                <FormControlLabel
                    control={<Checkbox checked={drawScaleBar} onChange={handleScaleBarChange} style={{ color: "black" }} />}
                    label="Draw Scale Bar"
                    style={{ color: "black" }}
                />
                <TextField
                    label="Brightness Factor"
                    type="number"
                    value={brightnessFactor}
                    onChange={handleBrightnessChange}
                    InputProps={{
                        inputProps: { min: 0.1, step: 0.1 },
                        onWheel: handleWheel
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