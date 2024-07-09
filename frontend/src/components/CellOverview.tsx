import React, { useEffect, useState } from "react";
import axios from "axios";
import { Stack, Select, MenuItem, FormControl, InputLabel, Grid, Box, Button, Typography, TextField, FormControlLabel, Checkbox } from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import { settings } from "../settings";
import { Scatter } from 'react-chartjs-2';
import { ChartOptions } from 'chart.js';

import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const CellImageGrid: React.FC = () => {
    const [cellIds, setCellIds] = useState<string[]>([]);
    const [images, setImages] = useState<{ [key: string]: { ph: string, fluo: string } }>({});
    const [label, setLabel] = useState<string>("1");
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [drawContour, setDrawContour] = useState<boolean>(false);
    const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
    const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);
    const [contourData, setContourData] = useState<number[][]>([]);
    const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number } | null>(null);
    const [drawMode, setDrawMode] = useState<string>("light");


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
                    const imageDimensions = await new Promise<{ width: number; height: number }>((resolve, reject) => {
                        const img = new Image();
                        img.onload = () => resolve({ width: img.width, height: img.height });
                        img.onerror = reject;
                        img.src = imageUrl;
                    });
                    setImageDimensions(imageDimensions);
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

    useEffect(() => {
        const fetchContour = async (cellId: string) => {
            try {
                const response = await axios.get(`${settings.api_url}/cells/${cellId}/contour/raw`);
                setContourData(response.data.contour);
            } catch (error) {
                console.error("Error fetching contour data:", error);
            }
        };

        if (cellIds.length > 0) {
            fetchContour(cellIds[currentIndex]);
        }
    }, [cellIds, currentIndex]);

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

    const handleDrawModeChange = (event: SelectChangeEvent<string>) => {
        setDrawMode(event.target.value);
    };

    // プロット用のデータを生成
    const contourPlotData = {
        datasets: [
            {
                label: 'Contour',
                data: contourData.map(point => ({ x: point[0], y: point[1] })),
                borderColor: 'lime',
                backgroundColor: 'lime',
                pointRadius: 1,
            }
        ]
    };

    const contourPlotOptions: ChartOptions<'scatter'> = {
        maintainAspectRatio: true,
        aspectRatio: 1,
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                min: 0,
                max: imageDimensions?.width,
                reverse: false,
            },
            y: {
                type: 'linear',
                min: 0,
                max: imageDimensions?.height,
                reverse: true
            }
        },
    };

    return (
        <>
            <Stack direction="row" spacing={2} alignItems="flex-start">
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
                    <Box mt={2}>
                        <FormControlLabel
                            control={<Checkbox checked={drawContour} onChange={handleContourChange} style={{ color: "black" }} />}
                            label="Detect Contour"
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
                    <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
                        <Button variant="contained" color="primary" onClick={handlePrev} disabled={cellIds.length === 0} style={{ backgroundColor: "black", minWidth: "100px" }}>
                            Prev
                        </Button>
                        <Typography variant="h6">
                            {cellIds.length > 0 ? `Cell ${currentIndex + 1} of ${cellIds.length}` : "Loading..."}
                        </Typography>
                        <Button variant="contained" color="primary" onClick={handleNext} disabled={cellIds.length === 0} style={{ backgroundColor: "black", minWidth: "100px" }}>
                            Next
                        </Button>
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
                <Box sx={{ width: 430, height: 430, marginLeft: 2 }}>
                    <FormControl fullWidth>
                        <InputLabel id="draw-mode-select-label">Draw Mode</InputLabel>
                        <Select
                            labelId="draw-mode-select-label"
                            value={drawMode}
                            onChange={handleDrawModeChange}
                            displayEmpty
                        >
                            <MenuItem value="light">Light</MenuItem>
                            <MenuItem value="replot">Replot</MenuItem>
                        </Select>
                    </FormControl>
                    {drawMode === "light" && <Scatter data={contourPlotData} options={contourPlotOptions} />}

                </Box>
            </Stack>
        </>
    );
};

export default CellImageGrid;
