import React, { useEffect, useState } from "react";
import axios from "axios";
import {
    Stack, Select, MenuItem, FormControl, InputLabel, Grid, Box, Button, Typography, TextField, FormControlLabel, Checkbox, Breadcrumbs, Link,
    Menu
} from "@mui/material";
import { SelectChangeEvent } from "@mui/material/Select";
import { Scatter } from 'react-chartjs-2';
import { ChartOptions } from 'chart.js';
import Spinner from './Spinner';
import CellMorphologyTable from "./CellMorphoTable";
import { settings } from "../settings";
import { useSearchParams } from 'react-router-dom';
import MedianEngine from "./MedianEngine";
import MeanEngine from "./MeanEngine";
import HeatmapEngine from "./HeatmapEngine";

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

const url_prefix = settings.url_prefix;

const CellImageGrid: React.FC = () => {
    const [cellIds, setCellIds] = useState<string[]>([]);
    const [images, setImages] = useState<{ [key: string]: { ph: string, fluo: string, replot?: string, path?: string } }>({});
    const [label, setLabel] = useState<string>("1");
    const [currentIndex, setCurrentIndex] = useState<number>(0);
    const [drawContour, setDrawContour] = useState<boolean>(false);
    const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
    const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);
    const [contourData, setContourData] = useState<number[][]>([]);
    const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number } | null>(null);
    const [drawMode, setDrawMode] = useState<string>("light");
    const [fitDegree, setFitDegree] = useState<number>(4);
    const [isLoading, setIsLoading] = useState(false);
    const [engineMode, setEngineMode] = useState<string>("None");
    const [searchParams] = useSearchParams();
    const db_name = searchParams.get('db_name') ?? "test_database.db";

    useEffect(() => {
        const fetchCellIds = async () => {
            const response = await axios.get(`${url_prefix}/cells/${db_name}/${label}`);
            const ids = response.data.map((cell: { cell_id: string }) => cell.cell_id);
            setCellIds(ids);
        };

        fetchCellIds();
    }, [db_name, label]);

    useEffect(() => {
        const fetchImages = async (cellId: string) => {
            try {
                const fetchImage = async (type: 'ph_image' | 'fluo_image', brightnessFactor: number = 1.0) => {
                    let url = `${url_prefix}/cells/${cellId}/${db_name}/${drawContour}/${drawScaleBar}/`;
                    if (type === 'fluo_image') {
                        url += `${type}?brightness_factor=${brightnessFactor}`;
                    } else {
                        url += `${type}`;
                    }
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

                const phImage = await fetchImage('ph_image');
                const fluoImage = await fetchImage('fluo_image', brightnessFactor);

                return { ph: phImage, fluo: fluoImage };
            } catch (error) {
                console.error("Error fetching images: FE", error);
                return null;
            }
        };

        const handleFetchImages = async (cellId: string) => {
            const newImages = await fetchImages(cellId);
            if (newImages) {
                setImages((prevImages) => ({
                    ...prevImages,
                    [cellId]: { ...prevImages[cellId], ...newImages }
                }));
                fetchContour(cellId);
            }
        };

        if (cellIds.length > 0) {
            handleFetchImages(cellIds[currentIndex]);
        }
    }, [cellIds, currentIndex, db_name, drawContour, drawScaleBar, brightnessFactor]);

    useEffect(() => {
        const fetchInitialLabel = async () => {
            if (cellIds.length > 0) {
                const cellId = cellIds[currentIndex];
                try {
                    const response = await axios.get(`${url_prefix}/${db_name}/${cellId}/label`);
                    setLabel(response.data);
                } catch (error) {
                    console.error("Error fetching initial label:", error);
                }
            }
        };

        fetchInitialLabel();
    }, [cellIds, currentIndex, db_name]);

    const fetchContour = async (cellId: string) => {
        try {
            const response = await axios.get(`${url_prefix}/cells/${cellId}/contour/raw?db_name=${db_name}`);
            setContourData(response.data.contour);
        } catch (error) {
            console.error("Error fetching contour data:", error);
        }
    };

    const fetchReplotImage = async (cellId: string, dbName: string, fitDegree: number) => {
        try {
            const response = await axios.get(`${url_prefix}/cells/${cellId}/${dbName}/replot?degree=${fitDegree}`, { responseType: 'blob' });
            const replotImageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], replot: replotImageUrl }
            }));
        } catch (error) {
            console.error("Error fetching replot image:", error);
        }
    };

    const fetchPeakPath = async (cellId: string, dbName: string, fitDegree: number) => {
        setIsLoading(true);
        try {
            const response = await axios.get(`${url_prefix}/cells/${cellId}/${dbName}/path?degree=${fitDegree}`, { responseType: 'blob' });
            const pathImageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], path: pathImageUrl }
            }));
        } catch (error) {
            console.error("Error fetching peak path:", error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (drawMode === "replot" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.replot) {
                fetchReplotImage(cellId, db_name, fitDegree);
            }
        }
    }, [drawMode, cellIds, currentIndex, fitDegree]);

    useEffect(() => {
        if (drawMode === "path" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.path) {
                fetchPeakPath(cellId, db_name, fitDegree);
            }
        }
    }, [drawMode, cellIds, currentIndex, fitDegree]);

    const handleNext = () => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % cellIds.length);
    };

    const handlePrev = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1 + cellIds.length) % cellIds.length);
    };

    const handleLabelChange = (event: SelectChangeEvent<string>) => {
        setLabel(event.target.value);
    };

    const handleCellLabelChange = async (event: SelectChangeEvent<string>) => {
        const newLabel = event.target.value;
        const cellId = cellIds[currentIndex];

        try {
            await axios.patch(`${url_prefix}/${db_name}/${cellId}/${newLabel}`);
            setLabel(newLabel); // Update the local state with the new label
        } catch (error) {
            console.error("Error updating cell label:", error);
        }
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

    const handleFitDegreeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFitDegree(parseInt(e.target.value));
    };

    type EngineName = 'None' | 'MorphoEngine 2.0' | 'MorphoEngine 3.0' | 'MorphoEngine 4.0' | 'MorphoEngine 5.0';

    const engineLogos: Record<EngineName, string> = {
        None: 'path_to_none_logo.png',
        'MorphoEngine 2.0': '/logo_tp.png',
        'MorphoEngine 3.0': '/logo_dots.png',
        'MorphoEngine 4.0': '/logo_circular.png',
        'MorphoEngine 5.0': '/logo_heatmap.png',
    };

    const handleEngineModeChange = (event: SelectChangeEvent<string>) => {
        setEngineMode(event.target.value);
    };

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
            <Box>
                <Breadcrumbs aria-label="breadcrumb">
                    <Link underline="hover" color="inherit" href="/">
                        Top
                    </Link>
                    <Link underline="hover" color="inherit" href="/dbconsole">
                        Database Console
                    </Link>
                    <Typography color="text.primary">{db_name}</Typography>
                </Breadcrumbs>
            </Box>
            <Stack direction="row" spacing={2} sx={{ marginTop: 8 }}>
                <Box sx={{ width: 580, height: 420, marginLeft: 2 }}>
                    <FormControl fullWidth>
                        <InputLabel id="label-select-label">Label</InputLabel>
                        <Select
                            labelId="label-select-label"
                            value={label}
                            onChange={handleLabelChange}
                        >
                            <MenuItem value="1000">N/A</MenuItem>
                            <MenuItem value="1">1</MenuItem>
                            <MenuItem value="2">2</MenuItem>
                            <MenuItem value="3">3</MenuItem>
                        </Select>
                    </FormControl>
                    <Box mt={2}>
                        <Grid container spacing={2} alignItems="center">
                            <Grid item xs={3}>
                                <FormControlLabel
                                    control={<Checkbox checked={drawContour} onChange={handleContourChange} style={{ color: "black" }} />}
                                    label="Contour"
                                    style={{ color: "black" }}
                                />
                            </Grid>
                            <Grid item xs={3}>
                                <FormControlLabel
                                    control={<Checkbox checked={drawScaleBar} onChange={handleScaleBarChange} style={{ color: "black" }} />}
                                    label="Scale Bar"
                                    style={{ color: "black" }}
                                />
                            </Grid>
                            <Grid item xs={3}>
                                <TextField
                                    label="Brightness Factor"
                                    type="number"
                                    value={brightnessFactor}
                                    onChange={handleBrightnessChange}
                                    InputProps={{
                                        inputProps: { min: 0.1, step: 0.1 },
                                        onWheel: handleWheel,
                                        autoComplete: "off"
                                    }}
                                />
                            </Grid>
                            <Grid item xs={3}>
                                <FormControl fullWidth>
                                    <InputLabel id="cell-label-select-label">Manual Label</InputLabel>
                                    <Select
                                        labelId="cell-label-select-label"
                                        value={label}
                                        onChange={handleCellLabelChange}
                                    >
                                        <MenuItem value="74">None</MenuItem>
                                        <MenuItem value="1000">N/A</MenuItem>
                                        <MenuItem value="1">1</MenuItem>
                                        <MenuItem value="2">2</MenuItem>
                                        <MenuItem value="3">3</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                        </Grid>
                    </Box>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
                        <Button variant="contained" color="primary" onClick={handlePrev} disabled={cellIds.length === 0} style={{ backgroundColor: "black", minWidth: "100px" }}>
                            Prev
                        </Button>
                        <Typography variant="h6">
                            {cellIds.length > 0 ? `Cell ${currentIndex + 1} of ${cellIds.length}` : `Cell ${currentIndex} of ${cellIds.length}`}
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
                <Box sx={{ width: 420, height: 420, marginLeft: 2 }}>
                    {drawMode === "replot" && (
                        <Grid container spacing={2}>
                            <Grid item xs={8}>
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
                                        <MenuItem value="path">Peak-path</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={4}>
                                <TextField
                                    label="Polyfit Degree"
                                    type="number"
                                    value={fitDegree}
                                    onChange={handleFitDegreeChange}
                                    InputProps={{
                                        inputProps: { min: 0, step: 1 },
                                        onWheel: handleWheel,
                                        autoComplete: "off"
                                    }}
                                />
                            </Grid>
                        </Grid>
                    )}
                    {drawMode === "light" && (
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
                                <MenuItem value="path">Peak-path</MenuItem>
                            </Select>
                        </FormControl>
                    )}
                    {drawMode === "path" && (
                        <Grid container spacing={2}>
                            <Grid item xs={8}>
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
                                        <MenuItem value="path">Peak-path</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                            <Grid item xs={4}>
                                <TextField
                                    label="Polyfit Degree"
                                    type="number"
                                    value={fitDegree}
                                    onChange={handleFitDegreeChange}
                                    InputProps={{
                                        inputProps: { min: 0, step: 1 },
                                        onWheel: handleWheel,
                                        autoComplete: "off"
                                    }}
                                />
                            </Grid>
                        </Grid>
                    )}
                    <Box mt={2}>
                        {drawMode === "light" && <Scatter data={contourPlotData} options={contourPlotOptions} />}
                        {drawMode === "replot" && images[cellIds[currentIndex]]?.replot && (
                            <img src={images[cellIds[currentIndex]]?.replot} alt={`Cell ${cellIds[currentIndex]} Replot`} style={{ width: "100%" }} />
                        )}
                        {drawMode === "path" && isLoading ? (
                            <Box display="flex" justifyContent="center" alignItems="center" style={{ height: 400 }}>
                                <Spinner />
                            </Box>
                        ) : drawMode === "path" && images[cellIds[currentIndex]]?.path && (
                            <img src={images[cellIds[currentIndex]]?.path} alt={`Cell ${cellIds[currentIndex]} Path`} style={{ width: "100%" }} />
                        )}
                    </Box>
                </Box>
                <Box sx={{ width: 350, height: 420, marginLeft: 2 }}>
                    <FormControl fullWidth>
                        <InputLabel id="engine-select-label">MorphoEngine</InputLabel>
                        <Select
                            labelId="engine-select-label"
                            id="engine-select"
                            value={engineMode}
                            onChange={handleEngineModeChange}
                            renderValue={(selected: string) => {
                                if (selected === 'None') {
                                    return "None";
                                } else {
                                    const engineName = selected as EngineName;
                                    let displayText: string = engineName;
                                    if (engineName === 'MorphoEngine 2.0') {
                                        displayText = engineName;
                                    } else if (engineName === 'MorphoEngine 3.0') {
                                        displayText = "MedianEngine";
                                    } else if (engineName === 'MorphoEngine 4.0') {
                                        displayText = "MeanEngine";
                                    }
                                    return (
                                        <Box display="flex" alignItems="center">
                                            {engineName !== 'None' && <img src={engineLogos[engineName]} alt="" style={{ width: 24, height: 24, marginRight: 8 }} />}
                                            {displayText}
                                        </Box>
                                    );
                                }
                            }}
                        >
                            {Object.entries(engineLogos).map(([engine, logoPath]) => (
                                <MenuItem key={engine} value={engine}>
                                    <Box display="flex" alignItems="center">
                                        {engine !== 'None' && <img src={logoPath} alt="" style={{ width: 24, height: 24, marginRight: 8 }} />}
                                        {engine === 'None' && <span>None</span>}
                                        {engine === 'MorphoEngine 2.0' && <span>{engine}</span>}
                                        {engine === 'MorphoEngine 3.0' && <span>MedianEngine</span>}
                                        {engine === 'MorphoEngine 4.0' && <span>MeanEngine</span>}
                                        {engine === 'MorphoEngine 5.0' && <span>HeatmapEngine</span>}
                                    </Box>
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                    {engineMode === "None" && (
                        <Box
                            sx={{
                                display: 'flex',
                                justifyContent: 'center',
                                alignItems: 'center',
                                height: '100%'
                            }}
                        >
                            <Typography variant="h5">MorphoEngine is off.</Typography>
                            <img src="/logo_tp.png" alt="Morpho Engine is off" style={{ maxWidth: '15%', maxHeight: '15%' }} />
                        </Box>
                    )}
                    {engineMode === "MorphoEngine 2.0" && (
                        <Box mt={2}>
                            <CellMorphologyTable cellId={cellIds[currentIndex]} db_name={db_name} polyfitDegree={fitDegree} />
                        </Box>)}
                    {engineMode === "MorphoEngine 3.0" && (
                        <Box mt={6}>
                            <MedianEngine dbName={db_name} label={label} cellId={cellIds[currentIndex]} />
                        </Box>)}
                    {engineMode === "MorphoEngine 4.0" && (
                        <Box mt={6}>
                            <MeanEngine dbName={db_name} label={label} cellId={cellIds[currentIndex]} />
                        </Box>)}
                    {engineMode === "MorphoEngine 5.0" && (
                        <Box mt={6}>
                            <HeatmapEngine dbName={db_name} label={label} cellId={cellIds[currentIndex]} degree={fitDegree} />
                        </Box>)}
                </Box>
            </Stack>
        </>
    );
};

export default CellImageGrid;
