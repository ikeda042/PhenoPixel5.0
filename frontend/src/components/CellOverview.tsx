import React, { useEffect, useState, useRef } from "react";
import axios from "axios";
import {
    Stack, Select, MenuItem, FormControl, InputLabel, Grid, Box, Button, Typography, TextField, FormControlLabel, Checkbox, Breadcrumbs, Link,
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
import VarEngine from "./VarEngine"; 

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

type ImageState = {
    ph: string;
    fluo?: string | null;
    replot?: string;
    distribution?: string;
    path?: string;
    prediction?: string;
    cloud_points?: string;
    cloud_points_ph?: string;

};
const url_prefix = settings.url_prefix;

const CellImageGrid: React.FC = () => {
    const [searchParams] = useSearchParams();
    const db_name = searchParams.get('db_name') ?? "test_database.db";
    const cell_number = searchParams.get('cell') ?? "1";
    const init_draw_mode = searchParams.get('init_draw_mode') ?? "light";
    const [cellIds, setCellIds] = useState<string[]>([]);
    const [images, setImages] = useState<{ [key: string]: ImageState }>({});
    const [selectedLabel, setSelectedLabel] = useState<string>("74");
    const [manualLabel, setManualLabel] = useState<string>("");
    const [currentIndex, setCurrentIndex] = useState<number>(parseInt(cell_number) - 1);
    const [drawContour, setDrawContour] = useState<boolean>(true);
    const [drawScaleBar, setDrawScaleBar] = useState<boolean>(false);
    const [autoPlay, setAutoPlay] = useState<boolean>(false);
    const [brightnessFactor, setBrightnessFactor] = useState<number>(1.0);
    const [contourData, setContourData] = useState<number[][]>([]);
    const [contourDataT1, setContourDataT1] = useState<number[][]>([]);
    const [imageDimensions, setImageDimensions] = useState<{ width: number, height: number } | null>(null);
    const [drawMode, setDrawMode] = useState<string>(init_draw_mode);
    const [fitDegree, setFitDegree] = useState<number>(4);
    const [isLoading, setIsLoading] = useState(false);
    const [engineMode, setEngineMode] = useState<string>("None");

    const lastCallTimeRef = useRef<number | null>(null);

    const debounce = (func: () => void, wait: number) => {
        return () => {
            const now = new Date().getTime();
            if (lastCallTimeRef.current === null || (now - lastCallTimeRef.current) > wait) {
                lastCallTimeRef.current = now;
                func();
            }
        };
    };

    useEffect(() => {
        const fetchCellIds = async () => {
            const response = await axios.get(`${url_prefix}/cells/${db_name}/${selectedLabel}`);
            const ids = response.data.map((cell: { cell_id: string }) => cell.cell_id);
            setCellIds(ids);
        };

        fetchCellIds();
    }, [db_name, selectedLabel]);



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
                let fluoImage: string | null = null;
                if (!db_name.includes("single_layer")) {
                    fluoImage = await fetchImage('fluo_image', brightnessFactor);
                }

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
                fetchContourT1(cellId);
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
                    const response = await axios.get(`${url_prefix}/cells/${db_name}/${cellId}/label`);
                    const label = response.data.toString();
                    console.log(`Initial label for cell ${cellId}: ${label}`);
                    setManualLabel(label);
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

    const fetchContourT1 = async (cellId: string) => {
        try {
            const response = await axios.get(`${url_prefix}/cell_ai/${db_name}/${cellId}/plot_data`);
            setContourDataT1(response.data);
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
        if (drawMode === "distribution" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.distribution) {
                fetchDistributionImage(cellId, db_name, selectedLabel);
            }
        }
    }
    , [drawMode, cellIds, currentIndex, selectedLabel]);

    useEffect(() => {
        if (drawMode === "path" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.path) {
                fetchPeakPath(cellId, db_name, fitDegree);
            }
        }
    }, [drawMode, cellIds, currentIndex, fitDegree]);

    const handleNext = debounce(() => {
        setCurrentIndex((prevIndex) => (prevIndex + 1) % cellIds.length);
    }, 500);

    const handlePrev = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1 + cellIds.length) % cellIds.length);
    };

    const handleLabelChange = (event: SelectChangeEvent<string>) => {
        setSelectedLabel(event.target.value);
    };

    const handleCellLabelChange = async (event: SelectChangeEvent<string>) => {
        const newLabel = event.target.value;
        const cellId = cellIds[currentIndex];

        try {
            await axios.patch(`${url_prefix}/cells/${db_name}/${cellId}/${newLabel}`);
            setManualLabel(newLabel);
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

    const handleAutoPlayChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setAutoPlay(e.target.checked);
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

    // キーボードイベントの設定
    useEffect(() => {
        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'Enter') {
                handleNext();  // EnterキーでNextを押す
            } else if (['1', '2', '3', 'n'].includes(event.key)) {
                let newLabel = event.key;
                if (newLabel === 'n') {
                    newLabel = "1000";  // 'n' は "1000" に対応
                }
                setManualLabel(newLabel);
                handleCellLabelChange({ target: { value: newLabel } } as SelectChangeEvent<string>);
            }
        };

        window.addEventListener('keydown', handleKeyDown);

        return () => {
            window.removeEventListener('keydown', handleKeyDown);
        };
    }, [cellIds, currentIndex]);  // cellIds と currentIndex を依存に追加

    const handleEngineModeChange = (event: SelectChangeEvent<string>) => {
        setEngineMode(event.target.value);
    };

    type EngineName = 'None' | 'MorphoEngine 2.0' | 'MedianEngine' | 'MeanEngine' | 'HeatmapEngine' | 'VarEngine';

    const engineLogos: Record<EngineName, string> = {
        None: 'path_to_none_logo.png',
        'MorphoEngine 2.0': '/logo_tp.png',
        'MedianEngine': '/logo_dots.png',
        'MeanEngine': '/logo_circular.png',
        'VarEngine': '/var_logo.png',
        'HeatmapEngine': '/logo_heatmap.png',
    };

    useEffect(() => {
        let autoNextInterval: NodeJS.Timeout | undefined;

        if (autoPlay) {
            autoNextInterval = setInterval(handleNext, 3000);  // 3秒ごとにNextボタンを自動で押す
        }

        return () => {
            if (autoNextInterval) clearInterval(autoNextInterval);
        };
    }, [autoPlay]);  // autoPlayが変わるたびにエフェクトを再実行

    const contourPlotData = {
        datasets: [
            {
                label: 'Contour',
                data: contourData.map(point => ({ x: point[0], y: point[1] })),
                borderColor: 'lime',
                backgroundColor: 'lime',
                pointRadius: 1,
            },
            ...(drawMode === "t1contour" ? [{
                label: 'Model T1',
                data: contourDataT1.map(point => ({ x: point[0], y: point[1] })),
                borderColor: 'red',
                backgroundColor: 'red',
                pointRadius: 1,
            }] : [])
        ]
    };

    const fetchPredictionImage = async (cellId: string, dbName: string) => {
        try {
            const response = await axios.get(`${url_prefix}/cell_ai/${dbName}/${cellId}`, { responseType: 'blob' });
            const predictionImageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], prediction: predictionImageUrl }
            }));
        } catch (error) {
            console.error("Error fetching prediction image:", error);
        }
    };

    const fetchDistributionImage = async (cellId: string, dbName: string, label: string) => {
        try {
            const response = await axios.get(
                `${url_prefix}/cells/${dbName}/${cellId}/distribution`, 
                { responseType: 'blob' }
            );
            const distributionImageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { 
                    ...prevImages[cellId], 
                    distribution: distributionImageUrl 
                }
            }));
        } catch (error) {
            console.error("Error fetching distribution image:", error);
        }
    };

    useEffect(() => {
        if (drawMode === "prediction" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.prediction) {
                fetchPredictionImage(cellId, db_name);
            }
        }
    }, [drawMode, cellIds, currentIndex]);


    const fetch3DImage = async (cellId: string, dbName: string) => {
        try {
            const response = await axios.get(`${url_prefix}/cells/${dbName}/${cellId}/3d`, { responseType: 'blob' });
            const imageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], cloud_points: imageUrl }
            }));
        } catch (error) {
            console.error("Error fetching 3D image:", error);
        }
    };

    useEffect(() => {
        if (drawMode === "cloud_points" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.cloud_points) {
                fetch3DImage(cellId, db_name);
            }
        }
    }, [drawMode, cellIds, currentIndex]);

    const fetch3DPhImage = async (cellId: string, dbName: string) => {
        try {
            const response = await axios.get(`${url_prefix}/cells/${dbName}/${cellId}/3d-ph`, { responseType: 'blob' });
            const imageUrl = URL.createObjectURL(response.data);
            setImages((prevImages) => ({
                ...prevImages,
                [cellId]: { ...prevImages[cellId], cloud_points_ph: imageUrl }
            }));
        } catch (error) {
            console.error("Error fetching 3D image:", error);
        }
    };

    useEffect(() => {
        if (drawMode === "cloud_points_ph" && cellIds.length > 0) {
            const cellId = cellIds[currentIndex];
            if (!images[cellId]?.cloud_points_ph) {
                fetch3DPhImage(cellId, db_name);
            }
        }
    }, [drawMode, cellIds, currentIndex]);

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
                            value={selectedLabel}
                            onChange={handleLabelChange}
                        >
                            <MenuItem value="74">All</MenuItem>
                            <MenuItem value="1000">N/A</MenuItem>
                            <MenuItem value="1">1</MenuItem>
                            <MenuItem value="2">2</MenuItem>
                            <MenuItem value="3">3</MenuItem>
                        </Select>
                    </FormControl>
                    <Box mt={2}>
                        <Grid container spacing={2} alignItems="center">
                            <Grid item xs={2}>
                                <FormControlLabel
                                    control={<Checkbox checked={drawContour} onChange={handleContourChange} style={{ color: "black" }} />}
                                    label="Contour"
                                    style={{ color: "black" }}
                                />
                            </Grid>
                            <Grid item xs={2}>
                                <FormControlLabel
                                    control={<Checkbox checked={drawScaleBar} onChange={handleScaleBarChange} style={{ color: "black" }} />}
                                    label="Scale"
                                    style={{ color: "black" }}
                                />
                            </Grid>
                            <Grid item xs={2}>
                                <FormControlLabel
                                    control={<Checkbox checked={autoPlay} onChange={handleAutoPlayChange} style={{ color: "black" }} />}
                                    label="Auto"
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
                                    <InputLabel id="manual-label-select-label">Manual Label</InputLabel>
                                    <Select
                                        labelId="manual-label-select-label"
                                        value={manualLabel}
                                        onChange={handleCellLabelChange}
                                    >
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
                            {cellIds.length > 0 ? `Cell ${currentIndex + 1} of ${cellIds.length}` : `Cell ${currentIndex} of ${cellIds.length}`} / ({cellIds[currentIndex]})
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
                            {images[cellIds[currentIndex]] && images[cellIds[currentIndex]].fluo ? (
                                <img src={images[cellIds[currentIndex]].fluo as string} alt={`Cell ${cellIds[currentIndex]} Fluo`} style={{ width: "100%" }} />
                            ) : (
                                db_name.includes("single_layer") ? <Box
                                    sx={{
                                        display: 'flex',
                                        justifyContent: 'center',
                                        alignItems: 'center',
                                        height: '100%'
                                    }}
                                >
                                    <Typography variant="h5">Single layer mode.</Typography>
                                    <img src="/logo_dots.png" alt="Morpho Engine is off" style={{ maxWidth: '15%', maxHeight: '15%' }} />
                                </Box> : <div>Not available</div>
                            )}
                        </Grid>
                    </Grid>
                </Box>
                <Box sx={{ width: 420, height: 420, marginLeft: 2 }}>
                    {drawMode === "distribution" && (
                        <Grid container spacing={2}>
                            <Grid item xs={12}>
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
                                        <MenuItem value="distribution">Distribution</MenuItem>
                                        <MenuItem value="path">Peak-path</MenuItem>
                                        <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                        <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                        <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                        <MenuItem value="cloud_points_ph">3D PH</MenuItem>
                                    </Select>
                                </FormControl>
                            </Grid>
                        </Grid>
                    )}
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
                                        <MenuItem value="distribution">Distribution</MenuItem>
                                        <MenuItem value="path">Peak-path</MenuItem>
                                        <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                        <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                        <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                        <MenuItem value="cloud_points_ph">3D PH</MenuItem>
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
                                <MenuItem value="distribution">Distribution</MenuItem>
                                <MenuItem value="path">Peak-path</MenuItem>
                                <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                <MenuItem value="cloud_points_ph">3D PH</MenuItem>


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
                                        <MenuItem value="distribution">Distribution</MenuItem>
                                        <MenuItem value="path">Peak-path</MenuItem>
                                        <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                        <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                        <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                        <MenuItem value="cloud_points_ph">3D PH</MenuItem>
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
                    {
                        drawMode === "prediction" && (
                            <Grid container spacing={2}>
                                <Grid item xs={12}>
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
                                            <MenuItem value="distribution">Distribution</MenuItem>
                                            <MenuItem value="path">Peak-path</MenuItem>
                                            <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                            <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                            <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                            <MenuItem value="cloud_points_ph">3D PH</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                            </Grid>
                        )
                    }
                    {
                        drawMode === "t1contour" && (
                            <Grid container spacing={2}>
                                <Grid item xs={12}>
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
                                            <MenuItem value="distribution">Distribution</MenuItem>
                                            <MenuItem value="path">Peak-path</MenuItem>
                                            <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                            <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                            <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                            <MenuItem value="cloud_points_ph">3D PH</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                            </Grid>
                        )
                    }
                    {
                        drawMode === "cloud_points" && (
                            <Grid container spacing={2}>
                                <Grid item xs={12}>
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
                                            <MenuItem value="distribution">Distribution</MenuItem>
                                            <MenuItem value="path">Peak-path</MenuItem>
                                            <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                            <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                            <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                            <MenuItem value="cloud_points_ph">3D PH</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                            </Grid>
                        )
                    }
                    {
                        drawMode === "cloud_points_ph" && (
                            <Grid container spacing={2}>
                                <Grid item xs={12}>
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
                                            <MenuItem value="distribution">Distribution</MenuItem>
                                            <MenuItem value="path">Peak-path</MenuItem>
                                            <MenuItem value="t1contour">Light+Model T1</MenuItem>
                                            <MenuItem value="prediction">Model T1(Torch GPU)</MenuItem>
                                            <MenuItem value="cloud_points">3D Fluo</MenuItem>
                                            <MenuItem value="cloud_points_ph">3D PH</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>
                            </Grid>
                        )
                    }

                    <Box mt={2}>
                        {drawMode === "light" && <Scatter data={contourPlotData} options={contourPlotOptions} />}
                        {drawMode === "replot" && images[cellIds[currentIndex]]?.replot && (
                            <img src={images[cellIds[currentIndex]]?.replot} alt={`Cell ${cellIds[currentIndex]} Replot`} style={{ width: "100%" }} />
                        )}
                         {drawMode === "distribution" && images[cellIds[currentIndex]]?.distribution && (
                        <img src={images[cellIds[currentIndex]]?.distribution} alt={`Cell ${cellIds[currentIndex]} Distribution`} style={{ width: "100%" }} />
                         )}
                        {drawMode === "path" && isLoading ? (
                            <Box display="flex" justifyContent="center" alignItems="center" style={{ height: 400 }}>
                                <Spinner />
                            </Box>
                        ) : drawMode === "path" && images[cellIds[currentIndex]]?.path && (
                            <img src={images[cellIds[currentIndex]]?.path} alt={`Cell ${cellIds[currentIndex]} Path`} style={{ width: "100%" }} />
                        )}
                        {drawMode === "prediction" && images[cellIds[currentIndex]]?.prediction && (
                            <img src={images[cellIds[currentIndex]]?.prediction} alt={`Cell ${cellIds[currentIndex]} Prediction`} style={{ width: "100%" }} />
                        )}
                        {drawMode === "t1contour" && <Scatter data={contourPlotData} options={contourPlotOptions} />}
                        {drawMode === "cloud_points" && images[cellIds[currentIndex]]?.cloud_points && (
                            <img src={images[cellIds[currentIndex]]?.cloud_points} alt={`Cell ${cellIds[currentIndex]} 3D`} style={{ width: "100%" }} />
                        )}
                        {drawMode === "cloud_points_ph" && images[cellIds[currentIndex]]?.cloud_points_ph && (
                            <img src={images[cellIds[currentIndex]]?.cloud_points_ph} alt={`Cell ${cellIds[currentIndex]} 3D`} style={{ width: "100%" }} />
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
                                    } else if (engineName === 'MedianEngine') {
                                        displayText = "MedianEngine";
                                    } else if (engineName === 'MeanEngine') {
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
                                        {engine}
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
                    {engineMode === "MedianEngine" && (
                        <Box mt={6}>
                            <MedianEngine dbName={db_name} label={selectedLabel} cellId={cellIds[currentIndex]} />
                        </Box>)}
                    {engineMode === "MeanEngine" && (
                        <Box mt={6}>
                            <MeanEngine dbName={db_name} label={selectedLabel} cellId={cellIds[currentIndex]} />
                        </Box>)}
                    {engineMode === "VarEngine" && (
                        <Box mt={6}>
                            <VarEngine dbName={db_name} label={selectedLabel} cellId={cellIds[currentIndex]} />
                        </Box>
                    )}
                    {engineMode === "HeatmapEngine" && (
                        <Box mt={6}>
                            <HeatmapEngine dbName={db_name} label={selectedLabel} cellId={cellIds[currentIndex]} degree={fitDegree} />
                        </Box>)}
                  
                </Box>
            </Stack>
        </>
    );
};

export default CellImageGrid;