import React, { useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
    Box, Grid, Typography, TextField, Button, MenuItem, Select, FormControl, InputLabel, CircularProgress, IconButton
} from "@mui/material";
import { styled } from '@mui/system';
import axios from "axios";
import { settings } from "../settings";
import ArrowBackIosIcon from '@mui/icons-material/ArrowBackIos';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';

const url_prefix = settings.url_prefix;

const CustomTextField = styled(TextField)({
    '& .MuiOutlinedInput-root': {
        '& fieldset': {
            borderColor: 'black',
        },
        '&:hover fieldset': {
            borderColor: 'black',
        },
        '&.Mui-focused fieldset': {
            borderColor: 'black',
        },
    },
});

const Extraction: React.FC = () => {
    const [searchParams] = useSearchParams();
    const fileName = searchParams.get("file_name") || "";
    const [mode, setMode] = useState("dual");
    const [param1, setParam1] = useState(100);
    const [imageSize, setImageSize] = useState(200);
    const [isLoading, setIsLoading] = useState(false);
    const [numImages, setNumImages] = useState(0);
    const [currentImage, setCurrentImage] = useState(0);

    const handleExtractCells = async () => {
        setIsLoading(true);
        try {
            const response = await axios.get(`${url_prefix}/cell_extraction/${fileName}/${mode}`, {
                params: {
                    param1,
                    image_size: imageSize,
                },
            });
            setNumImages(response.data.num_images);
            setCurrentImage(0);
        } catch (error) {
            console.error("Failed to extract cells", error);
        } finally {
            setIsLoading(false);
        }
    };

    const handlePreviousImage = () => {
        setCurrentImage((prev) => (prev > 0 ? prev - 1 : prev));
    };

    const handleNextImage = () => {
        setCurrentImage((prev) => (prev < numImages - 1 ? prev + 1 : prev));
    };

    return (
        <Box>
            <Grid container spacing={2} alignItems="center" justifyContent="center">
                <Grid item xs={12} md={numImages > 0 ? 4 : 8}>
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
                    <CustomTextField
                        label="Param1"
                        type="number"
                        fullWidth
                        margin="normal"
                        value={param1}
                        onChange={(e) => setParam1(Number(e.target.value))}
                    />
                    <CustomTextField
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
                {numImages > 0 && (
                    <Grid item xs={12} md={8}>
                        <Box display="flex" flexDirection="column" alignItems="center">
                            <Box component="img" src={`${url_prefix}/cell_extraction/ph_contours/${currentImage}`} alt={`Extracted cell ${currentImage}`} />
                            <Box display="flex" justifyContent="space-between" mt={2}>
                                <IconButton onClick={handlePreviousImage} disabled={currentImage === 0}>
                                    <ArrowBackIosIcon />
                                </IconButton>
                                <Typography variant="body1">{currentImage + 1} / {numImages}</Typography>
                                <IconButton onClick={handleNextImage} disabled={currentImage === numImages - 1}>
                                    <ArrowForwardIosIcon />
                                </IconButton>
                            </Box>
                        </Box>
                    </Grid>
                )}
            </Grid>
        </Box>
    );
};

export default Extraction;
