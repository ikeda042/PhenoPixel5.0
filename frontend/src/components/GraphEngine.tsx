import React, { useState } from "react";
import { Box, Button, Container, Typography, Select, MenuItem, FormControl, InputLabel, SelectChangeEvent, CircularProgress } from "@mui/material";
import { settings } from "../settings";
import { Breadcrumbs, Link } from "@mui/material";

const url_prefix = settings.url_prefix;

const GraphEngine: React.FC = () => {
    const [mode, setMode] = useState("heatmap_abs");
    const [file, setFile] = useState<File | null>(null);
    const [imageSrc, setImageSrc] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);

    const handleModeChange = (event: SelectChangeEvent<string>) => {
        setMode(event.target.value as string);
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files) {
            setFile(event.target.files[0]);
        }
    };

    const handleGenerateGraph = async () => {
        if (!file) {
            alert("Please select a CSV file.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        setIsLoading(true);

        try {
            let requestUrl;
            if (mode === "mcpr") {
                requestUrl = `${url_prefix}/graph_engine/mcpr?blank_index=2&timespan_sec=180`;
            } else {
                requestUrl = `${url_prefix}/graph_engine/${mode}`;
            }

            const response = await fetch(requestUrl, {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                setImageSrc(imageUrl);
            } else {
                alert("Failed to generate graph.");
            }
        } catch (error) {
            console.error("Error generating graph:", error);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <Container>
            <Box mb={3}>
                <Breadcrumbs aria-label="breadcrumb">
                    <Link underline="hover" color="inherit" href="/">
                        Top
                    </Link>
                    <Typography color="text.primary">Graph engine</Typography>
                </Breadcrumbs>
            </Box>
            <FormControl fullWidth>
                <InputLabel id="select-label">Graph Mode</InputLabel>
                <Select
                    labelId="select-label"
                    value={mode}
                    onChange={handleModeChange}
                    style={{ color: isLoading ? "black" : undefined }}
                >
                    <MenuItem value="heatmap_abs">Heatmap abs.</MenuItem>
                    <MenuItem value="heatmap_rel">Heatmap rel.</MenuItem>
                    <MenuItem value="mcpr">MCPR</MenuItem> 
                </Select>
            </FormControl>
            <Box my={2}>
                <input type="file" accept=".csv" onChange={handleFileChange} />
            </Box>
            <Button
                variant="contained"
                color="success"
                onClick={handleGenerateGraph}
                disabled={isLoading}
                style={{ opacity: isLoading ? 0.6 : 1 }}
            >
                {isLoading ? <CircularProgress size={24} sx={{ color: '#000000' }} /> : "Generate Graph"}
            </Button>

            {imageSrc && (
                <Box mt={4}>
                    <Typography variant="h6">Generated Graph:</Typography>
                    <img
                        src={imageSrc}
                        alt="Generated Graph"
                        style={{
                            maxWidth: "100%",
                            maxHeight: "80vh",
                            display: "block",
                            margin: "0 auto"
                        }}
                    />
                </Box>
            )}
        </Container>
    );
};

export default GraphEngine;
