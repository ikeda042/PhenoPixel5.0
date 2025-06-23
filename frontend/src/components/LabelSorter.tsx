import React, { useEffect, useState } from "react";
import {
  Container,
  Grid,
  Typography,
  Box,
  Breadcrumbs,
  Link,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import axios from "axios";
import { useSearchParams } from "react-router-dom";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;
const resizeFactor = 0.5;

const LabelSorter: React.FC = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name") ?? "";

  const [naCells, setNaCells] = useState<string[]>([]);
  const [selectedLabel, setSelectedLabel] = useState<string>("1");
  const [labelCells, setLabelCells] = useState<string[]>([]);
  const [channel, setChannel] = useState<"ph" | "fluo1" | "fluo2">("ph");
  const [images, setImages] = useState<{ [key: string]: string }>({});

  const labelOptions = ["1", "2", "3", "4"];

  useEffect(() => {
    const fetchNaCells = async () => {
      try {
        const naRes = await axios.get(`${url_prefix}/cells/${dbName}/1000`);
        setNaCells(naRes.data.map((c: { cell_id: string }) => c.cell_id));
      } catch (error) {
        console.error("Failed to fetch N/A cell ids", error);
      }
    };
    if (dbName) fetchNaCells();
  }, [dbName]);

  useEffect(() => {
    const fetchLabelCells = async () => {
      try {
        const res = await axios.get(
          `${url_prefix}/cells/${dbName}/${selectedLabel}`
        );
        setLabelCells(res.data.map((c: { cell_id: string }) => c.cell_id));
      } catch (error) {
        console.error("Failed to fetch label cell ids", error);
      }
    };
    if (dbName && selectedLabel) fetchLabelCells();
  }, [dbName, selectedLabel]);

  useEffect(() => {
    const fetchImages = async (cellIds: string[]) => {
      await Promise.all(
        cellIds.map(async (id) => {
          const key = `${id}_${channel}`;
          if (!images[key]) {
            try {
              const endpoint =
                channel === "ph"
                  ? "ph_image"
                  : channel === "fluo1"
                  ? "fluo_image"
                  : "fluo2_image";
              const res = await axios.get(
                `${url_prefix}/cells/${id}/${dbName}/true/false/${endpoint}?resize_factor=${resizeFactor}`,
                { responseType: "blob" }
              );
              const url = URL.createObjectURL(res.data);
              setImages((prev) => ({ ...prev, [key]: url }));
            } catch (err) {
              console.error("Failed to fetch image", err);
            }
          }
        })
      );
    };
    fetchImages(naCells);
  }, [naCells, dbName, channel, images]);

  useEffect(() => {
    const fetchImages = async (cellIds: string[]) => {
      await Promise.all(
        cellIds.map(async (id) => {
          const key = `${id}_${channel}`;
          if (!images[key]) {
            try {
              const endpoint =
                channel === "ph"
                  ? "ph_image"
                  : channel === "fluo1"
                  ? "fluo_image"
                  : "fluo2_image";
              const res = await axios.get(
                `${url_prefix}/cells/${id}/${dbName}/true/false/${endpoint}?resize_factor=${resizeFactor}`,
                { responseType: "blob" }
              );
              const url = URL.createObjectURL(res.data);
              setImages((prev) => ({ ...prev, [key]: url }));
            } catch (err) {
              console.error("Failed to fetch image", err);
            }
          }
        })
      );
    };
    fetchImages(labelCells);
  }, [labelCells, dbName, channel, images]);

  const handleClick = async (cellId: string, fromLabel: "N/A" | "selected") => {
    const newLabel = fromLabel === "N/A" ? selectedLabel : "1000";
    try {
      await axios.patch(`${url_prefix}/cells/${dbName}/${cellId}/${newLabel}`);
      if (fromLabel === "N/A") {
        setNaCells((prev) => prev.filter((id) => id !== cellId));
        setLabelCells((prev) => [...prev, cellId]);
      } else {
        setLabelCells((prev) => prev.filter((id) => id !== cellId));
        setNaCells((prev) => [...prev, cellId]);
      }
    } catch (err) {
      console.error("Failed to update label", err);
    }
  };

  const renderCells = (cellIds: string[], column: "N/A" | "selected") => (
    <Grid container spacing={1}>
      {cellIds.map((id) => (
        <Grid item xs={4} sm={3} md={2} key={id}>
          <Box
            component="img"
            src={images[`${id}_${channel}`]}
            alt={id}
            sx={{ width: "100%", cursor: "pointer" }}
            onClick={() => handleClick(id, column)}
          />
          <Typography variant="caption" display="block" align="center">
            {id}
          </Typography>
        </Grid>
      ))}
    </Grid>
  );

  return (
    <Container maxWidth={false} disableGutters>
      <Box mb={2}>
        <Breadcrumbs aria-label="breadcrumb">
          <Link underline="hover" color="inherit" href="/">
            Top
          </Link>
          <Link underline="hover" color="inherit" href="/dbconsole">
            Database Console
          </Link>
          <Typography color="text.primary">Sort labels</Typography>
        </Breadcrumbs>
      </Box>
      <Box mb={2} display="flex" alignItems="center" gap={2}>
        <Typography variant="h6">Database: {dbName}</Typography>
        <FormControl size="small" sx={{ minWidth: 100 }}>
          <InputLabel id="channel-select">channel</InputLabel>
          <Select
            labelId="channel-select"
            label="channel"
            value={channel}
            onChange={(e) => setChannel(e.target.value as "ph" | "fluo1" | "fluo2")}
          >
            <MenuItem value="ph">ph</MenuItem>
            <MenuItem value="fluo1">fluo1</MenuItem>
            <MenuItem value="fluo2">fluo2</MenuItem>
          </Select>
        </FormControl>
      </Box>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Box border={1} borderColor="divider" borderRadius={1} p={1} height="100%">
            <Typography variant="h6" gutterBottom>
              N/A
            </Typography>
            {renderCells(naCells, "N/A")}
          </Box>
        </Grid>
        <Grid item xs={12} md={6}>
          <Box border={1} borderColor="divider" borderRadius={1} p={1} height="100%">
            <Box display="flex" alignItems="center" mb={1}>
              <FormControl size="small" sx={{ minWidth: 80 }}>
                <InputLabel id="label-select">Label</InputLabel>
                <Select
                  labelId="label-select"
                  label="Label"
                  value={selectedLabel}
                  onChange={(e) => setSelectedLabel(e.target.value)}
                >
                  {labelOptions.map((opt) => (
                    <MenuItem key={opt} value={opt}>{`Label ${opt}`}</MenuItem>
                  ))}
                </Select>
              </FormControl>
              <Typography ml={2}>{labelCells.length} cells</Typography>
            </Box>
            {renderCells(labelCells, "selected")}
          </Box>
        </Grid>
      </Grid>
    </Container>
  );
};

export default LabelSorter;
