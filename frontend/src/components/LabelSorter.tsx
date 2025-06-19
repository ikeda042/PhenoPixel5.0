import React, { useEffect, useState } from "react";
import { Container, Grid, Typography, Box } from "@mui/material";
import axios from "axios";
import { useSearchParams } from "react-router-dom";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const LabelSorter: React.FC = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name") ?? "";

  const [naCells, setNaCells] = useState<string[]>([]);
  const [label1Cells, setLabel1Cells] = useState<string[]>([]);
  const [images, setImages] = useState<{ [key: string]: string }>({});

  useEffect(() => {
    const fetchCellIds = async () => {
      try {
        const naRes = await axios.get(`${url_prefix}/cells/${dbName}/1000`);
        setNaCells(naRes.data.map((c: { cell_id: string }) => c.cell_id));
        const label1Res = await axios.get(`${url_prefix}/cells/${dbName}/1`);
        setLabel1Cells(label1Res.data.map((c: { cell_id: string }) => c.cell_id));
      } catch (error) {
        console.error("Failed to fetch cell ids", error);
      }
    };
    if (dbName) fetchCellIds();
  }, [dbName]);

  useEffect(() => {
    const fetchImages = async (cellIds: string[]) => {
      await Promise.all(
        cellIds.map(async (id) => {
          if (!images[id]) {
            try {
              const res = await axios.get(
                `${url_prefix}/cells/${id}/${dbName}/true/false/ph_image`,
                { responseType: "blob" }
              );
              const url = URL.createObjectURL(res.data);
              setImages((prev) => ({ ...prev, [id]: url }));
            } catch (err) {
              console.error("Failed to fetch image", err);
            }
          }
        })
      );
    };
    fetchImages(naCells);
  }, [naCells, dbName, images]);

  useEffect(() => {
    const fetchImages = async (cellIds: string[]) => {
      await Promise.all(
        cellIds.map(async (id) => {
          if (!images[id]) {
            try {
              const res = await axios.get(
                `${url_prefix}/cells/${id}/${dbName}/true/false/ph_image`,
                { responseType: "blob" }
              );
              const url = URL.createObjectURL(res.data);
              setImages((prev) => ({ ...prev, [id]: url }));
            } catch (err) {
              console.error("Failed to fetch image", err);
            }
          }
        })
      );
    };
    fetchImages(label1Cells);
  }, [label1Cells, dbName, images]);

  const handleClick = async (cellId: string, fromLabel: "N/A" | "1") => {
    const newLabel = fromLabel === "N/A" ? "1" : "1000";
    try {
      await axios.patch(`${url_prefix}/cells/${dbName}/${cellId}/${newLabel}`);
      if (fromLabel === "N/A") {
        setNaCells((prev) => prev.filter((id) => id !== cellId));
        setLabel1Cells((prev) => [...prev, cellId]);
      } else {
        setLabel1Cells((prev) => prev.filter((id) => id !== cellId));
        setNaCells((prev) => [...prev, cellId]);
      }
    } catch (err) {
      console.error("Failed to update label", err);
    }
  };

  const renderCells = (cellIds: string[], label: "N/A" | "1") => (
    <Grid container spacing={1}>
      {cellIds.map((id) => (
        <Grid item xs={4} sm={3} md={2} key={id}>
          <Box
            component="img"
            src={images[id]}
            alt={id}
            sx={{ width: "100%", cursor: "pointer" }}
            onClick={() => handleClick(id, label)}
          />
          <Typography variant="caption" display="block" align="center">
            {id}
          </Typography>
        </Grid>
      ))}
    </Grid>
  );

  return (
    <Container>
      <Box mb={2}>
        <Typography variant="h6">Database: {dbName}</Typography>
      </Box>
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>
            N/A
          </Typography>
          {renderCells(naCells, "N/A")}
        </Grid>
        <Grid item xs={12} md={6}>
          <Typography variant="h6" gutterBottom>
            Label 1
          </Typography>
          {renderCells(label1Cells, "1")}
        </Grid>
      </Grid>
    </Container>
  );
};

export default LabelSorter;
