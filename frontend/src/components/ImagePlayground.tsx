import React, { useState } from "react";
import {
  Container,
  TextField,
  Button,
  Stack,
  Box,
  Breadcrumbs,
  Link,
  Typography,
} from "@mui/material";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

const ImagePlayground: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [threshold1, setThreshold1] = useState(100);
  const [threshold2, setThreshold2] = useState(200);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setFile(e.target.files[0]);
    }
  };

  const handleProcess = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    setIsLoading(true);
    try {
      const res = await fetch(
        `${url_prefix}/image_playground/canny?threshold1=${threshold1}&threshold2=${threshold2}`,
        {
          method: "POST",
          body: formData,
        }
      );
      if (!res.ok) throw new Error("Request failed");
      const blob = await res.blob();
      setImageSrc(URL.createObjectURL(blob));
    } catch (err) {
      console.error(err);
      alert("Failed to process image");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="sm" sx={{ py: 4 }}>
      <Breadcrumbs sx={{ mb: 3 }}>
        <Link underline="hover" color="inherit" href="/">
          Top
        </Link>
        <Typography color="text.primary">Image Playground</Typography>
      </Breadcrumbs>
      <Stack spacing={2}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <TextField
          label="Threshold1"
          type="number"
          value={threshold1}
          onChange={(e) => setThreshold1(parseInt(e.target.value, 10))}
          size="small"
        />
        <TextField
          label="Threshold2"
          type="number"
          value={threshold2}
          onChange={(e) => setThreshold2(parseInt(e.target.value, 10))}
          size="small"
        />
        <Button variant="contained" onClick={handleProcess} disabled={isLoading}>
          Apply Canny
        </Button>
        {imageSrc && (
          <Box component="img" src={imageSrc} sx={{ width: "100%" }} />
        )}
      </Stack>
    </Container>
  );
};

export default ImagePlayground;
