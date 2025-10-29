import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Breadcrumbs,
  CircularProgress,
  Container,
  Grid,
  Link,
  Paper,
  Skeleton,
  Typography,
} from "@mui/material";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import axios from "axios";
import { settings } from "../settings";

const urlPrefix = settings.url_prefix;
const resizeFactor = 0.5;
const cycleIntervalMs = 500;
const resetDelayMs = 1000;

interface CellResponse {
  cell_id: string;
}

const LabelSorterDemo: React.FC = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name") ?? "";

  const [naCells, setNaCells] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [images, setImages] = useState<Record<string, string>>({});
  const [index, setIndex] = useState(0);

  const objectUrlsRef = useRef<string[]>([]);

  useEffect(() => {
    return () => {
      objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  useEffect(() => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current = [];
    setImages({});
    setIndex(0);
  }, [dbName]);

  useEffect(() => {
    let cancelled = false;

    if (!dbName) {
      setNaCells([]);
      setError(null);
      return;
    }

    const fetchCells = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await axios.get<CellResponse[]>(
          `${urlPrefix}/cells/${encodeURIComponent(dbName)}/1000`
        );
        if (!cancelled) {
          setNaCells(response.data.map((cell) => cell.cell_id));
        }
      } catch (err) {
        console.error("Failed to fetch N/A cells", err);
        if (!cancelled) {
          setError("細胞の取得に失敗しました。");
          setNaCells([]);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };

    fetchCells();

    return () => {
      cancelled = true;
    };
  }, [dbName]);

  const cycleCells = useMemo(() => naCells, [naCells]);

  useEffect(() => {
    setIndex(0);
  }, [cycleCells.length]);

  useEffect(() => {
    const length = cycleCells.length;
    if (!length) {
      return;
    }

    const delay = index < length ? cycleIntervalMs : resetDelayMs;

    const timer = setTimeout(() => {
      setIndex((prev) => (prev < length ? prev + 1 : 0));
    }, delay);

    return () => clearTimeout(timer);
  }, [index, cycleCells.length]);

  const leftCells = useMemo(() => {
    if (!cycleCells.length) {
      return [];
    }
    const cut = Math.min(index, cycleCells.length);
    return cycleCells.slice(cut);
  }, [cycleCells, index]);

  const rightCells = useMemo(() => {
    if (!cycleCells.length) {
      return [];
    }
    const cut = Math.min(index, cycleCells.length);
    return cycleCells.slice(0, cut);
  }, [cycleCells, index]);

  useEffect(() => {
    if (!dbName) {
      return;
    }
    const requiredIds = new Set<string>([...leftCells, ...rightCells]);
    const missingIds = Array.from(requiredIds).filter(
      (cellId) => !images[`${cellId}_ph`]
    );
    if (!missingIds.length) {
      return;
    }

    let cancelled = false;

    const fetchImages = async () => {
      const newEntries: Record<string, string> = {};
      await Promise.all(
        missingIds.map(async (cellId) => {
          try {
            const res = await axios.get(
              `${urlPrefix}/cells/${encodeURIComponent(cellId)}/${encodeURIComponent(
                dbName
              )}/true/false/ph_image?resize_factor=${resizeFactor}&contour_thickness=3`,
              { responseType: "blob" }
            );
            if (cancelled) {
              return;
            }
            const url = URL.createObjectURL(res.data);
            objectUrlsRef.current.push(url);
            newEntries[`${cellId}_ph`] = url;
          } catch (err) {
            console.error(`Failed to fetch image for ${cellId}`, err);
          }
        })
      );
      if (!cancelled && Object.keys(newEntries).length) {
        setImages((prev) => ({ ...prev, ...newEntries }));
      }
    };

    fetchImages();

    return () => {
      cancelled = true;
    };
  }, [dbName, leftCells, rightCells, images]);

  const renderCells = (cellIds: string[]) => (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
        gap: 2,
        maxHeight: "70vh",
        overflowY: "auto",
        p: 1,
      }}
    >
      {cellIds.map((cellId) => {
        const key = `${cellId}_ph`;
        const imageSrc = images[key];
        return (
          <Box
            key={cellId}
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              border: "1px solid",
              borderColor: "divider",
              borderRadius: 1,
              p: 1,
              backgroundColor: "background.paper",
            }}
          >
            {imageSrc ? (
              <Box
                component="img"
                src={imageSrc}
                alt={cellId}
                sx={{ width: "100%", borderRadius: 1 }}
              />
            ) : (
              <Skeleton
                variant="rectangular"
                width="100%"
                height={120}
                sx={{ borderRadius: 1 }}
              />
            )}
            <Typography variant="caption" sx={{ mt: 1 }}>
              {cellId}
            </Typography>
          </Box>
        );
      })}
    </Box>
  );

  return (
    <Container maxWidth="xl">
      <Box sx={{ my: 2 }}>
        <Breadcrumbs aria-label="breadcrumb">
          <Link component={RouterLink} underline="hover" color="inherit" to="/">
            Home
          </Link>
          <Link
            component={RouterLink}
            underline="hover"
            color="inherit"
            to="/labelsorter"
          >
            Label Sorter
          </Link>
          <Typography color="text.primary">Demo</Typography>
        </Breadcrumbs>
      </Box>

      <Typography variant="h4" gutterBottom>
        Label Sorter Demo
      </Typography>
      <Typography variant="body1" sx={{ mb: 3 }}>
        N/A ラベルの細胞を左パネルに表示し、0.5 秒ごとに自動でラベル "1"
        に移動して右パネルへ写します。すべて移動し終わると自動で元の状態に戻り、デモを繰り返します。
      </Typography>

      {!dbName ? (
        <Paper sx={{ p: 3 }}>
          <Typography variant="body1">
            クエリパラメータ
            <code>?db_name=...</code>
            を指定してデモ対象のデータベースを選択してください。
          </Typography>
        </Paper>
      ) : (
        <>
          <Typography variant="subtitle1" sx={{ mb: 2 }}>
            データベース: {dbName}
          </Typography>

          {isLoading && (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 1,
                mb: 2,
              }}
            >
              <CircularProgress size={20} />
              <Typography variant="body2">細胞データを読み込み中...</Typography>
            </Box>
          )}

          {error && (
            <Typography color="error" sx={{ mb: 2 }}>
              {error}
            </Typography>
          )}

          {!isLoading && !error && !naCells.length && (
            <Typography variant="body2">
              N/A ラベルの細胞が見つかりませんでした。
            </Typography>
          )}

          {!!naCells.length && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    N/A (Left)
                  </Typography>
                  {renderCells(leftCells)}
                </Paper>
              </Grid>
              <Grid item xs={12} md={6}>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Label "1" (Right)
                  </Typography>
                  {renderCells(rightCells)}
                </Paper>
              </Grid>
            </Grid>
          )}
        </>
      )}
    </Container>
  );
};

export default LabelSorterDemo;
