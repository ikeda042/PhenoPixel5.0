import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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

type CyclePhase = "idle" | "cycling" | "resetting";

interface CellResponse {
  cell_id: string;
}

const LabelSorterDemo: React.FC = () => {
  const [searchParams] = useSearchParams();
  const dbName = searchParams.get("db_name") ?? "";

  const [naCells, setNaCells] = useState<string[]>([]);
  const [confirmedCells, setConfirmedCells] = useState<string[]>([]);
  const [phase, setPhase] = useState<CyclePhase>("idle");
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [images, setImages] = useState<Record<string, string>>({});

  const objectUrlsRef = useRef<string[]>([]);
  const rightPanelRef = useRef<HTMLDivElement | null>(null);
  const canMutate = useMemo(
    () => dbName.includes("-uploaded"),
    [dbName]
  );

  useEffect(() => {
    return () => {
      objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      objectUrlsRef.current = [];
    };
  }, []);

  useEffect(() => {
    objectUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
    objectUrlsRef.current = [];
    setImages({});
    setNaCells([]);
    setConfirmedCells([]);
    setCurrentIndex(0);
    setPhase("idle");
  }, [dbName]);

  const fetchNaCells = useCallback(async () => {
    if (!dbName) {
      setNaCells([]);
      setConfirmedCells([]);
      setCurrentIndex(0);
      setPhase("idle");
      setError(null);
      return;
    }

    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get<CellResponse[]>(
        `${urlPrefix}/cells/${encodeURIComponent(dbName)}/1000`
      );
      const ids = response.data.map((cell) => cell.cell_id);
      setNaCells(ids);
      setConfirmedCells([]);
      setCurrentIndex(0);
      if (ids.length && canMutate) {
        setPhase("cycling");
      } else {
        setPhase("idle");
        if (ids.length && !canMutate) {
          setError(
            "デモの自動ラベル変更は '-uploaded.db' のデータベースでのみ利用できます。"
          );
        }
      }
    } catch (err) {
      console.error("Failed to fetch N/A cell ids", err);
      setError("細胞の取得に失敗しました。");
      setNaCells([]);
      setConfirmedCells([]);
      setCurrentIndex(0);
      setPhase("idle");
    } finally {
      setIsLoading(false);
    }
  }, [dbName, canMutate]);

  useEffect(() => {
    void fetchNaCells();
  }, [fetchNaCells]);

  const leftCells = useMemo(
    () => naCells.filter((id) => !confirmedCells.includes(id)),
    [naCells, confirmedCells]
  );
  const rightCells = confirmedCells;

  useEffect(() => {
    if (rightPanelRef.current) {
      rightPanelRef.current.scrollTop = rightPanelRef.current.scrollHeight;
    }
  }, [rightCells.length]);

  const fetchImagesForVisibleCells = useCallback(
    async (cellIds: string[]) => {
      if (!dbName) {
        return;
      }
      const newEntries: Record<string, string> = {};
      await Promise.all(
        cellIds.map(async (cellId) => {
          const key = `${cellId}_ph`;
          if (images[key]) {
            return;
          }
          try {
            const response = await axios.get(
              `${urlPrefix}/cells/${encodeURIComponent(cellId)}/${encodeURIComponent(
                dbName
              )}/true/false/ph_image?resize_factor=${resizeFactor}&contour_thickness=3`,
              { responseType: "blob" }
            );
            const url = URL.createObjectURL(response.data);
            objectUrlsRef.current.push(url);
            newEntries[key] = url;
          } catch (err) {
            console.error(`Failed to fetch image for ${cellId}`, err);
          }
        })
      );
      if (Object.keys(newEntries).length) {
        setImages((prev) => ({ ...prev, ...newEntries }));
      }
    },
    [dbName, images]
  );

  useEffect(() => {
    const visibleIds = Array.from(new Set([...leftCells, ...rightCells]));
    if (visibleIds.length) {
      void fetchImagesForVisibleCells(visibleIds);
    }
  }, [leftCells, rightCells, fetchImagesForVisibleCells]);

  const promoteCurrentCell = useCallback(async () => {
    if (!dbName || currentIndex >= naCells.length) {
      return;
    }
    if (!canMutate) {
      setError("デモでラベルを変更するには '-uploaded.db' で終わるデータベースを指定してください。");
      setPhase("idle");
      return;
    }
    const cellId = naCells[currentIndex];
    setIsProcessing(true);
    try {
      await axios.patch(
        `${urlPrefix}/cells/${encodeURIComponent(dbName)}/${encodeURIComponent(
          cellId
        )}/1`
      );
      const labelResponse = await axios.get(
        `${urlPrefix}/cells/${encodeURIComponent(dbName)}/${encodeURIComponent(
          cellId
        )}/label`
      );
      const normalizedLabel = String(labelResponse.data);
      if (normalizedLabel === "1") {
        setConfirmedCells((prev) =>
          prev.includes(cellId) ? prev : [...prev, cellId]
        );
      } else {
        console.warn(
          `Cell ${cellId} label is ${normalizedLabel}, skipping display.`
        );
      }
    } catch (err) {
      console.error(`Failed to update or verify label for ${cellId}`, err);
      setError("ラベルの更新または確認に失敗しました。");
    } finally {
      const nextIndex = currentIndex + 1;
      setCurrentIndex(nextIndex);
      setIsProcessing(false);
      if (nextIndex >= naCells.length) {
        setPhase("resetting");
      }
    }
  }, [dbName, currentIndex, naCells, canMutate]);

  const resetToNa = useCallback(async () => {
    if (!dbName) {
      setPhase("idle");
      return;
    }
    if (!canMutate) {
      setConfirmedCells([]);
      setCurrentIndex(0);
      setPhase("idle");
      return;
    }
    const cellsToReset = [...confirmedCells];
    if (!cellsToReset.length) {
      setCurrentIndex(0);
      setPhase(naCells.length ? "cycling" : "idle");
      return;
    }

    setIsProcessing(true);
    try {
      await Promise.all(
        cellsToReset.map((cellId) =>
          axios.patch(
            `${urlPrefix}/cells/${encodeURIComponent(dbName)}/${encodeURIComponent(
              cellId
            )}/1000`
          )
        )
      );
      setError(null);
    } catch (err) {
      console.error("Failed to reset labels to N/A", err);
      setError("ラベルのリセットに失敗しました。");
    } finally {
      setConfirmedCells([]);
      setCurrentIndex(0);
      setIsProcessing(false);
      setPhase(naCells.length ? "cycling" : "idle");
    }
  }, [dbName, confirmedCells, naCells.length, canMutate]);

  useEffect(() => {
    if (phase !== "cycling") {
      return;
    }
    if (!dbName || !naCells.length || !canMutate) {
      setPhase("idle");
      return;
    }
    if (isProcessing) {
      return;
    }
    if (currentIndex >= naCells.length) {
      setPhase("resetting");
      return;
    }

    const timer = setTimeout(() => {
      void promoteCurrentCell();
    }, cycleIntervalMs);
    return () => clearTimeout(timer);
  }, [
    phase,
    dbName,
    naCells.length,
    isProcessing,
    currentIndex,
    promoteCurrentCell,
    canMutate,
  ]);

  useEffect(() => {
    if (phase !== "resetting" || isProcessing) {
      return;
    }
    const timer = setTimeout(() => {
      void resetToNa();
    }, resetDelayMs);
    return () => clearTimeout(timer);
  }, [phase, isProcessing, resetToNa]);

  const renderCells = (
    cellIds: string[],
    panelRef?: React.RefObject<HTMLDivElement>
  ) => (
    <Box
      ref={panelRef}
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
                  {renderCells(rightCells, rightPanelRef)}
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
