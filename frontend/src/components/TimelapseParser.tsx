import React, { useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
    Box,
    Grid,
    Typography,
    Button,
    Backdrop,
    CircularProgress,
    Breadcrumbs,
    Link,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    SelectChangeEvent
} from "@mui/material";
import axios from "axios";
import { settings } from "../settings";

const url_prefix = settings.url_prefix;

// 不要なカスタム TextField やスタイルは削除（または必要に応じて残す）
const TimelapseParser: React.FC = () => {
    const [searchParams] = useSearchParams();
    const fileName = searchParams.get("file_name") || "";
    const [isLoading, setIsLoading] = useState(false);

    // Field 関連
    const [fields, setFields] = useState<string[]>([]);
    const [selectedField, setSelectedField] = useState<string>("");
    const [gifUrl, setGifUrl] = useState<string>("");

    /**
     * ND2 ファイルをパースする
     * 1) GET /tlengine/nd2_files/{file_name}
     * 2) パース完了後に、GET /tlengine/nd2_files/{file_name}/fields を呼び出す
     */
    const handleParseND2 = async () => {
        if (!fileName) return;
        setIsLoading(true);
        try {
            // 1) ND2 ファイルの解析
            await axios.get(`${url_prefix}/tlengine/nd2_files/${fileName}`);
            // 2) Field 一覧を取得
            const fieldsResponse = await axios.get(
                `${url_prefix}/tlengine/nd2_files/${fileName}/fields`
            );
            if (fieldsResponse.data.fields) {
                setFields(fieldsResponse.data.fields);
            } else {
                setFields([]);
            }
        } catch (error) {
            console.error("Failed to parse ND2 file or get fields", error);
        } finally {
            setIsLoading(false);
        }
    };

    /**
     * Field 選択時に呼び出す
     * GET /tlengine/nd2_files/{file_name}/gif/{Field} で GIF を取得
     */
    const fetchGif = async (fieldValue: string) => {
        if (!fileName || !fieldValue) return;
        setIsLoading(true);
        try {
            const response = await axios.get(
                `${url_prefix}/tlengine/nd2_files/${fileName}/gif/${fieldValue}`,
                { responseType: "blob" }
            );
            const blobUrl = URL.createObjectURL(response.data);
            setGifUrl(blobUrl);
        } catch (error) {
            console.error("Failed to fetch GIF", error);
        } finally {
            setIsLoading(false);
        }
    };

    /**
     * Field セレクト変更時のハンドラ
     */
    const handleFieldChange = (event: SelectChangeEvent<string>) => {
        const newField = event.target.value as string;
        setSelectedField(newField);
        fetchGif(newField);
    };

    return (
        <Box>
            <Backdrop open={isLoading} style={{ zIndex: 1201 }}>
                <CircularProgress color="inherit" />
            </Backdrop>

            {/* パンくずリストは必要に応じて修正 */}
            <Box mb={2}>
                <Breadcrumbs aria-label="breadcrumb">
                    <Link underline="hover" color="inherit" href="/">
                        Top
                    </Link>
                    <Link underline="hover" color="inherit" href="/tlengine">
                        Timelapse ND2 files
                    </Link>
                    <Typography color="text.primary">ND2 parse</Typography>
                </Breadcrumbs>
            </Box>

            <Grid container spacing={2} >
                <Grid item xs={12} md={4}>
                    <Typography variant="body1" mb={2}>
                        ND2 filename: {fileName}
                    </Typography>
                    
                    {/* ND2 パースボタン */}
                    <Button
                        variant="contained"
                        color="primary"
                        fullWidth
                        onClick={handleParseND2}
                        disabled={isLoading || !fileName}
                        sx={{
                            backgroundColor: 'black',
                            color: 'white',
                            height: '56px',
                            textTransform: 'none',
                            '&:hover': {
                                backgroundColor: 'grey'
                            }
                        }}
                    >
                        Parse ND2 File
                    </Button>

                    {/* Field ドロップダウン */}
                    {fields.length > 0 && (
                        <FormControl fullWidth margin="normal">
                            <InputLabel>Field</InputLabel>
                            <Select
                                value={selectedField}
                                onChange={handleFieldChange}
                            >
                                {fields.map((field) => (
                                    <MenuItem key={field} value={field}>
                                        {field}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>
                    )}
                </Grid>

                {/* GIF 表示エリア */}
                {gifUrl && (
                    <Grid item xs={12} md={8}>
                        <Box display="flex" flexDirection="column" alignItems="center" mt={5}>
                            <Typography variant="body1" mb={2}>
                                Selected Field: {selectedField}
                            </Typography>
                            <Box
                                component="img"
                                src={gifUrl}
                                alt={`GIF for Field: ${selectedField}`}
                                sx={{ maxWidth: '400px', height: 'auto', objectFit: 'contain' }}
                            />
                        </Box>
                    </Grid>
                )}
            </Grid>
        </Box>
    );
};

export default TimelapseParser;
