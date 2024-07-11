import React from "react";
import { Box, Button, Container } from "@mui/material";
import DatabaseIcon from '@mui/icons-material/Storage';
import ScienceIcon from '@mui/icons-material/Science';
import { useNavigate } from "react-router-dom";

const TopPage: React.FC = () => {
    const navigate = useNavigate();

    const handleNavigate = (path: string) => {
        navigate(path);
    };

    return (
        <Container>
            <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" gap={2} height="100vh">
                <Button
                    variant="contained"
                    component="span"
                    startIcon={<DatabaseIcon />}
                    onClick={() => handleNavigate('/dbconsole')}
                    sx={{
                        backgroundColor: 'white',
                        color: 'black',
                        width: '100%',
                        height: '56px',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    Data analyses
                </Button>
                <Button
                    variant="contained"
                    component="span"
                    startIcon={<ScienceIcon />}
                    onClick={() => handleNavigate('/phenopixel')}
                    sx={{
                        backgroundColor: 'white',
                        color: 'black',
                        width: '100%',
                        height: '56px',
                        '&:hover': {
                            backgroundColor: 'lightgrey'
                        }
                    }}
                >
                    Cell extraction
                </Button>
            </Box>
        </Container>
    );
};

export default TopPage;