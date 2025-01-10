import * as React from 'react';
import { Link } from 'react-router-dom';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import CssBaseline from '@mui/material/CssBaseline';
import Divider from '@mui/material/Divider';
import Drawer from '@mui/material/Drawer';
import IconButton from '@mui/material/IconButton';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import MenuIcon from '@mui/icons-material/Menu';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import GitHubIcon from '@mui/icons-material/GitHub';
import { settings } from '../settings';

interface Props {
    window?: () => Window;
    title: string;
}

const drawerWidth = 240;
const navItems = [""];

export default function Nav(props: Props) {
    const { window } = props;
    const [mobileOpen, setMobileOpen] = React.useState(false);

    const handleDrawerToggle = () => {
        setMobileOpen(!mobileOpen);
    };

    const drawer = (
        <Box onClick={handleDrawerToggle} sx={{ textAlign: 'center' }}>
            <Typography variant="h6" sx={{ my: 2 }}>
                Cell APIs
            </Typography>
            <Divider />
            <List>
                {navItems.map((item) => (
                    <ListItem key={item} disablePadding>
                        <ListItemButton sx={{ textAlign: 'center' }}>
                            <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
                                <ListItemText primary={item} />
                            </Link>
                        </ListItemButton>
                    </ListItem>
                ))}
            </List>
        </Box>
    );

    const container = window !== undefined ? () => window().document.body : undefined;

    return (
        <Box sx={{ display: 'flex' }}>
            <CssBaseline />
            <AppBar component="nav" sx={{ backgroundColor: '#fff' }}>
                <Toolbar>
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        edge="start"
                        onClick={handleDrawerToggle}
                        sx={{ mr: 2, display: { sm: 'none' }, color: '#000' }}
                    >
                        <MenuIcon />
                    </IconButton>
                    <Link to="/" style={{ textDecoration: 'none', color: '#000' }}>
                        <Typography variant="h5" component="div" sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', color: '#000' }}>
                            <Box
                                component="img"
                                sx={{
                                    height: 30,
                                    width: 30,
                                    display: 'block',
                                    marginRight: '10px'
                                }}
                                src={"/logo192.png"}
                            />
                            {props.title}
                        </Typography>
                    </Link>
                    <Box sx={{ flexGrow: 1 }} />
                    <IconButton
                        color="inherit"
                        component="a"
                        sx={{ color: '#000' }}
                        href={`${settings.url_prefix}/docs`}
                        target="_blank"
                        rel="noopener noreferrer"
                    >
                        <Box
                            component="img"
                            src="logo-fastapi.png"
                            alt="FastAPI Logo"
                            sx={{ height: 40 }}
                        />
                    </IconButton>
                    <IconButton
                        color="inherit"
                        component="a"
                        href="https://github.com/ikeda042/PhenoPixel5.0"
                        target="_blank"
                        rel="noopener noreferrer"
                        sx={{ color: '#000' }}
                    >
                        <GitHubIcon />
                    </IconButton>
                </Toolbar>
            </AppBar>
            <Drawer
                container={container}
                variant="temporary"
                open={mobileOpen}
                onClose={handleDrawerToggle}
                ModalProps={{ keepMounted: true }}
                sx={{
                    display: { xs: 'block', sm: 'none' },
                    '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                }}
            >
                {drawer}
            </Drawer>

            {/* NavBarの下にコンテンツを表示するために Toolbar() を配置 */}
            <Box component="main" sx={{ p: 1 }}>
                <Toolbar />
            </Box>
        </Box>
    );
}
