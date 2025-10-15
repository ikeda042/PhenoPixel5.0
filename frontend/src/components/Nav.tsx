import * as React from 'react';
import { Link, useNavigate } from 'react-router-dom';
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
import RefreshIcon from '@mui/icons-material/Refresh';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import GitHubIcon from '@mui/icons-material/GitHub';
import AccountCircle from '@mui/icons-material/AccountCircle';
import ScienceIcon from '@mui/icons-material/Science';
import DatabaseIcon from '@mui/icons-material/Storage';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import Snackbar from '@mui/material/Snackbar';
import Alert from '@mui/material/Alert';
import { settings } from '../settings';

interface Props {
    window?: () => Window;
    title: string;
    mode: 'light' | 'dark';
    toggleMode: () => void;
}

const drawerWidth = 240;
const navItems = [""];

export default function Nav(props: Props) {
    const { window, mode, toggleMode } = props;
    const [mobileOpen, setMobileOpen] = React.useState(false);
    const [refreshing, setRefreshing] = React.useState(false);
    const [refreshStatus, setRefreshStatus] = React.useState<{ message: string; severity: 'success' | 'error' } | null>(null);
    const navigate = useNavigate();

    const handleDrawerToggle = () => {
        setMobileOpen(!mobileOpen);
    };

    // ローカルストレージにaccess_tokenが存在すればログイン状態とする
    const isLoggedIn = Boolean(localStorage.getItem("access_token"));

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

    const handleRefresh = async () => {
        if (refreshing) {
            return;
        }
        setRefreshing(true);
        try {
            const response = await fetch(`${settings.url_prefix}/api/dev/git-pull`);
            if (!response.ok) {
                throw new Error(`Request failed with status ${response.status}`);
            }
            setRefreshStatus({ message: 'Git pull successful.', severity: 'success' });
        } catch (error) {
            console.error('Failed to run git pull:', error);
            setRefreshStatus({ message: 'Failed to run git pull.', severity: 'error' });
        } finally {
            setRefreshing(false);
        }
    };

    const handleRefreshStatusClose = (_event?: React.SyntheticEvent | Event, reason?: string) => {
        if (reason === 'clickaway') {
            return;
        }
        setRefreshStatus(null);
    };

    return (
        <Box sx={{ display: 'flex' }}>
            <CssBaseline />
            <AppBar
                component="nav"
                color="default"
                enableColorOnDark
                sx={{
                    backgroundColor: mode === 'light' ? '#fff' : 'background.paper',
                    color: mode === 'light' ? '#000' : 'text.primary',
                }}
            >
                <Toolbar>
                    <IconButton
                        color="inherit"
                        aria-label="open drawer"
                        edge="start"
                        onClick={handleDrawerToggle}
                        sx={{ mr: 2, display: { sm: 'none' } }}
                    >
                        <MenuIcon />
                    </IconButton>
                    <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
                        <Typography
                            variant="h5"
                            component="div"
                            sx={{ display: 'flex', alignItems: 'center' }}
                        >
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
                    {/* ロゴと右側アイコン群の間にスペーサーを配置 */}
                    <Box sx={{ flexGrow: 1 }} />
                    <IconButton
                        color="inherit"
                        onClick={handleRefresh}
                        disabled={refreshing}
                        aria-label="Refresh repository"
                    >
                        <RefreshIcon />
                    </IconButton>
                    <IconButton
                        color="inherit"
                        onClick={() => navigate('/nd2files')}
                    >
                        <ScienceIcon />
                    </IconButton>
                    <IconButton
                        color="inherit"
                        onClick={() => navigate('/dbconsole')}
                    >
                        <DatabaseIcon />
                    </IconButton>
                    <IconButton
                        color="inherit"
                        component="a"
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
                    >
                        <GitHubIcon />
                    </IconButton>
                    <IconButton color="inherit" onClick={toggleMode}>
                        {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
                    </IconButton>
                    {isLoggedIn ? (
                        <IconButton
                            color="inherit"
                            onClick={() => navigate("/user_info")}
                        >
                            <AccountCircle fontSize="large" />
                        </IconButton>
                    ) : (
                        <Link to="/login" style={{ textDecoration: 'none', color: 'inherit' }}>
                            <Typography variant="body1" sx={{ mr: 2 }}>Login</Typography>
                        </Link>
                    )}
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

            {/* NavBarの下にコンテンツを表示するために Toolbar を配置 */}
            <Box component="main" sx={{ p: 1 }}>
                <Toolbar />
            </Box>
            <Snackbar
                open={Boolean(refreshStatus)}
                autoHideDuration={4000}
                onClose={handleRefreshStatusClose}
                anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            >
                {refreshStatus ? (
                    <Alert onClose={handleRefreshStatusClose} severity={refreshStatus.severity} sx={{ width: '100%' }}>
                        {refreshStatus.message}
                    </Alert>
                ) : null}
            </Snackbar>
        </Box>
    );
}
