"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.default = DrawerAppBar;
const React = __importStar(require("react"));
const AppBar_1 = __importDefault(require("@mui/material/AppBar"));
const Box_1 = __importDefault(require("@mui/material/Box"));
const CssBaseline_1 = __importDefault(require("@mui/material/CssBaseline"));
const Divider_1 = __importDefault(require("@mui/material/Divider"));
const Drawer_1 = __importDefault(require("@mui/material/Drawer"));
const IconButton_1 = __importDefault(require("@mui/material/IconButton"));
const List_1 = __importDefault(require("@mui/material/List"));
const ListItem_1 = __importDefault(require("@mui/material/ListItem"));
const ListItemButton_1 = __importDefault(require("@mui/material/ListItemButton"));
const ListItemText_1 = __importDefault(require("@mui/material/ListItemText"));
const Menu_1 = __importDefault(require("@mui/icons-material/Menu"));
const Toolbar_1 = __importDefault(require("@mui/material/Toolbar"));
const Typography_1 = __importDefault(require("@mui/material/Typography"));
const Button_1 = __importDefault(require("@mui/material/Button"));
const drawerWidth = 240;
const navItems = ['Home', 'About', 'Contact'];
function DrawerAppBar(props) {
    const { window } = props;
    const [mobileOpen, setMobileOpen] = React.useState(false);
    const handleDrawerToggle = () => {
        setMobileOpen((prevState) => !prevState);
    };
    const drawer = (React.createElement(Box_1.default, { onClick: handleDrawerToggle, sx: { textAlign: 'center' } },
        React.createElement(Typography_1.default, { variant: "h6", sx: { my: 2 } }, "MUI"),
        React.createElement(Divider_1.default, null),
        React.createElement(List_1.default, null, navItems.map((item) => (React.createElement(ListItem_1.default, { key: item, disablePadding: true },
            React.createElement(ListItemButton_1.default, { sx: { textAlign: 'center' } },
                React.createElement(ListItemText_1.default, { primary: item }))))))));
    const container = window !== undefined ? () => window().document.body : undefined;
    return (React.createElement(Box_1.default, { sx: { display: 'flex' } },
        React.createElement(CssBaseline_1.default, null),
        React.createElement(AppBar_1.default, { component: "nav" },
            React.createElement(Toolbar_1.default, null,
                React.createElement(IconButton_1.default, { color: "inherit", "aria-label": "open drawer", edge: "start", onClick: handleDrawerToggle, sx: { mr: 2, display: { sm: 'none' } } },
                    React.createElement(Menu_1.default, null)),
                React.createElement(Typography_1.default, { variant: "h6", component: "div", sx: { flexGrow: 1, display: { xs: 'none', sm: 'block' } } }, "MUI"),
                React.createElement(Box_1.default, { sx: { display: { xs: 'none', sm: 'block' } } }, navItems.map((item) => (React.createElement(Button_1.default, { key: item, sx: { color: '#fff' } }, item)))))),
        React.createElement("nav", null,
            React.createElement(Drawer_1.default, { container: container, variant: "temporary", open: mobileOpen, onClose: handleDrawerToggle, ModalProps: {
                    keepMounted: true, // Better open performance on mobile.
                }, sx: {
                    display: { xs: 'block', sm: 'none' },
                    '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
                } }, drawer)),
        React.createElement(Box_1.default, { component: "main", sx: { p: 3 } },
            React.createElement(Toolbar_1.default, null),
            React.createElement(Typography_1.default, null, "Lorem ipsum dolor sit amet consectetur adipisicing elit. Similique unde fugit veniam eius, perspiciatis sunt? Corporis qui ducimus quibusdam, aliquam dolore excepturi quae. Distinctio enim at eligendi perferendis in cum quibusdam sed quae, accusantium et aperiam? Quod itaque exercitationem, at ab sequi qui modi delectus quia corrupti alias distinctio nostrum. Minima ex dolor modi inventore sapiente necessitatibus aliquam fuga et. Sed numquam quibusdam at officia sapiente porro maxime corrupti perspiciatis asperiores, exercitationem eius nostrum consequuntur iure aliquam itaque, assumenda et! Quibusdam temporibus beatae doloremque voluptatum doloribus soluta accusamus porro reprehenderit eos inventore facere, fugit, molestiae ab officiis illo voluptates recusandae. Vel dolor nobis eius, ratione atque soluta, aliquam fugit qui iste architecto perspiciatis. Nobis, voluptatem! Cumque, eligendi unde aliquid minus quis sit debitis obcaecati error, delectus quo eius exercitationem tempore. Delectus sapiente, provident corporis dolorum quibusdam aut beatae repellendus est labore quisquam praesentium repudiandae non vel laboriosam quo ab perferendis velit ipsa deleniti modi! Ipsam, illo quod. Nesciunt commodi nihil corrupti cum non fugiat praesentium doloremque architecto laborum aliquid. Quae, maxime recusandae? Eveniet dolore molestiae dicta blanditiis est expedita eius debitis cupiditate porro sed aspernatur quidem, repellat nihil quasi praesentium quia eos, quibusdam provident. Incidunt tempore vel placeat voluptate iure labore, repellendus beatae quia unde est aliquid dolor molestias libero. Reiciendis similique exercitationem consequatur, nobis placeat illo laudantium! Enim perferendis nulla soluta magni error, provident repellat similique cupiditate ipsam, et tempore cumque quod! Qui, iure suscipit tempora unde rerum autem saepe nisi vel cupiditate iusto. Illum, corrupti? Fugiat quidem accusantium nulla. Aliquid inventore commodi reprehenderit rerum reiciendis! Quidem alias repudiandae eaque eveniet cumque nihil aliquam in expedita, impedit quas ipsum nesciunt ipsa ullam consequuntur dignissimos numquam at nisi porro a, quaerat rem repellendus. Voluptates perspiciatis, in pariatur impedit, nam facilis libero dolorem dolores sunt inventore perferendis, aut sapiente modi nesciunt."))));
}
