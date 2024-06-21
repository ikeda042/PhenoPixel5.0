const electron = require("electron");   
//electron読み込んでapp,BrowserWindowを取り出す
const { app, BrowserWindow } = electron;
const path = require("path");           //pathはjoin用
let mainWindow;

//electron が準備終わったとき
app.on("ready", function () {

    //新しいウインドウを開く
    mainWindow = new BrowserWindow();

    //mainWindowでhtmlファイルを開く
    //"file://" + path.join(__dirname, 'index.html'); => file://作業ディレクトリ/index.html
    mainWindow.loadURL("file://" + path.join(__dirname, 'index.html'));

    //メインウインドウが閉じたらアプリが終了する
    mainWindow.on("closed", function () {
        app.quit();
    });
});