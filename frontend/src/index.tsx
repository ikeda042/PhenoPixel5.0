import React from 'react';
import ReactDOM from 'react-dom';
import DrawerAppBar from './Component1';

const App: React.FC = () => {
    return (
        <div>
            <h1>Hello, Electron with TypeScript and React!</h1>
            {/* ここに main.ts の内容を移行します */}
            <>
                <DrawerAppBar />
            </>
        </div>
    );
};

ReactDOM.render(<App />, document.getElementById('root'));
