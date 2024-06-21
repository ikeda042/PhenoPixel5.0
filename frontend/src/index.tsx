import React from 'react';
import ReactDOM from 'react-dom';

const App: React.FC = () => {
    return <h1>Hello, Electron with TypeScript and React!</h1>;
};

ReactDOM.render(<App />, document.getElementById('root'));