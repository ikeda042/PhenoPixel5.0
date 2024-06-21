import React from 'react';
import ReactDOM from 'react-dom';
import DrawerAppBar from './components/Component1';

const App: React.FC = () => {
    return (
        <div>
            <>
                <DrawerAppBar />
            </>
        </div>
    );
};

ReactDOM.render(<App />, document.getElementById('root'));
