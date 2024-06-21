import React from 'react';
import ReactDOM from 'react-dom';
import Nav from './components/Component1';

const App: React.FC = () => {
    return (
        <div>
            <>
                <Nav />
            </>
        </div>
    );
};

ReactDOM.render(<App />, document.getElementById('root'));
