import React from 'react';
import ReactDOM from 'react-dom';
import Nav from './components/NavigationBar';

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
