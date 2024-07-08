//配列
// {
//     "contour": [
//         [
//             -53.968862045043096,
//             -98.49041541979149
//         ],
//         [
//             -54.89701430739714,
//             -97.42339462435821
//         ],
//         [
//             -54.82758004085752,
//             -96.42580809546455
//         ],
//         [
//             -55.755732303211566,
//             -95.35878730003127
//         ],,,,,,


// PlotComponent.tsx
import React, { useEffect, useState } from 'react';
import Plot from 'react-plotly.js';

interface DataPoint {
    u1: number;
    u2: number;
    points_inside_cell_1: number[];
    min_u1: number;
    max_u1: number;
    u1_c: number;
    u2_c: number;
}

const PlotComponent: React.FC = () => {
    const [data, setData] = useState<DataPoint | null>(null);

    useEffect(() => {
        fetch('/replot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                gray: /* grayデータ */,
                image_fluo: /* image_fluoデータ */,
                contour_raw: /* contour_rawデータ */,
            }),
        })
            .then((response) => response.json())
            .then((data) => setData(data));
    }, []);

    if (!data) {
        return <p>Loading...</p>;
    }

    return (
        <Plot
            data={[
                {
                    x: data.u1,
                    y: data.u2,
                    mode: 'markers',
                    marker: {
                        color: data.points_inside_cell_1,
                        colorscale: 'Inferno',
                        size: 10,
                    },
                },
            ]}
            layout={{ width: 800, height: 600, title: 'Plot' }}
        />
    );
};

export default PlotComponent;
