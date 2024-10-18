import React, { useRef, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { Points, PointMaterial } from '@react-three/drei';
import * as THREE from 'three';

type PointCloudProps = {
    jsonUrl: string;
};

const PointCloud: React.FC<PointCloudProps> = ({ jsonUrl }) => {
    const pointsRef = useRef<THREE.Points>(null);
    const [positions, setPositions] = useState<Float32Array | null>(null);

    useEffect(() => {
        fetch(jsonUrl)
            .then(response => response.json())
            .then(data => {
                const tempPositions: number[] = [];

                data.forEach((point: { x: number; y: number; z: number }) => {
                    tempPositions.push(point.x, point.y, point.z);
                });

                setPositions(new Float32Array(tempPositions));
            });
    }, [jsonUrl]);

    return (
        <Canvas camera={{ position: [0, 0, 500], fov: 75 }}>
            <ambientLight />
            <pointLight position={[10, 10, 10]} />

            {positions && (
                <Points ref={pointsRef}>
                    <bufferGeometry>
                        <bufferAttribute
                            attachObject={['attributes', 'position']}
                            array={positions}
                            itemSize={3}
                            count={positions.length / 3}
                        />
                    </bufferGeometry>
                    <PointMaterial size={1} color="white" />
                </Points>
            )}
        </Canvas>
    );
};

export default PointCloud;
