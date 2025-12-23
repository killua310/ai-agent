import dynamic from 'next/dynamic';
import { useRef, useState, useEffect } from 'react';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
    ssr: false,
    loading: () => <p>Loading Graph...</p>
});

const GraphView = ({ data, onNodeClick, width: propsWidth, height: propsHeight, ...props }) => {
    const containerRef = useRef(null);
    const [dimensions, setDimensions] = useState({ width: propsWidth || 1, height: propsHeight || 1 });

    useEffect(() => {
        // If explicit props are provided, use them and skip observation
        if (propsWidth && propsHeight) {
            setDimensions({ width: propsWidth, height: propsHeight });
            return;
        }

        if (!containerRef.current) return;

        const resizeObserver = new ResizeObserver(entries => {
            for (let entry of entries) {
                const { width, height } = entry.contentRect;
                if (width > 0 && height > 0) {
                    setDimensions({
                        width: propsWidth || width,
                        height: propsHeight || height
                    });
                }
            }
        });

        resizeObserver.observe(containerRef.current);
        return () => resizeObserver.disconnect();
    }, [propsWidth, propsHeight]);

    return (
        <div ref={containerRef} style={{ width: '100%', height: '100%', minHeight: '100px' }}>
            {dimensions.width > 1 && (
                <ForceGraph2D
                    graphData={data}
                    width={dimensions.width}
                    height={dimensions.height}
                    nodeLabel="id"
                    backgroundColor="#000"
                    linkColor={() => '#444'}
                    linkDirectionalArrowLength={3.5}
                    linkDirectionalArrowRelPos={1}
                    linkCurvature={(link) => {
                        if (link.key && link.key > 0) return 0.2 * link.key;
                        return 0;
                    }}
                    onNodeClick={onNodeClick}
                    {...props}
                />
            )}
        </div>
    );
};

export default GraphView;
