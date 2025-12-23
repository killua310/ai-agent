import dynamic from 'next/dynamic';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
    ssr: false,
    loading: () => <p>Loading Graph...</p>
});

const GraphView = ({ data, onNodeClick, ...props }) => {
    return (
        <ForceGraph2D
            graphData={data}
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
    );
};

export default GraphView;
