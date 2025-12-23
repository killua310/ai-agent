import dynamic from 'next/dynamic';

const ForceGraph2D = dynamic(() => import('react-force-graph-2d'), {
    ssr: false,
    loading: () => <p>Loading Graph...</p>
});

const GraphView = ({ data, onNodeClick }) => {
    return (
        <div style={{ border: '1px solid #ccc', borderRadius: '8px', overflow: 'hidden' }}>
            <ForceGraph2D
                graphData={data}
                width={600}
                height={400}
                nodeLabel="id"
                linkLabel="relation"
                nodeAutoColorBy="id"
                linkDirectionalArrowLength={6}
                linkDirectionalArrowRelPos={1}
                linkCurvature={(link) => {
                    // Logic to curve identical links
                    // We need to check if there are multiple links between source and target
                    // For simplicity in this specialized view:
                    // If the link has a 'key' > 0, curve it.
                    if (link.key && link.key > 0) return 0.2 * link.key;
                    return 0;
                }}
                onNodeClick={onNodeClick}
            />
        </div>
    );
};

export default GraphView;
