import { useState, useEffect } from 'react';
import axios from 'axios';

import Head from 'next/head';
import GraphView from '../components/GraphView';

export default function Home() {
  const [messages, setMessages] = useState([]); // Start empty, populate on load
  const [input, setInput] = useState('');
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);

  // Profile State
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [filterMode, setFilterMode] = useState(null); // { type: 'multi', ids: [] }
  const [profileLoading, setProfileLoading] = useState(false);

  const fetchGraph = async () => {
    try {
      const res = await axios.get('http://localhost:8000/graph');
      console.log("Fetched Graph Data:", res.data);
      setGraphData(res.data);
      return res.data;
    } catch (error) {
      console.error("Error fetching graph:", error);
      return null;
    }
  };

  const handleNodeClick = async (node) => {
    if (!node) {
      setSelectedEntity(null);
      return;
    }
    setSelectedEntity({ id: node.id, summary: null });
    setProfileLoading(true);
    try {
      const res = await axios.get(`http://localhost:8000/summary/${node.id}`);
      setSelectedEntity(prev => ({ ...prev, bio: res.data.bio, facts: res.data.facts }));
    } catch (error) {
      console.error("Error fetching summary:", error);
      setSelectedEntity(prev => ({ ...prev, summary: "Could not load summary." }));
    } finally {
      setProfileLoading(false);
    }
  };

  useEffect(() => {
    fetchGraph().then(data => {
      if (data && data.nodes.filter(n => n.type === 'person').length === 0) {
        setMessages([{ role: 'agent', content: "Hello! I am Synapse. I am here to help you remember important things and answer your questions. \n\n Tell me about yourself so we can get started!" }]);
      } else {
        setMessages([{ role: 'agent', content: "Welcome back! Tell me something to remember or ask a question." }]);
      }
    });
  }, []);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, newMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post('http://localhost:8000/chat', { message: input });
      setMessages(prev => [...prev, { role: 'agent', content: res.data.response }]);

      // Auto-visualize if suggested by agent
      if (res.data.visualize_targets && res.data.visualize_targets.length > 0) {
        const targetName = res.data.visualize_targets[0]; // Just take first for now
        // Find matching node in graphData (case-insensitive)
        const targetNode = graphData.nodes.find(n => n.id.toLowerCase() === targetName.toLowerCase());
        if (targetNode) {
          handleNodeClick(targetNode);
        }
      }

      // Refresh graph after every message as it might have changed
      fetchGraph();
    } catch (error) {
      setMessages(prev => [...prev, { role: 'agent', content: 'Error: Could not reach the agent.' }]);
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <Head>
        <title>your second brain</title>
      </Head>



      <div style={styles.main}>
        <div style={styles.profileSection}>
          <div style={{ marginBottom: '20px', paddingLeft: '10px' }}>
            <h1 style={{ margin: 0, fontSize: '1.8em', letterSpacing: '4px', background: 'linear-gradient(to right, #00c6ff, #0072ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>S Y N A P S E</h1>
          </div>
          <div style={styles.scrollableContent}>
            <div style={styles.sectionHeader}>People</div>
            <div style={styles.grid}>
              {(() => {
                const people = graphData.nodes.filter(n => n.type === 'person');
                // Sort: Miccel first, then alphabetical
                people.sort((a, b) => {
                  if (a.id === 'Miccel') return -1;
                  if (b.id === 'Miccel') return 1;
                  return a.id.localeCompare(b.id);
                });

                if (people.length === 0) {
                  return <div style={{ color: '#666', fontStyle: 'italic', padding: '10px', gridColumn: '1 / -1' }}>No people found.</div>;
                }

                return people.map(node => {
                  const isUser = node.id === 'Miccel';
                  return (
                    <div
                      key={node.id}
                      style={{
                        ...styles.entityCard,
                        ...(isUser ? {
                          border: '1px solid #00c6ff',
                          boxShadow: '0 0 15px rgba(0, 198, 255, 0.2)',
                          backgroundColor: 'rgba(0, 198, 255, 0.1)'
                        } : {})
                      }}
                      onClick={() => handleNodeClick(node)}
                    >
                      <div style={styles.avatar}>{node.id.charAt(0).toUpperCase()}</div>
                      <div style={styles.entityName}>
                        {node.id}
                        {isUser && <span style={{ fontSize: '0.8em', color: '#00c6ff', marginLeft: '4px' }}>★</span>}
                      </div>
                    </div>
                  );
                });
              })()}
            </div>

            <div style={styles.sectionHeader}>Organizations</div>
            <div style={styles.grid}>
              {graphData.nodes.filter(n => n.type === 'organization').length > 0 ? (
                graphData.nodes.filter(n => n.type === 'organization').map(node => (
                  <div
                    key={node.id}
                    style={styles.entityCard}
                    onClick={() => handleNodeClick(node)}
                  >
                    <div style={{ ...styles.avatar, backgroundColor: '#ff0080' }}>{node.id.charAt(0).toUpperCase()}</div>
                    <div style={styles.entityName}>{node.id}</div>
                  </div>
                ))
              ) : (
                <div style={{ color: '#666', fontStyle: 'italic', padding: '10px', gridColumn: '1 / -1' }}>No organizations found.</div>
              )}
            </div>



            {/* 6-TIER SCHEMA MODAL */}
            {selectedEntity && (
              <div style={{
                position: 'fixed',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                backgroundColor: 'rgba(0,0,0,0.8)',
                zIndex: 1000,
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                backdropFilter: 'blur(5px)'
              }} onClick={() => setSelectedEntity(null)}>
                <div style={{
                  backgroundColor: '#0a0a0a',
                  width: '90%',
                  maxWidth: '500px',
                  borderRadius: '20px',
                  border: '1px solid #333',
                  boxShadow: '0 20px 50px rgba(0,0,0,0.8)',
                  padding: '30px',
                  maxHeight: '85vh',
                  overflowY: 'auto',
                  position: 'relative'
                }} onClick={e => e.stopPropagation()}>

                  {/* Header */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '25px', borderBottom: '1px solid #222', paddingBottom: '20px' }}>
                    <div>
                      <h2 style={{ margin: 0, color: '#fff', fontSize: '2em', display: 'flex', alignItems: 'center', gap: '10px' }}>
                        {selectedEntity.id}
                        {selectedEntity.id === 'Miccel' && <span style={{ color: '#00c6ff', fontSize: '0.6em' }}>★</span>}
                      </h2>
                      <div style={{ color: '#666', marginTop: '5px', textTransform: 'uppercase', fontSize: '0.8em', letterSpacing: '2px' }}>
                        {selectedEntity.type}
                      </div>
                    </div>
                    <button onClick={() => setSelectedEntity(null)} style={{ background: 'none', border: 'none', color: '#444', fontSize: '2em', cursor: 'pointer' }}>×</button>
                  </div>

                  {/* Schema Renderer */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    {(() => {
                      const facts = selectedEntity.facts || {};

                      // 1. DEDUPLICATION & CLEANUP
                      const cleanFacts = {};
                      Object.entries(facts).forEach(([key, values]) => {
                        let cleanKey = key;
                        if (key.endsWith(' (from)')) {
                          const baseKey = key.replace(' (from)', '');
                          // If base key exists (e.g. "is related to"), merge and skip this one
                          if (facts[baseKey]) {
                            return;
                          }
                          cleanKey = baseKey;
                        }

                        if (!cleanFacts[cleanKey]) cleanFacts[cleanKey] = new Set();
                        values.forEach(v => cleanFacts[cleanKey].add(v));
                      });

                      // Convert sets back to arrays
                      const displayFacts = {};
                      Object.keys(cleanFacts).forEach(k => {
                        displayFacts[k] = Array.from(cleanFacts[k]);
                      });

                      // 2. BIO SUMMARY GENERATOR
                      const generateSummary = () => {
                        if (selectedEntity.bio) return selectedEntity.bio;

                        const id = selectedEntity.id;
                        const occupation = (displayFacts['occupation'] || displayFacts['job'] || [])[0];
                        const org = (displayFacts['works at'] || [])[0];
                        const location = (displayFacts['location'] || displayFacts['lives in'] || [])[0];
                        const school = (displayFacts['studies'] || displayFacts['education'] || [])[0];

                        let parts = [];
                        if (occupation) parts.push(`is a ${occupation}`);
                        if (school) parts.push(`studying at ${school}`);
                        if (org) parts.push(`working at ${org}`);
                        if (location) parts.push(`based in ${location}`);

                        if (parts.length === 0) return "No summary available.";
                        return `${id} ${parts.join(', ')}.`;
                      };

                      // Helper to group attributes
                      const tiers = {
                        "Identity": ['name', 'nickname', 'role', 'gender', 'is'],
                        "Facts & Status": ['age', 'birthday', 'location', 'lives in', 'occupation', 'job', 'education', 'studies'],
                        "Social Connections": ['is related to', 'works at', 'member of'],
                        "Preferences & Skills": ['likes', 'dislikes', 'skills', 'good at', 'weakness'],
                        "Personality": ['personality', 'traits', 'habits'],
                        "Timeline Events": ['went to', 'moved to', 'started']
                      };

                      const allTierKeys = Object.values(tiers).flat();

                      return (
                        <>
                          {/* Bio Summary Section */}
                          <div style={{ marginBottom: '10px', fontStyle: 'italic', color: '#ccc', fontSize: '1.1em', paddingBottom: '15px', borderBottom: '1px solid #222' }}>
                            {generateSummary()}
                          </div>

                          {Object.entries(tiers).map(([tierName, keys]) => {
                            // Find matching attributes in CLEANED facts
                            const matches = Object.entries(displayFacts).filter(([k]) => keys.some(key => k.toLowerCase().includes(key)));
                            const hasData = matches.length > 0;
                            // ... rest of rendering uses 'matches'
                            return (
                              <div key={tierName} style={{ borderBottom: '1px solid #1a1a1a', paddingBottom: '15px' }}>
                                <h4 style={{ margin: '0 0 10px', color: '#444', fontSize: '0.75em', textTransform: 'uppercase', letterSpacing: '1.5px' }}>{tierName}</h4>
                                {hasData ? (
                                  <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                                    {matches.map(([k, v]) => (
                                      <div key={k} style={{ display: 'flex', gap: '10px', fontSize: '0.95em' }}>
                                        <span style={{ color: '#666', minWidth: '80px' }}>{k}:</span>
                                        <span style={{ color: '#e0e0e0' }}>{v.join(', ')}</span>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <div style={{ color: '#333', fontStyle: 'italic', fontSize: '0.9em' }}>No information recorded yet.</div>
                                )}
                              </div>
                            );
                          })}

                          {/* Catch-all for other attributes using CLEANED facts */}
                          {(() => {
                            const others = Object.entries(displayFacts).filter(([k]) => !allTierKeys.some(key => k.toLowerCase().includes(key)));
                            if (others.length > 0) {
                              return (
                                <div key="others" style={{ borderBottom: '1px solid #1a1a1a', paddingBottom: '15px' }}>
                                  <h4 style={{ margin: '0 0 10px', color: '#444', fontSize: '0.75em', textTransform: 'uppercase', letterSpacing: '1.5px' }}>Other Details</h4>
                                  <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '8px' }}>
                                    {others.map(([k, v]) => (
                                      <div key={k} style={{ display: 'flex', gap: '10px', fontSize: '0.95em' }}>
                                        <span style={{ color: '#666', minWidth: '80px' }}>{k}:</span>
                                        <span style={{ color: '#e0e0e0' }}>{v.join(', ')}</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              );
                            }
                            return null;
                          })()}
                        </>
                      );
                    })()}
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Permanent Graph View & Neural Console */}
          {/* Permanent Graph View & Neural Console */}
          <div style={{ marginTop: 'auto', paddingTop: '15px', borderTop: '1px solid #333' }}>
            <div style={{
              borderRadius: '16px',
              overflow: 'hidden',
              border: '1px solid #333',
              boxShadow: '0 10px 30px rgba(0,0,0,0.5)',
              position: 'relative',
              display: 'flex',        // Flexbox to enforce vertical stacking
              flexDirection: 'column'
            }}>
              <div style={{ padding: '10px', background: '#111', color: '#666', fontSize: '0.75em', textTransform: 'uppercase', textAlign: 'center', letterSpacing: '2px', fontWeight: 'bold', display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingLeft: '20px', paddingRight: '20px', flexShrink: 0 }}>
                <span>Neural Mind Map</span>
                {selectedEntity && <span style={{ color: '#0070f3', fontSize: '0.9em' }}>Filtering: {selectedEntity.id}</span>}
                {filterMode && <span style={{ color: '#00c6ff', fontSize: '0.9em' }}>Filtering: {filterMode.ids.join(', ')}</span>}
              </div>

              {/* Graph Container */}
              <div style={{
                height: '275px',
                backgroundColor: '#000',
                flexShrink: 0
              }}>
                <GraphView
                  height={275}
                  // width prop removed to let it fill container
                  data={
                    filterMode ? {
                      // Multi-Entity Filter: INTERSECTION (Common Neighbors)
                      nodes: graphData.nodes.filter(n => {
                        // 1. Is it one of the selected nodes?
                        if (filterMode.ids.includes(n.id)) return true;

                        // 2. Is it a common neighbor to ALL selected nodes?
                        // Check if this node 'n' has a link to every single node in filterMode.ids
                        const isCommon = filterMode.ids.every(targetId =>
                          graphData.links.some(l =>
                            (l.source.id === n.id && l.target.id === targetId) ||
                            (l.target.id === n.id && l.source.id === targetId)
                          )
                        );
                        return isCommon;
                      }),
                      // Only show links between the visible nodes
                      links: graphData.links.filter(l => {
                        // Re-calculate visibility to determine valid links
                        // (Optimized: In a real app we'd memoize this, but graphData is small)
                        const isVisible = (id) => filterMode.ids.includes(id) || filterMode.ids.every(targetId =>
                          graphData.links.some(link =>
                            (link.source.id === id && link.target.id === targetId) ||
                            (link.target.id === id && link.source.id === targetId)
                          )
                        );

                        return isVisible(l.source.id) && isVisible(l.target.id);
                      })
                    } : selectedEntity ? {
                      nodes: graphData.nodes.filter(n => n.id === selectedEntity.id || graphData.links.some(l => (l.source.id === selectedEntity.id && l.target.id === n.id) || (l.target.id === selectedEntity.id && l.source.id === n.id))),
                      links: graphData.links.filter(l => l.source.id === selectedEntity.id || l.target.id === selectedEntity.id)
                    } : graphData
                  }
                  onNodeClick={handleNodeClick}
                  linkLabel="relation"
                  nodeCanvasObject={(node, ctx, globalScale) => {
                    // 1. Draw Node Circle
                    const size = 5;
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
                    let color = '#666'; // Default Gray for concepts/locations
                    if (node.type === 'person') color = '#0070f3'; // Blue
                    if (node.type === 'organization') color = '#ff0080'; // Pink
                    ctx.fillStyle = color;
                    ctx.fill();

                    // 2. Draw Label strictly ABOVE
                    const label = node.id;
                    const fontSize = 12 / globalScale;
                    ctx.font = `${fontSize}px Sans-Serif`; // Removed bold to see if clarity improves
                    const textWidth = ctx.measureText(label).width;
                    const bckgDimensions = [textWidth, fontSize].map(n => n + fontSize * 0.2);

                    // -- LABEL POSITION SETTING --
                    // Change the number (currently 25) to move text higher/lower
                    // Negative '(- size - 25)' moves it UP
                    const textY = node.y - size - 2;

                    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
                    // Rect is drawn relative to textY (which is bottom of text)
                    ctx.fillRect(node.x - bckgDimensions[0] / 2, textY - bckgDimensions[1], bckgDimensions[0], bckgDimensions[1]);

                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'bottom';
                    ctx.fillStyle = '#fff';
                    ctx.fillText(label, node.x, textY);
                  }}
                  nodeCanvasObjectMode={() => 'replace'}
                />
              </div>

              {/* Graph Console Input: Fixed min-height and flex grow if needed */}
              <div style={{
                padding: '12px',
                background: '#0d0d0d', // Slightly lighter than graph for contrast
                borderTop: '1px solid #333',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                flexShrink: 0 // Do not shrink console
              }}>
                <form onSubmit={(e) => {
                  e.preventDefault();
                  const val = e.target.elements.cmd.value.trim().toLowerCase();
                  if (val) {
                    if (val === 'clear' || val === 'reset' || val === 'all') {
                      handleNodeClick(null);
                      setFilterMode(null);
                    } else {
                      // Support "A and B" or "A, B"
                      const terms = val.split(/ and | & |, /).map(t => t.trim()).filter(t => t);

                      const matchedNodes = [];
                      terms.forEach(term => {
                        const found = graphData.nodes.find(n => n.id.toLowerCase().includes(term));
                        if (found) matchedNodes.push(found);
                      });

                      if (matchedNodes.length > 0) {
                        // Whether 1 or many, just FILTER visually. Do not open profile.
                        setSelectedEntity(null);
                        setFilterMode({ type: 'multi', ids: matchedNodes.map(n => n.id) });
                      } else {
                        console.log("No nodes found for:", val);
                      }
                    }
                    e.target.elements.cmd.value = '';
                  }
                }} style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <span style={{ color: '#00c6ff', fontSize: '1.2em' }}>›</span>
                  <input
                    name="cmd"
                    placeholder="Filter Connections (e.g. 'A and B', 'Reset')"
                    style={{
                      background: '#1a1a1a', // Distinct input background
                      border: '1px solid #333',
                      borderRadius: '4px',
                      padding: '8px 12px',
                      color: '#00c6ff',
                      width: '100%',
                      fontFamily: 'monospace',
                      outline: 'none',
                      fontSize: '0.9em'
                    }}
                  />
                </form>
              </div>
            </div>
          </div>
        </div>



        <div style={styles.chatSection}>
          <div style={styles.chatWindow}>
            {messages.map((msg, i) => (
              <div key={i} style={{
                ...styles.message,
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                backgroundColor: msg.role === 'user' ? '#0070f3' : '#333',
                color: '#fff',
                border: msg.role === 'agent' ? '1px solid #444' : 'none',
              }}>
                {msg.content}
              </div>
            ))}
            {loading && <div style={styles.loading}>Thinking...</div>}
          </div>

          <form onSubmit={sendMessage} style={styles.inputForm}>
            <input
              style={styles.input}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type something..."
            />
            <button type="submit" style={styles.button} disabled={loading}>Send</button>
          </form>
        </div>
      </div>
      <style jsx global>{`
        body {
          margin: 0;
          background-color: #0a0a0a;
          color: #e0e0e0;
        }
      `}</style>
    </div >
  );
}

const styles = {
  container: {
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: '#0a0a0a',
    color: '#e0e0e0',
  },
  header: {
    padding: '20px 40px',
    borderBottom: '1px solid #333',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-start',
    background: 'rgba(20, 20, 20, 0.8)',
    backdropFilter: 'blur(10px)',
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  main: {
    flex: 1,
    display: 'flex',
    padding: '20px',
    gap: '40px',
    maxWidth: '1200px',
    margin: '0 auto',
    width: '100%',
    minHeight: 0, // Ensure main container respects 100vh
  },
  chatSection: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    maxWidth: '500px',
    minWidth: '350px',
    background: '#111',
    borderRadius: '16px',
    border: '1px solid #222',
    overflow: 'hidden',
    boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
    minHeight: 0, // Ensure chat section does not overflow parent
  },
  chatWindow: {
    flex: 1,
    padding: '20px',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '15px',
    background: 'linear-gradient(180deg, rgba(20,20,20,0) 0%, rgba(20,20,20,0.5) 100%)',
    minHeight: 0, // Ensure scrolling works
  },
  message: {
    padding: '12px 18px',
    borderRadius: '18px',
    maxWidth: '85%',
    lineHeight: '1.5',
    fontSize: '0.95em',
    boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
  },
  inputForm: {
    display: 'flex',
    gap: '10px',
    padding: '15px',
    borderTop: '1px solid #222',
    background: '#151515',
  },
  input: {
    flex: 1,
    padding: '12px 15px',
    borderRadius: '25px',
    border: '1px solid #333',
    background: '#222',
    color: '#fff',
    outline: 'none',
    fontSize: '0.95em',
  },
  button: {
    padding: '10px 25px',
    borderRadius: '25px',
    border: 'none',
    background: 'linear-gradient(135deg, #0070f3 0%, #00c6ff 100%)',
    color: 'white',
    cursor: 'pointer',
    fontWeight: '600',
    boxShadow: '0 4px 15px rgba(0,112,243,0.3)',
  },
  profileSection: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflowY: 'auto', // Changed to auto to prevent cutting off bottom content
  },
  scrollableContent: {
    flex: 1,
    overflowY: 'auto',
    paddingRight: '10px',
    minHeight: 0, // Ensure flex child can shrink below content size
  },
  sectionHeader: {
    fontSize: '0.9em',
    textTransform: 'uppercase',
    letterSpacing: '1.5px',
    fontWeight: '700',
    color: '#888',
    marginTop: '30px',
    marginBottom: '15px',
    paddingLeft: '5px',
    borderBottom: 'none',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
    gap: '12px',
  },
  entityCard: {
    backgroundColor: 'rgba(30, 30, 30, 0.6)',
    backdropFilter: 'blur(5px)',
    border: '1px solid #333',
    borderRadius: '12px',
    padding: '15px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  avatar: {
    width: '40px',
    height: '40px',
    borderRadius: '12px',
    backgroundColor: '#333',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.2em',
    marginBottom: '12px',
    fontWeight: 'bold',
    boxShadow: '0 4px 10px rgba(0,0,0,0.3)',
  },
  entityName: {
    fontWeight: '500',
    color: '#eee',
    textAlign: 'center',
    fontSize: '0.85em',
    width: '100%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    marginTop: '8px'
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.8)',
    backdropFilter: 'blur(5px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  },
  modalContent: {
    backgroundColor: '#1a1a1a',
    padding: '30px',
    borderRadius: '20px',
    width: '90%',
    maxWidth: '500px',
    boxShadow: '0 20px 50px rgba(0,0,0,0.5)',
    border: '1px solid #333',
    position: 'relative',
    color: '#ddd',
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
    borderBottom: '1px solid #333',
    paddingBottom: '15px',
  },
  modalBody: {
    lineHeight: '1.6',
    color: '#bbb',
    fontSize: '1em',
  },
  closeButton: {
    background: 'none',
    border: 'none',
    fontSize: '2em',
    cursor: 'pointer',
    color: '#666',
    transition: 'color 0.2s',
  },
  loading: { alignSelf: 'center', color: '#666', fontStyle: 'italic', fontSize: '0.9em' },
};
