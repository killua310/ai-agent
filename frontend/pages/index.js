import { useState, useEffect } from 'react';
import axios from 'axios';

import Head from 'next/head';

export default function Home() {
  const [messages, setMessages] = useState([
    { role: 'agent', content: 'Hello! I am your Second Brain. Tell me something to remember or ask a question.' }
  ]);
  const [input, setInput] = useState('');
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [loading, setLoading] = useState(false);

  // Profile State
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);

  const fetchGraph = async () => {
    try {
      const res = await axios.get('http://localhost:8000/graph');
      setGraphData(res.data);
    } catch (error) {
      console.error("Error fetching graph:", error);
    }
  };

  const handleNodeClick = async (node) => {
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
    fetchGraph();
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
        <title>Second Brain Agent</title>
      </Head>

      <div style={styles.header}>
        <h1>your second brain</h1>
      </div>

      <div style={styles.main}>
        <div style={styles.profileSection}>
          <div style={styles.scrollableContent}>
            <div style={styles.sectionHeader}>People</div>
            <div style={styles.grid}>
              {graphData.nodes.filter(n => n.type === 'person').map(node => (
                <div
                  key={node.id}
                  style={styles.entityCard}
                  onClick={() => handleNodeClick(node)}
                >
                  <div style={styles.avatar}>{node.id.charAt(0).toUpperCase()}</div>
                  <div style={styles.entityName}>{node.id}</div>
                </div>
              ))}
            </div>

            <div style={styles.sectionHeader}>Organizations</div>
            <div style={styles.grid}>
              {graphData.nodes.filter(n => n.type === 'organization').map(node => (
                <div
                  key={node.id}
                  style={{ ...styles.entityCard, borderLeft: '4px solid #ff0080' }}
                  onClick={() => handleNodeClick(node)}
                >
                  <div style={{ ...styles.avatar, backgroundColor: '#ff0080' }}>{node.id.charAt(0).toUpperCase()}</div>
                  <div style={styles.entityName}>{node.id}</div>
                </div>
              ))}
            </div>

            {/* Catch-all for others */}
            {graphData.nodes.filter(n => !['person', 'organization'].includes(n.type)).length > 0 && (
              <>
                <div style={styles.sectionHeader}>Other</div>
                <div style={styles.grid}>
                  {graphData.nodes.filter(n => !['person', 'organization'].includes(n.type)).map(node => (
                    <div
                      key={node.id}
                      style={{ ...styles.entityCard, backgroundColor: '#f9f9f9' }}
                      onClick={() => handleNodeClick(node)}
                    >
                      <div style={{ ...styles.avatar, backgroundColor: '#888' }}>{node.id.charAt(0).toUpperCase()}</div>
                      <div style={styles.entityName}>{node.id}</div>
                    </div>
                  ))}
                </div>
              </>
            )}

            {selectedEntity && (
              <div style={styles.modalOverlay}>
                <div style={styles.modalContent}>
                  <div style={styles.modalHeader}>
                    <h3>{selectedEntity.id}</h3>
                    <button onClick={() => setSelectedEntity(null)} style={styles.closeButton}>×</button>
                  </div>
                  <div style={styles.modalBody}>
                    {profileLoading ? (
                      <p style={{ color: '#888' }}>Generating profile...</p>
                    ) : (
                      <>
                        <p style={{ marginBottom: '15px' }}>{selectedEntity.bio}</p>

                        {selectedEntity.facts && Object.keys(selectedEntity.facts).length > 0 && (
                          <div>
                            <h4 style={{ margin: '10px 0 5px', color: '#555', fontSize: '0.9em' }}>Known Facts</h4>
                            <div style={{
                              fontSize: '0.85em',
                              color: '#666',
                              maxHeight: '150px',
                              overflowY: 'auto'
                            }}>
                              {Object.entries(selectedEntity.facts).map(([key, values]) => (
                                <div key={key} style={{ marginBottom: '6px' }}>
                                  <span style={{ fontWeight: 'bold', color: '#333' }}>{key}:</span>
                                  {' '}{values.join(', ')}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>


        <div style={styles.chatSection}>
          <div style={styles.chatWindow}>
            {messages.map((msg, i) => (
              <div key={i} style={{
                ...styles.message,
                alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
                backgroundColor: msg.role === 'user' ? '#0070f3' : '#e5e5ea',
                color: msg.role === 'user' ? 'white' : 'black',
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
    </div>


  );
}

const styles = {
  container: { fontFamily: 'sans-serif', height: '100vh', display: 'flex', flexDirection: 'column' },
  header: { padding: '20px', borderBottom: '1px solid #eee', textAlign: 'center' },
  main: { flex: 1, display: 'flex', padding: '20px', gap: '20px' },
  chatSection: { flex: 1, display: 'flex', flexDirection: 'column', maxWidth: '600px' },
  chatWindow: {
    flex: 1,
    border: '1px solid #eee',
    borderRadius: '8px',
    padding: '20px',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    marginBottom: '20px'
  },
  message: { padding: '10px 15px', borderRadius: '15px', maxWidth: '80%' },
  inputForm: { display: 'flex', gap: '10px' },
  input: { flex: 1, padding: '10px', borderRadius: '4px', border: '1px solid #ccc' },
  button: { padding: '10px 20px', borderRadius: '4px', border: 'none', background: '#0070f3', color: 'white', cursor: 'pointer' },
  input: { flex: 1, padding: '10px', borderRadius: '4px', border: '1px solid #ccc' },
  button: { padding: '10px 20px', borderRadius: '4px', border: 'none', background: '#0070f3', color: 'white', cursor: 'pointer' },
  profileSection: { flex: 1, display: 'flex', flexDirection: 'column', overflowY: 'hidden' }, // Prevent double scroll
  scrollableContent: { flex: 1, overflowY: 'auto', paddingRight: '10px' },
  sectionHeader: {
    fontSize: '1.2em',
    fontWeight: 'bold',
    color: '#333',
    marginTop: '20px',
    marginBottom: '10px',
    paddingLeft: '10px',
    borderBottom: '2px solid #eee',
    paddingBottom: '5px'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
    gap: '20px',
    padding: '10px'
  },
  entityCard: {
    backgroundColor: '#fff',
    border: '1px solid #e0e0e0',
    borderRadius: '12px',
    padding: '20px',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    cursor: 'pointer',
    transition: 'transform 0.2s, box-shadow 0.2s',
    boxShadow: '0 2px 5px rgba(0,0,0,0.05)'
  },
  avatar: {
    width: '50px',
    height: '50px',
    borderRadius: '50%',
    backgroundColor: '#0070f3',
    color: '#fff',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.5em',
    marginBottom: '10px',
    fontWeight: 'bold'
  },
  entityName: {
    fontWeight: '600',
    color: '#333',
    textAlign: 'center',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    width: '100%'
  },
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0,0,0,0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000
  },
  modalContent: {
    backgroundColor: '#fff',
    padding: '30px',
    borderRadius: '12px',
    width: '90%',
    maxWidth: '500px',
    boxShadow: '0 5px 15px rgba(0,0,0,0.3)',
    position: 'relative'
  },
  modalHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
    borderBottom: '1px solid #eee',
    paddingBottom: '10px'
  },
  modalBody: {
    lineHeight: '1.6',
    color: '#444',
    fontSize: '1.1em'
  },
  closeButton: {
    background: 'none',
    border: 'none',
    fontSize: '2em',
    cursor: 'pointer',
    color: '#999',
    lineHeight: '0.5'
  },
  loading: { alignSelf: 'center', color: '#888' },
  stats: { marginTop: '10px', color: '#666', fontSize: '0.9em' }
};
