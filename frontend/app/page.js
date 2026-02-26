"use client";

import { useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLE_SEQUENCE =
  "MFPQSRHFGATVIALTLAAGQAQADNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS";

export default function Home() {
  const [sequence, setSequence] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    setError("");
    setResult(null);

    const seq = sequence.trim();
    if (seq.length < 20) {
      setError("Please enter at least 20 amino acid characters.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: seq, model_name: "rf_kmer" }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Prediction failed");
      }

      setResult(await res.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadExample = () => {
    setSequence(EXAMPLE_SEQUENCE);
    setResult(null);
    setError("");
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>PlasticDeg Predictor</h1>
        <p style={styles.subtitle}>
          AI-based prediction of plastic-degrading potential in microbial
          enzymes using genomic data
        </p>
      </header>

      <main style={styles.main}>
        <div style={styles.card}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <h2 style={styles.cardTitle}>Enter Protein Sequence</h2>
            <button onClick={loadExample} style={styles.exampleBtn}>
              Load Example (PETase)
            </button>
          </div>

          <textarea
            value={sequence}
            onChange={(e) => setSequence(e.target.value)}
            placeholder="Paste your amino acid sequence here (single letter code, e.g. MFPQSRH...)"
            rows={8}
            style={styles.textarea}
          />

          <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
            <button
              onClick={handlePredict}
              disabled={loading}
              style={{
                ...styles.predictBtn,
                opacity: loading ? 0.6 : 1,
              }}
            >
              {loading ? "Predicting..." : "Predict"}
            </button>
            <span style={styles.seqLength}>
              {sequence.replace(/\s/g, "").length} amino acids
            </span>
          </div>
        </div>

        {error && (
          <div style={styles.errorCard}>
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div style={styles.resultCard}>
            <h2 style={styles.cardTitle}>Prediction Result</h2>

            <div
              style={{
                ...styles.predictionBadge,
                backgroundColor:
                  result.prediction === "Plastic-Degrading"
                    ? "#059669"
                    : "#dc2626",
              }}
            >
              {result.prediction}
            </div>

            <div style={styles.metricsGrid}>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Confidence</span>
                <span style={styles.metricValue}>
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>P(Plastic-Degrading)</span>
                <span style={styles.metricValue}>
                  {(result.probability_positive * 100).toFixed(1)}%
                </span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>P(Non-Degrading)</span>
                <span style={styles.metricValue}>
                  {(result.probability_negative * 100).toFixed(1)}%
                </span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Sequence Length</span>
                <span style={styles.metricValue}>
                  {result.sequence_length} aa
                </span>
              </div>
            </div>

            <p style={styles.modelNote}>Model: {result.model_used}</p>
          </div>
        )}

        <div style={styles.infoCard}>
          <h3 style={{ margin: "0 0 8px 0" }}>How it works</h3>
          <ol style={{ margin: 0, paddingLeft: "20px", lineHeight: 1.8 }}>
            <li>
              Paste a protein/enzyme amino acid sequence into the text box
            </li>
            <li>
              The system extracts k-mer frequency features (amino acid and
              dipeptide composition)
            </li>
            <li>
              A trained Random Forest model classifies the sequence as
              plastic-degrading or not
            </li>
            <li>Results show the prediction with confidence scores</li>
          </ol>
        </div>
      </main>

      <footer style={styles.footer}>
        <p>
          Mini Project — AI-Based Prediction of Plastic-Degrading Potential in
          Bacteria and Fungi
        </p>
      </footer>
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    backgroundColor: "#f0fdf4",
    display: "flex",
    flexDirection: "column",
  },
  header: {
    background: "linear-gradient(135deg, #065f46, #047857)",
    color: "white",
    padding: "40px 20px",
    textAlign: "center",
  },
  title: { margin: 0, fontSize: "2rem" },
  subtitle: { margin: "8px 0 0", opacity: 0.9, fontSize: "1rem" },
  main: {
    flex: 1,
    maxWidth: "720px",
    margin: "0 auto",
    padding: "24px 16px",
    width: "100%",
    boxSizing: "border-box",
  },
  card: {
    backgroundColor: "white",
    borderRadius: "12px",
    padding: "24px",
    marginBottom: "16px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
  cardTitle: { margin: "0 0 16px", fontSize: "1.1rem", color: "#1e293b" },
  textarea: {
    width: "100%",
    padding: "12px",
    borderRadius: "8px",
    border: "1px solid #d1d5db",
    fontFamily: "monospace",
    fontSize: "13px",
    resize: "vertical",
    marginBottom: "12px",
    boxSizing: "border-box",
  },
  predictBtn: {
    backgroundColor: "#059669",
    color: "white",
    border: "none",
    padding: "10px 32px",
    borderRadius: "8px",
    fontSize: "15px",
    fontWeight: 600,
    cursor: "pointer",
  },
  exampleBtn: {
    backgroundColor: "transparent",
    color: "#059669",
    border: "1px solid #059669",
    padding: "6px 14px",
    borderRadius: "6px",
    fontSize: "13px",
    cursor: "pointer",
  },
  seqLength: { fontSize: "13px", color: "#6b7280" },
  errorCard: {
    backgroundColor: "#fef2f2",
    color: "#991b1b",
    borderRadius: "12px",
    padding: "16px 24px",
    marginBottom: "16px",
  },
  resultCard: {
    backgroundColor: "white",
    borderRadius: "12px",
    padding: "24px",
    marginBottom: "16px",
    boxShadow: "0 1px 3px rgba(0,0,0,0.1)",
  },
  predictionBadge: {
    display: "inline-block",
    color: "white",
    padding: "8px 20px",
    borderRadius: "20px",
    fontSize: "16px",
    fontWeight: 600,
    marginBottom: "20px",
  },
  metricsGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "12px",
  },
  metric: {
    backgroundColor: "#f8fafc",
    borderRadius: "8px",
    padding: "12px",
    display: "flex",
    flexDirection: "column",
  },
  metricLabel: { fontSize: "12px", color: "#6b7280", marginBottom: "4px" },
  metricValue: { fontSize: "18px", fontWeight: 600, color: "#1e293b" },
  modelNote: { fontSize: "12px", color: "#9ca3af", margin: "16px 0 0" },
  infoCard: {
    backgroundColor: "#ecfdf5",
    borderRadius: "12px",
    padding: "20px 24px",
    fontSize: "14px",
    color: "#065f46",
  },
  footer: {
    textAlign: "center",
    padding: "20px",
    fontSize: "13px",
    color: "#6b7280",
  },
};
