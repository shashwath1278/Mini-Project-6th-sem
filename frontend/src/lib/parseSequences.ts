/**
 * Parse FASTA (multiple records) or a single raw amino-acid string.
 * IDs default to query_1, query_2, ...
 */
export function parseSequencesInput(raw: string): { id: string; sequence: string }[] {
  const text = raw.trim();
  if (!text) return [];

  const lines = text.split(/\r?\n/);
  const hasFastaHeader = lines.some((l) => l.trim().startsWith(">"));

  if (!hasFastaHeader) {
    const seq = lines.join("").trim();
    if (!seq) return [];
    return [{ id: "query_1", sequence: seq }];
  }

  const out: { id: string; sequence: string }[] = [];
  let currentId = "query_1";
  let buf: string[] = [];

  const flush = () => {
    const seq = buf.join("").trim();
    if (seq) out.push({ id: currentId, sequence: seq });
    buf = [];
  };

  for (const line of lines) {
    const t = line.trim();
    if (!t) continue;
    if (t.startsWith(">")) {
      flush();
      currentId = t.slice(1).trim().split(/\s+/)[0] || `query_${out.length + 1}`;
    } else {
      buf.push(t);
    }
  }
  flush();
  return out;
}
