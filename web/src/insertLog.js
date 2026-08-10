// sure: exact at the recorded offset, or the only occurrence left
export function locateInsert(fullText, entry) {
    const text = String(fullText);
    if (text.slice(entry.offset, entry.offset + entry.inserted.length) === entry.inserted) {
        return { at: entry.offset, sure: true };
    }
    const hits = [];
    for (let i = text.indexOf(entry.inserted); i !== -1; i = text.indexOf(entry.inserted, i + 1)) {
        hits.push(i);
    }
    if (!hits.length) {
        return { at: -1, sure: true };
    }
    const at = hits.reduce((best, i) => (Math.abs(i - entry.offset) < Math.abs(best - entry.offset) ? i : best));
    return { at, sure: hits.length === 1 };
}
