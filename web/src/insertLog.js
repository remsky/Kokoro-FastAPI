export const INSERT_LOG_LIMIT = 8;

/** Newest first, capped. */
export function recordInsert(log, entry) {
    return [entry, ...log].slice(0, INSERT_LOG_LIMIT);
}

/** Finds where an insert sits now: exact at the recorded offset, else the nearest occurrence. */
export function locateInsert(fullText, entry) {
    const text = String(fullText);
    if (text.slice(entry.offset, entry.offset + entry.inserted.length) === entry.inserted) {
        return entry.offset;
    }
    let best = -1;
    for (let i = text.indexOf(entry.inserted); i !== -1; i = text.indexOf(entry.inserted, i + 1)) {
        if (best === -1 || Math.abs(i - entry.offset) < Math.abs(best - entry.offset)) {
            best = i;
        }
    }
    return best;
}
