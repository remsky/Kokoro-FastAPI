import { weightOf } from './readAlong.js';

const WORD_PATTERN = /[\p{L}\p{N}']+/gu;
const TAG_PATTERN = /\[voice:\s*[a-zA-Z0-9_][a-zA-Z0-9_+\-(). ]*?\s*\]|\[pause:\d+(?:\.\d+)?s\]/gi;

const ANCHOR_WORDS = 5;
const SEARCH_BACK = 40;
const SEARCH_AHEAD = 80;

function rawTokens(text) {
    const tags = [];
    for (const match of text.matchAll(TAG_PATTERN)) {
        tags.push([match.index, match.index + match[0].length]);
    }

    const tokens = [];
    for (const match of text.matchAll(WORD_PATTERN)) {
        if (tags.some(([start, end]) => match.index >= start && match.index < end)) {
            continue;
        }
        tokens.push({ word: match[0].toLowerCase(), offset: match.index });
    }
    return tokens;
}

function findAnchor(raw, target, from, to) {
    const need = Math.max(1, target.length - 1);
    let best = -1;
    let bestScore = 0;
    for (let position = Math.max(0, from); position <= to && position < raw.length; position++) {
        let score = 0;
        for (let j = 0; j < target.length && position + j < raw.length; j++) {
            if (raw[position + j].word === target[j]) {
                score++;
            }
        }
        if (score >= need) {
            return position;
        }
        if (score > bestScore) {
            bestScore = score;
            best = position;
        }
    }
    return bestScore >= Math.ceil(target.length / 2) ? best : -1;
}

function fillSegment(sentences, times, charStart, charEnd, timeStart, timeEnd) {
    const inside = [];
    for (let i = 0; i < sentences.length; i++) {
        const mid = (sentences[i].start + sentences[i].end) / 2;
        if (sentences[i].spoken > 0 && mid >= charStart && mid < charEnd) {
            inside.push(i);
        }
    }

    const total = inside.reduce((sum, i) => sum + weightOf(sentences[i]), 0);
    if (!total) {
        return;
    }

    let accumulated = 0;
    for (const i of inside) {
        times[i] = timeStart + ((timeEnd - timeStart) * accumulated) / total;
        accumulated += weightOf(sentences[i]);
    }
}

export function alignChunks(sourceText, sentences, chunks) {
    const spoken = (chunks || []).filter((chunk) => chunk && chunk.text && chunk.end > chunk.start);
    if (!spoken.length || !sentences.length) {
        return null;
    }

    const raw = rawTokens(sourceText);
    if (!raw.length) {
        return null;
    }

    const anchors = new Array(spoken.length).fill(-1);
    let cursor = 0;
    let expected = 0;
    for (let k = 0; k < spoken.length; k++) {
        const words = spoken[k].text.toLowerCase().match(WORD_PATTERN) || [];
        const target = words.slice(0, ANCHOR_WORDS);
        const found = target.length
            ? findAnchor(raw, target, Math.max(cursor, expected - SEARCH_BACK), expected + SEARCH_AHEAD)
            : -1;
        if (found >= 0) {
            anchors[k] = found;
            cursor = found + 1;
            expected = found + words.length;
        } else {
            expected += words.length;
        }
    }
    if (anchors[0] < 0) {
        anchors[0] = 0;
    }

    const matched = [];
    for (let k = 0; k < spoken.length; k++) {
        if (anchors[k] >= 0) {
            matched.push(k);
        }
    }

    const times = new Array(sentences.length).fill(null);
    for (let m = 0; m < matched.length; m++) {
        const k = matched[m];
        const next = matched[m + 1];
        const charStart = raw[anchors[k]].offset;
        const charEnd = next !== undefined ? raw[anchors[next]].offset : sourceText.length;
        const timeEnd = next !== undefined ? spoken[next - 1].end : spoken[spoken.length - 1].end;
        fillSegment(sentences, times, charStart, charEnd, spoken[k].start, timeEnd);
    }
    return times;
}

export function sentenceIndexAtTime(times, time) {
    let index = -1;
    for (let i = 0; i < times.length; i++) {
        if (times[i] !== null && times[i] <= time) {
            index = i;
        }
    }
    if (index >= 0) {
        return index;
    }
    return times.findIndex((value) => value !== null);
}
