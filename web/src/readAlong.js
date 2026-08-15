// maps playback position to sentences by character fraction, assuming uniform speech rate
import { stripVoiceTags } from './voiceTags.js';

// sentence enders with trailing quotes/brackets, or a line break
const SENTENCE_BREAK = /[.!?…]+["')\]]*\s+|\n+/g;

// contiguous segments covering every character, so rendering the ranges reproduces the text exactly
export function segmentSentences(text) {
    const sentences = [];
    if (!text) {
        return sentences;
    }

    const push = (start, end) => {
        if (end <= start) {
            return;
        }
        const slice = text.slice(start, end);
        // spoken counts the characters that take audio time, so voice tags are out
        sentences.push({ start, end, spoken: stripVoiceTags(slice).trim().length });
    };

    let start = 0;
    SENTENCE_BREAK.lastIndex = 0;
    let match;
    while ((match = SENTENCE_BREAK.exec(text)) !== null) {
        push(start, match.index + match[0].length);
        start = match.index + match[0].length;
    }
    push(start, text.length);
    return sentences;
}

export function totalSpoken(sentences) {
    return sentences.reduce((sum, sentence) => sum + sentence.spoken, 0);
}

const SENTENCE_PAUSE_WEIGHT = 14;

export function weightOf(sentence) {
    return sentence.spoken > 0 ? sentence.spoken + SENTENCE_PAUSE_WEIGHT : 0;
}

function totalWeight(sentences) {
    return sentences.reduce((sum, sentence) => sum + weightOf(sentence), 0);
}

// the sentence playing at a fraction, skipping unspoken segments; -1 when nothing is speakable
export function sentenceIndexAt(sentences, fraction) {
    const total = totalWeight(sentences);
    if (total <= 0 || !Number.isFinite(fraction)) {
        return -1;
    }

    const target = Math.min(Math.max(fraction, 0), 1) * total;
    let accumulated = 0;
    let lastSpeakable = -1;
    for (let i = 0; i < sentences.length; i++) {
        if (sentences[i].spoken === 0) {
            continue;
        }
        lastSpeakable = i;
        accumulated += weightOf(sentences[i]);
        if (target < accumulated) {
            return i;
        }
    }
    return lastSpeakable;
}

// the playback fraction where a sentence begins
export function sentenceStartFraction(sentences, index) {
    const total = totalWeight(sentences);
    if (total <= 0) {
        return 0;
    }

    let accumulated = 0;
    for (let i = 0; i < index && i < sentences.length; i++) {
        accumulated += weightOf(sentences[i]);
    }
    return accumulated / total;
}
