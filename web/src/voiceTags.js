// mirrors VOICE_TAG_PATTERN in text_processor.py, which decides what is really valid
const TAG_SOURCE = String.raw`\[voice:\s*([a-zA-Z0-9_][a-zA-Z0-9_+\-(). ]*?)\s*\]`;

function tagPattern(flags = 'gi') {
    return new RegExp(TAG_SOURCE, flags);
}

export function formatVoiceTag(voice) {
    return `[voice:${voice}]`;
}

export function countVoiceTags(text) {
    return (String(text).match(tagPattern()) || []).length;
}

export function hasVoiceTags(text) {
    return tagPattern().test(String(text));
}

/**
 * Drops every tag and the whitespace it owned, so removing them cannot leave
 * double spaces or a blank line where a tag sat on its own.
 */
export function stripVoiceTags(text) {
    return String(text)
        .replace(new RegExp(`^[ \\t]*${TAG_SOURCE}[ \\t]*\\r?\\n`, 'gim'), '')
        .replace(new RegExp(`${TAG_SOURCE}[ \\t]*`, 'gi'), '');
}

/** Puts a tag at the front so enabling tags shows the syntax rather than describing it. */
export function seedVoiceTag(text, voice) {
    const source = String(text);
    if (!voice || hasVoiceTags(source)) {
        return { text: source, changed: false };
    }
    return { text: `${formatVoiceTag(voice)} ${source.replace(/^[ \t]+/, '')}`, changed: true };
}

/**
 * Nearest whitespace boundary, ties going backwards, so a tag inserted from a
 * caret parked mid-word lands in front of that word instead of splitting it.
 */
function snapToBoundary(text, cursor) {
    const inWord = cursor > 0 && cursor < text.length
        && !/\s/.test(text[cursor - 1]) && !/\s/.test(text[cursor]);
    if (!inWord) {
        return cursor;
    }

    let back = cursor;
    while (back > 0 && !/\s/.test(text[back - 1])) {
        back--;
    }
    let forward = cursor;
    while (forward < text.length && !/\s/.test(text[forward])) {
        forward++;
    }
    return cursor - back <= forward - cursor ? back : forward;
}

/** Returns the rewritten text and where the caret should land after it. */
export function insertVoiceTag(text, cursor, voice) {
    const source = String(text);
    const at = snapToBoundary(source, Math.max(0, Math.min(Number(cursor) || 0, source.length)));
    const before = source.slice(0, at);
    const after = source.slice(at);
    const lead = before && !/\s$/.test(before) ? ' ' : '';
    const trail = /^\s/.test(after) ? '' : ' ';
    const inserted = `${lead}${formatVoiceTag(voice)}${trail}`;

    return { text: before + inserted + after, cursor: at + inserted.length };
}
