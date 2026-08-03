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

/** The tag every word after it answers to, or nothing if the text opens with prose. */
export function leadingVoiceTag(text) {
    const match = String(text).match(new RegExp(`^\\s*${TAG_SOURCE}`, 'i'));
    return match ? match[1] : '';
}

function escapeRegExp(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Drops matching tags and the whitespace they owned, so removing them cannot leave
 * double spaces or a blank line where a tag sat on its own.
 */
function stripPattern(text, source) {
    return String(text)
        .replace(new RegExp(`^[ \\t]*${source}[ \\t]*\\r?\\n`, 'gim'), '')
        .replace(new RegExp(`[ \\t]*${source}[ \\t]*(?=\\r?\\n|$)`, 'gi'), '')
        .replace(new RegExp(`${source}[ \\t]*`, 'gi'), '');
}

export function stripVoiceTags(text) {
    return stripPattern(text, TAG_SOURCE);
}

/** Drops one speaker's tags and leaves everyone else's alone. */
export function removeVoiceTagsFor(text, voice) {
    return stripPattern(text, String.raw`\[voice:\s*${escapeRegExp(voice)}\s*\]`);
}

/** Follows a rename through the text, so the tags already placed keep pointing at the same voice. */
export function renameVoiceTags(text, from, to) {
    const pattern = new RegExp(String.raw`\[voice:\s*${escapeRegExp(from)}\s*\]`, 'gi');
    return String(text).replace(pattern, formatVoiceTag(to));
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

export const CAST_NAME_PATTERN = /^[A-Za-z0-9_-]{1,24}$/;

/**
 * A new member stands for its own mix, so the tag placed in the text is the plain
 * one the server would take anyway and nothing has to be defined. Renaming is what
 * turns a member into an alias. Membership is by exact mix string, so af_bella and
 * af_bella(2) are separate members and the same mix cannot be staged twice.
 */
export function addToCast(cast, mix) {
    const value = String(mix || '').trim();
    if (!value || cast.some((member) => member.mix === value)) {
        return cast;
    }

    return [...cast, { name: value, mix: value }];
}

export function removeFromCast(cast, name) {
    return cast.filter((member) => member.name !== name);
}

export function renameCastMember(cast, name, next) {
    return cast.map((member) => (member.name === name ? { ...member, name: next } : member));
}

export function updateCastMix(cast, name, mix) {
    return cast.map((member) => (member.name === name ? { ...member, mix } : member));
}

/** Only a name that stands for something else has to travel with the request. */
export function castAliases(cast) {
    return cast.reduce((aliases, member) => {
        if (member.name !== member.mix) {
            aliases[member.name] = member.mix;
        }
        return aliases;
    }, {});
}

/** Reads a mix string back into voice/weight pairs, so a cast member can return to the mixer. */
export function parseVoiceMix(mix) {
    return String(mix || '')
        .split('+')
        .map((part) => part.trim())
        .filter(Boolean)
        .map((part) => {
            const weighted = part.match(/^(.+?)\s*\(\s*([\d.]+)\s*\)$/);
            return weighted
                ? { voice: weighted[1].trim(), weight: parseFloat(weighted[2]) || 1 }
                : { voice: part, weight: 1 };
        });
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
