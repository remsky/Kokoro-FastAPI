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

/** Whether one speaker is still spoken anywhere, so dropping it can be refused while it is. */
export function hasVoiceTagFor(text, voice) {
    return new RegExp(String.raw`\[voice:\s*${escapeRegExp(voice)}\s*\]`, 'i').test(String(text));
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

// first char anchored like TAG_SOURCE, so a legal cast name always makes a matchable tag
export const CAST_NAME_PATTERN = /^[A-Za-z0-9_][A-Za-z0-9_-]{0,23}$/;

export function suggestCastName(mix, rate) {
    const value = String(mix || '').trim();
    const pace = normalizeRate(rate);
    if (!pace || !value) {
        return value;
    }
    const suffix = `__${String(Math.round(pace * 100)).padStart(3, '0')}`;
    const base = value.replace(/[^A-Za-z0-9_-]+/g, '_');
    return base.slice(0, 24 - suffix.length) + suffix;
}

/** Membership is keyed by name, so one mix can back several members. */
export function addToCast(cast, mix, rate, name) {
    const value = String(mix || '').trim();
    const pace = normalizeRate(rate);
    const chosen = String(name || '').trim() || suggestCastName(value, pace);
    if (!value || cast.some((member) => member.name === chosen)) {
        return cast;
    }

    return [...cast, { name: chosen, mix: value, ...(pace ? { rate: pace } : {}) }];
}

export function removeFromCast(cast, name) {
    return cast.filter((member) => member.name !== name);
}

export function renameCastMember(cast, name, next) {
    return cast.map((member) => (member.name === name ? { ...member, name: next } : member));
}

export function updateCastMix(cast, name, mix, rate) {
    const pace = normalizeRate(rate);
    return cast.map((member) => (member.name === name
        ? { name: member.name, mix, ...(pace ? { rate: pace } : {}) }
        : member));
}

/** A pace only counts inside the request bounds, and 1 is the default going unsaid. */
export function normalizeRate(value) {
    const rate = typeof value === 'string' ? parseFloat(value) : value;
    return Number.isFinite(rate) && rate >= 0.25 && rate <= 4 && rate !== 1 ? rate : undefined;
}

function aliasValue(member) {
    return member.rate ? { voice: member.mix, rate: member.rate } : member.mix;
}

/** A member travels when renamed or paced. */
export function castAliases(cast) {
    return cast.reduce((aliases, member) => {
        if (member.name !== member.mix || member.rate) {
            aliases[member.name] = aliasValue(member);
        }
        return aliases;
    }, {});
}

/** The cast as the request field it becomes, self-named members included. */
export function exportCast(cast) {
    return {
        voice_aliases: cast.reduce((aliases, member) => {
            aliases[member.name] = aliasValue(member);
            return aliases;
        }, {})
    };
}

/** Anything carrying that field reads back, so a saved request body imports too. */
export function parseCastFile(data) {
    if (!data || typeof data !== 'object' || Array.isArray(data)) {
        return [];
    }

    const aliases = data.voice_aliases && typeof data.voice_aliases === 'object' ? data.voice_aliases : data;
    return Object.entries(aliases)
        .map(([name, value]) => {
            const plain = typeof value === 'string';
            const pace = plain ? undefined : value?.rate;
            const rate = normalizeRate(pace);
            if (pace != null && rate === undefined && parseFloat(pace) !== 1) {
                return null;
            }
            const mix = plain ? value : String(value?.voice ?? '');
            return { name: String(name).trim(), mix: mix.trim(), ...(rate ? { rate } : {}) };
        })
        .filter((member) => member && member.mix
            && (member.name === member.mix || CAST_NAME_PATTERN.test(member.name)));
}

/** parseVoiceMix drops empty +-parts, so speakability is judged before that smoothing hides them. */
export function isSpeakableMix(mix, available) {
    return String(mix || '').split('+').every((part) => part.trim())
        && parseVoiceMix(mix).every(({ voice }) => available.includes(voice));
}

/** Tag names in the text the request could not speak: not a cast member, not a mix of real voices. */
export function unspeakableTagNames(text, cast, available) {
    const names = new Map();
    for (const match of String(text).matchAll(tagPattern())) {
        const folded = match[1].toLowerCase();
        if (!names.has(folded)) {
            names.set(folded, match[1]);
        }
    }

    return [...names.entries()]
        .filter(([folded, name]) => !cast.some((member) => member.name.toLowerCase() === folded)
            && !isSpeakableMix(name, available))
        .map(([, name]) => name);
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
