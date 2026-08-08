import { segmentSentences, sentenceIndexAt, sentenceStartFraction } from '../readAlong.js';
import { alignChunks, sentenceIndexAtTime } from '../readAlongTiming.js';

const SYNC_MS = 200;
// landing a hair early beats arriving mid-word
const SEEK_LEAD_SECONDS = 0.15;

export class ReadAlong {
    constructor(audioService, textEditor) {
        this.audioService = audioService;
        this.textEditor = textEditor;
        this.elements = {
            toggle: document.getElementById('read-along-btn'),
            editor: document.getElementById('text-editor')
        };
        this.view = null;
        this.spans = [];
        this.sentences = [];
        this.sourceText = '';
        this.sentenceTimes = null;
        this.timingFetched = false;
        this.active = false;
        this.activeIndex = -1;
        this.syncTimer = null;

        this.elements.toggle?.addEventListener('click', () => this.setActive(!this.active));
    }

    // the snapshot sent to the server, not the live editor text, so later edits cannot desync it
    setSource(text) {
        this.sourceText = text || '';
        this.sentences = segmentSentences(this.sourceText);
        this.sentenceTimes = null;
        this.timingFetched = false;
    }

    async loadTiming() {
        if (this.timingFetched) {
            return;
        }
        this.timingFetched = true;
        const source = this.sourceText;
        const url = await this.audioService.getTimingUrl?.();
        if (!url) {
            return;
        }
        try {
            const response = await fetch(url);
            if (!response.ok) {
                this.timingFetched = false;
                return;
            }
            if (this.sourceText !== source) {
                return;
            }
            const data = await response.json();
            if (this.sourceText !== source) {
                return;
            }
            this.sentenceTimes = alignChunks(this.sourceText, this.sentences, data.chunks);
        } catch {
            this.timingFetched = false;
            this.sentenceTimes = null;
        }
    }

    setAvailable(available) {
        const usable = available && this.sentences.length > 0;
        if (!usable) {
            this.setActive(false);
        }
        if (this.elements.toggle) {
            this.elements.toggle.disabled = !usable;
            this.elements.toggle.title = usable
                ? 'Follow the text as it plays'
                : 'Available when generation completes';
        }
    }

    setActive(active) {
        if (active === this.active || (active && !this.elements.editor)) {
            return;
        }
        this.active = active;
        this.elements.toggle?.classList.toggle('is-active', active);
        this.elements.toggle?.setAttribute('aria-pressed', String(active));

        if (active) {
            this.renderView();
            // css-hidden, not torn down, so the editor keeps its pages, caret and listeners
            this.elements.editor.classList.add('read-along-active');
            // one swap to the finished file: real duration for the fraction math, plain seeks after
            if (this.audioService.canSwapToFileSource()) {
                this.audioService.swapToFileSource(null, this.audioService.isPlaying());
            }
            this.loadTiming();
            this.startSync();
        } else {
            this.stopSync();
            const sentence = this.sentences[this.activeIndex];
            this.view?.remove();
            this.view = null;
            this.spans = [];
            this.activeIndex = -1;
            this.elements.editor.classList.remove('read-along-active');
            if (sentence) {
                this.textEditor?.revealOffset(sentence.start);
            }
        }
    }

    // sentences group into paragraph blocks so offscreen ones can skip rendering entirely
    renderView() {
        this.view = document.createElement('div');
        this.view.className = 'read-along-view';
        this.spans = [];

        let paragraph = null;
        this.sentences.forEach((sentence, index) => {
            if (!paragraph) {
                paragraph = document.createElement('div');
                paragraph.className = 'read-paragraph';
                this.view.appendChild(paragraph);
            }

            const slice = this.sourceText.slice(sentence.start, sentence.end);
            const breaks = (slice.match(/\s*$/)[0].match(/\n/g) || []).length;
            const span = document.createElement('span');
            span.className = sentence.spoken > 0 ? 'read-sentence' : 'read-sentence silent';
            // the block boundary renders the line break, so the newlines themselves would double it
            span.textContent = breaks > 0 ? slice.replace(/\s+$/, '') : slice;
            if (sentence.spoken > 0) {
                span.addEventListener('click', () => this.seekToSentence(index));
            }
            paragraph.appendChild(span);
            this.spans.push(span);

            if (breaks > 0) {
                // a run of blank lines is paragraph spacing, a single newline is just a break
                paragraph.classList.toggle('gap', breaks > 1);
                paragraph = null;
            }
        });

        this.elements.editor.querySelector('.page-navigation').insertAdjacentElement('afterend', this.view);
    }

    startSync() {
        this.stopSync();
        this.syncTimer = setInterval(() => this.sync(), SYNC_MS);
        this.sync();
    }

    stopSync() {
        if (this.syncTimer) {
            clearInterval(this.syncTimer);
            this.syncTimer = null;
        }
    }

    sync() {
        const duration = this.audioService.getDuration();
        if (!Number.isFinite(duration) || duration <= 0) {
            return;
        }
        const time = this.audioService.getCurrentTime();
        const index = this.sentenceTimes
            ? sentenceIndexAtTime(this.sentenceTimes, time)
            : sentenceIndexAt(this.sentences, time / duration);
        this.setActiveIndex(index);
    }

    setActiveIndex(index) {
        if (index === this.activeIndex) {
            return;
        }
        this.spans[this.activeIndex]?.classList.remove('is-active');
        this.activeIndex = index;
        const span = this.spans[index];
        if (span) {
            span.classList.add('is-active');
            span.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    }

    seekToSentence(index) {
        const duration = this.audioService.getDuration();
        if (!Number.isFinite(duration) || duration <= 0) {
            return;
        }
        const start = this.sentenceTimes?.[index]
            ?? sentenceStartFraction(this.sentences, index) * duration;
        this.audioService.seek(Math.max(0, start - SEEK_LEAD_SECONDS));
        this.setActiveIndex(index);
    }

    cleanup() {
        this.setActive(false);
    }
}

export default ReadAlong;
