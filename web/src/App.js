import AudioService from './services/AudioService.js';
import VoiceService from './services/VoiceService.js';
import PlayerState from './state/PlayerState.js';
import PlayerControls from './components/PlayerControls.js';
import VoiceSelector from './components/VoiceSelector.js';
import WaveVisualizer from './components/WaveVisualizer.js';
import TextEditor from './components/TextEditor.js';
import config from './config.js';
import { closeOnOutsidePress } from './dismiss.js';
import {
    CAST_NAME_PATTERN,
    addToCast,
    castAliases,
    countVoiceTags,
    exportCast,
    hasVoiceTagFor,
    insertVoiceTag,
    isSpeakableMix,
    leadingVoiceTag,
    parseCastFile,
    removeFromCast,
    removeVoiceTagsFor,
    renameCastMember,
    renameVoiceTags,
    seedVoiceTag,
    stripVoiceTags,
    updateCastMix
} from './voiceTags.js';

const NARROW_LAYOUT = '(max-width: 900px)';

export class App {
    constructor() {
        this.cast = [];
        this.editing = null;
        this.tagMode = false;
        this.stagedBeforeTags = '';
        this.elements = {
            generateBtn: document.getElementById('generate-btn'),
            generateBtnText: document.querySelector('#generate-btn .btn-text'),
            generateBtnLoader: document.querySelector('#generate-btn .loader'),
            downloadBtn: document.getElementById('download-btn'),
            autoplayToggle: document.getElementById('autoplay-toggle'),
            formatSelect: document.getElementById('format-select'),
            status: document.getElementById('status'),
            cancelBtn: document.getElementById('cancel-btn'),
            streamingNotice: document.getElementById('streaming-notice'),
            charCount: document.getElementById('char-count'),
            cup: document.querySelector('.logo-container .cup'),
            voiceTabs: document.querySelector('.card-tabs'),
            voicesTab: document.getElementById('voices-tab'),
            voiceTagsTab: document.getElementById('voice-tags-tab'),
            voiceTagNotice: document.getElementById('voice-tag-notice'),
            voiceTagNoticeText: document.getElementById('voice-tag-notice-text'),
            removeVoiceTagsBtn: document.getElementById('remove-voice-tags-btn'),
            castFileMenu: document.getElementById('cast-file-menu'),
            saveCastBtn: document.getElementById('save-cast-btn'),
            importCastBtn: document.getElementById('import-cast-btn'),
            importCastReplaceBtn: document.getElementById('import-cast-replace-btn'),
            importCastInput: document.getElementById('import-cast-input')
        };

        this.initialize();
    }

    async initialize() {
        // Initialize services and state
        this.playerState = new PlayerState();
        this.audioService = new AudioService();
        this.voiceService = new VoiceService();

        this.renderVersionBadge();
        this.renderStarBadge();

        // Initialize components
        this.playerControls = new PlayerControls(this.audioService, this.playerState);
        this.voiceSelector = new VoiceSelector(this.voiceService);
        this.waveVisualizer = new WaveVisualizer(this.playerState);
        
        // counter lives outside the component, in the editor status row
        const editorContainer = document.getElementById('text-editor');
        this.textEditor = new TextEditor(editorContainer, {
            linesPerPage: 20,
            onTextChange: (text) => {
                this.elements.charCount.textContent = `${text.length} characters`;
                this.updateVoiceTagNotice();
            }
        });

        this.setupNarrowLayout();

        // Initialize voice selector
        const voicesLoaded = await this.voiceSelector.initialize();
        if (!voicesLoaded) {
            this.showStatus('Failed to load voices', 'error');
            this.elements.generateBtn.disabled = true;
            return;
        }

        this.setupEventListeners();
        this.setupAudioEvents();
        this.setupVoiceTags();
        this.applyBrowserStreamingNotice();
    }

    setupVoiceTags() {
        const tabs = [this.elements.voicesTab, this.elements.voiceTagsTab];
        if (tabs.some((tab) => !tab)) {
            return;
        }

        tabs.forEach((tab, index) => {
            tab.addEventListener('click', () => this.setVoiceTagMode(tab === this.elements.voiceTagsTab));
            tab.addEventListener('keydown', (e) => {
                if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') {
                    return;
                }
                e.preventDefault();
                const step = e.key === 'ArrowRight' ? 1 : tabs.length - 1;
                const next = tabs[(index + step) % tabs.length];
                this.setVoiceTagMode(next === this.elements.voiceTagsTab);
                next.focus();
            });
        });

        this.elements.removeVoiceTagsBtn.addEventListener('click', () => {
            this.textEditor.replaceText(stripVoiceTags(this.textEditor.getText()));
        });

        if (this.elements.castFileMenu) {
            this.setupCastFileMenu();
        }

        this.setVoiceTagMode(this.tagMode);
    }

    setVoiceTagMode(enabled) {
        this.tagMode = enabled;
        this.renderVoiceTabs();
        if (!enabled) {
            // an edit only exists while the pane is open, so it cannot survive into the next visit
            this.setEditing(null);
        }
        this.voiceSelector.setTagMode(enabled, {
            onCommit: () => this.commitMix(),
            onInsert: (name) => this.insertVoiceTag(name),
            onRename: (name, next) => this.renameCastMember(name, next),
            onMenuAction: (action, name) => this.castMenuAction(action, name),
            isPlaced: (name) => hasVoiceTagFor(this.textEditor.getText(), name)
        });

        if (enabled) {
            // the staged mix joins the cast but is remembered, so leaving the tab hands the same voice back
            this.stagedBeforeTags = this.voiceService.getSelectedVoiceString() || this.cast[0]?.mix || '';
            this.commitMix();
            // the seeded tag is the whole explanation of the syntax
            const seeded = seedVoiceTag(this.textEditor.getText(), this.cast[0]?.name);
            if (seeded.changed) {
                this.textEditor.replaceText(seeded.text);
            }
        } else if (!this.voiceService.hasSelectedVoices() && this.cast.length) {
            this.voiceSelector.setMix(this.stagedBeforeTags || this.cast[0].mix);
        }

        this.updateVoiceTagNotice();
    }

    renderVoiceTabs() {
        const active = this.tagMode ? this.elements.voiceTagsTab : this.elements.voicesTab;
        this.elements.voiceTabs?.classList.toggle('is-tags', this.tagMode);
        for (const tab of [this.elements.voicesTab, this.elements.voiceTagsTab]) {
            const on = tab === active;
            tab.classList.toggle('is-active', on);
            tab.setAttribute('aria-selected', String(on));
            tab.tabIndex = on ? 0 : -1;
        }
    }

    /**
     * Moves the staged mix into the cast and empties the mixer, so building the next
     * voice starts from nothing. Placing it in the text stays a separate click.
     */
    commitMix() {
        const mix = this.voiceService.getSelectedVoiceString();
        if (!mix) {
            return;
        }

        if (this.editing) {
            this.saveEditedMix(this.editing, mix);
        } else {
            this.setCast(addToCast(this.cast, mix));
        }

        this.voiceSelector.setMix('');
    }

    /** A renamed member keeps its name, so the tags already in the text still point at it. */
    saveEditedMix(name, mix) {
        const member = this.cast.find((entry) => entry.name === name);
        let cast = updateCastMix(this.cast, name, mix);

        // a member still standing for its own mix has to follow it, tags and all
        if (member && member.name === member.mix && member.mix !== mix) {
            // that mix may already sit in the cast, and a twin chip would confuse every name-keyed lookup
            cast = this.cast.some((entry) => entry.name === mix)
                ? removeFromCast(cast, name)
                : renameCastMember(cast, name, mix);
            this.textEditor.replaceText(renameVoiceTags(this.textEditor.getText(), name, mix));
        }

        this.setCast(cast);
        this.setEditing(null);
    }

    castMenuAction(action, name) {
        const member = this.cast.find((entry) => entry.name === name);
        if (!member) {
            return;
        }

        if (action === 'edit') {
            this.setEditing(name);
            this.voiceSelector.setMix(member.mix);
        } else if (action === 'strip') {
            this.textEditor.replaceText(removeVoiceTagsFor(this.textEditor.getText(), name));
            this.updateVoiceTagNotice();
        } else if (action === 'reset' && member.name !== member.mix) {
            // the tags stay where they are, so the name they answer to becomes the mix again
            if (this.editing === name) {
                this.setEditing(member.mix);
            }
            this.textEditor.replaceText(renameVoiceTags(this.textEditor.getText(), name, member.mix));
            this.setCast(renameCastMember(this.cast, name, member.mix));
        } else if (action === 'remove' && !hasVoiceTagFor(this.textEditor.getText(), name)) {
            if (this.editing === name) {
                this.setEditing(null);
                this.voiceSelector.setMix('');
            }
            this.setCast(removeFromCast(this.cast, name));
        }
    }

    setEditing(name) {
        this.editing = name;
        this.voiceSelector.setEditing(name);
    }

    /**
     * A short name is only a label for the mix, so the rules are the tag syntax plus
     * anything that would shadow a real voice or another member.
     */
    renameCastMember(name, requested) {
        const next = String(requested || '').trim();
        // re-rendering is what puts the chip back, so it also ends the edit that was refused
        const keepName = (message) => {
            if (message) {
                this.showStatus(message, 'error');
            }
            this.setCast(this.cast);
        };

        if (next === name) {
            keepName();
            return;
        }

        if (!CAST_NAME_PATTERN.test(next)) {
            keepName('A cast name is 1 to 24 letters, numbers, dashes or underscores');
            return;
        }

        const taken = [
            ...this.cast.filter((entry) => entry.name !== name).map((entry) => entry.name),
            ...this.voiceService.getAvailableVoices()
        ];

        // folded like the server resolves aliases, so AF_Bella cannot shadow af_bella
        const folded = next.toLowerCase();
        if (taken.some((entry) => entry.toLowerCase() === folded)) {
            keepName(`"${next}" is already taken`);
            return;
        }

        if (this.editing === name) {
            this.setEditing(next);
        }
        this.textEditor.replaceText(renameVoiceTags(this.textEditor.getText(), name, next));
        this.setCast(renameCastMember(this.cast, name, next));
    }

    setCast(cast) {
        this.cast = cast;
        this.voiceSelector.renderCast(cast);
    }

    setupCastFileMenu() {
        const menu = this.elements.castFileMenu;
        const close = () => {
            menu.open = false;
        };

        this.elements.saveCastBtn.addEventListener('click', () => {
            close();
            this.saveCast();
        });
        this.elements.importCastBtn.addEventListener('click', () => {
            close();
            this.importReplace = false;
            this.elements.importCastInput.click();
        });
        this.elements.importCastReplaceBtn.addEventListener('click', () => {
            close();
            this.importReplace = true;
            this.elements.importCastInput.click();
        });
        this.elements.importCastInput.addEventListener('change', (e) => {
            const [file] = e.target.files;
            // the same file picked twice has to fire again, so the input is emptied either way
            e.target.value = '';
            if (file) {
                this.importCast(file, this.importReplace);
            }
            this.importReplace = false;
        });

        menu.addEventListener('toggle', () => {
            if (menu.open) {
                this.voiceSelector.closeCastMenu();
            }
        });
        closeOnOutsidePress(menu, close);
        menu.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                close();
                menu.querySelector('summary').focus();
            }
        });
    }

    saveCast() {
        if (!this.cast.length) {
            this.showStatus('There is no cast to save yet', 'error');
            return;
        }

        const blob = new Blob([`${JSON.stringify(exportCast(this.cast), null, 2)}\n`], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        Object.assign(document.createElement('a'), { href: url, download: 'voice-tags.json' }).click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    /** Imports drop mixes this server cannot speak rather than leave them to 400. */
    async importCast(file, replace = false) {
        let members = [];
        try {
            members = parseCastFile(JSON.parse(await file.text()));
        } catch {
            this.showStatus('That file is not readable JSON', 'error');
            return;
        }

        const available = this.voiceService.getAvailableVoices();
        const base = replace ? [] : this.cast;
        let cast = base;
        for (const { name, mix } of members) {
            // folded like the server resolves aliases, so a case-variant name cannot shadow a voice or member
            const folded = name.toLowerCase();
            const known = isSpeakableMix(mix, available);
            const taken = cast.some((entry) => entry.name.toLowerCase() === folded || entry.mix === mix)
                || (name !== mix && available.some((voice) => voice.toLowerCase() === folded));
            if (known && !taken) {
                cast = [...cast, { name, mix }];
            }
        }

        const added = cast.length - base.length;
        if (!added) {
            this.showStatus('Nothing in that file could join the cast', 'error');
            return;
        }

        if (this.editing && !cast.some((entry) => entry.name === this.editing)) {
            this.setEditing(null);
            this.voiceSelector.setMix('');
        }

        const skipped = members.length - added;
        this.setCast(cast);
        this.showStatus(replace
            ? `Cast replaced with ${added}${skipped ? `, skipped ${skipped}` : ''}`
            : `Added ${added} to the cast${skipped ? `, skipped ${skipped}` : ''}`, 'success');
    }

    insertVoiceTag(voice) {
        const { text, cursor } = insertVoiceTag(this.textEditor.getPageText(), this.textEditor.getCursor(), voice);
        this.textEditor.setPageText(text, cursor);
    }

    /**
     * Tags left in the text outside tag mode are sent as prose and read aloud, so the
     * count is offered with a way out rather than a warning to act on.
     */
    updateVoiceTagNotice() {
        const notice = this.elements.voiceTagNotice;
        if (!notice) {
            return;
        }

        const count = countVoiceTags(this.textEditor.getText());
        notice.hidden = count === 0 || this.tagMode;
        this.elements.voiceTagNoticeText.textContent =
            `${count} voice ${count === 1 ? 'tag' : 'tags'} will be read aloud.`;
    }

    async renderStarBadge() {
        const count = document.getElementById('gh-star-count');
        if (!count) return;
        try {
            const response = await fetch('https://api.github.com/repos/remsky/Kokoro-FastAPI');
            if (!response.ok) return;
            const stars = (await response.json()).stargazers_count;
            if (typeof stars !== 'number') return;
            count.textContent = stars >= 1000 ? `${(stars / 1000).toFixed(1).replace(/\.0$/, '')}k` : `${stars}`;
            count.hidden = false;
        } catch (_) {
            // leave hidden on failure
        }
    }

    relocateOnNarrow(node, narrowHome, wideHome) {
        const query = window.matchMedia(NARROW_LAYOUT);
        const place = () => {
            const [parent, anchor] = query.matches ? narrowHome : wideHome;
            if (node.parentElement === parent) return;
            const focused = node.contains(document.activeElement) ? document.activeElement : null;
            parent.insertBefore(node, anchor ?? null);
            focused?.focus();
        };

        place();
        query.addEventListener('change', place);
    }

    setupNarrowLayout() {
        const card = document.querySelector('.generate-card');
        const sidePane = document.querySelector('.side-pane');
        const dock = document.querySelector('.player-dock');
        const controls = dock?.querySelector('.player-controls');
        if (card && sidePane && controls) {
            this.relocateOnNarrow(card, [dock, controls], [sidePane, null]);
        }
    }

    async renderVersionBadge() {
        const badge = document.getElementById('version-badge');
        if (!badge) return;
        try {
            await config.ensureInitialized();
            if (config.version) {
                badge.textContent = `v${config.version}`;
                badge.hidden = false;
            }
        } catch (_) {
            // leave hidden on failure
        }
    }

    applyBrowserStreamingNotice() {
        const notice = this.elements.streamingNotice;
        if (!notice) {
            return;
        }
        const format = this.elements.formatSelect?.value || 'mp3';
        const formatLabel = format.toUpperCase();
        const isFirefox = /Firefox\//.test(navigator.userAgent);
        let message = '';

        if (format === 'pcm') {
            message = 'PCM may not play in-browser; download still works.';
        } else if (format !== 'mp3') {
            message = `${formatLabel} plays/downloads once generation finishes.`;
        } else if (!this.audioService.supportsMSEMp3()) {
            message = isFirefox
                ? 'No streaming in Firefox; playback/download ready once generation finishes.'
                : 'Streaming may be unsupported here; playback/download ready once generation finishes.';
        } else if (this.elements.autoplayToggle?.checked) {
            message = 'Auto-play on; pause when done for full seek.';
        }

        notice.textContent = message;
        notice.hidden = !message;
    }

    setupEventListeners() {
        // Generate button
        this.elements.generateBtn.addEventListener('click', () => this.generateSpeech());

        // Download button (div with role=button, so handle keyboard activation too)
        this.elements.downloadBtn.addEventListener('click', () => this.downloadAudio());
        this.elements.downloadBtn.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.downloadAudio();
            }
        });

        // Keep browser/output warning aligned with the selected format and autoplay state
        this.elements.formatSelect.addEventListener('change', () => this.applyBrowserStreamingNotice());
        this.elements.autoplayToggle.addEventListener('change', () => this.applyBrowserStreamingNotice());

        // Cancel button
        this.elements.cancelBtn.addEventListener('click', () => {
            this.audioService.cancel();
            this.setGenerating(false);
            this.elements.downloadBtn.classList.remove('ready');
            this.showStatus('Generation cancelled', 'info');
        });

        // Handle page unload
        window.addEventListener('beforeunload', () => {
            this.audioService.cleanup();
            this.playerControls.cleanup();
            this.waveVisualizer.cleanup();
        });
    }

    setupAudioEvents() {
        // Handle download button visibility
        this.audioService.addEventListener('downloadReady', () => {
            this.elements.downloadBtn.classList.add('ready');
        });

        // Handle buffer errors
        this.audioService.addEventListener('bufferError', () => {
            this.showStatus('Processing... (Download will be available when complete)', 'info');
        });

        // Handle completion
        this.audioService.addEventListener('complete', () => {
            this.setGenerating(false);
            
            // Show preparing status
            this.showStatus('Preparing file...', 'info');

            // Flash the coffee cup
            this.elements.cup.classList.add('done');
        });

        // Handle download ready
        this.audioService.addEventListener('downloadReady', () => {
            setTimeout(() => {
                if (!this._playbackFailed) {
                    this.showStatus('Generation complete', 'success');
                }
            }, 500); // Small delay to ensure "Preparing file..." is visible
        });

        // Handle audio end
        this.audioService.addEventListener('ended', () => {
            this.playerState.setPlaying(false);
        });

        // Handle errors
        this.audioService.addEventListener('error', (error) => {
            this.showStatus('Error: ' + error.message, 'error');
            this.setGenerating(false);
            this.elements.downloadBtn.style.display = 'none';
        });

        // Block-mode playback failure: file is still available for download
        this.audioService.addEventListener('playbackUnavailable', () => {
            this._playbackFailed = true;
            this.showStatus(
                'Playback unavailable in this browser. Use the download below.',
                'info'
            );
        });
    }

    /**
     * The voice parameter speaks anything ahead of the first tag, so in tag mode the
     * text has to open with one and that tag is the voice. Neither a staged mix nor a
     * cast member stands in for it, so what is spoken is only ever what the text says.
     */
    requestVoice() {
        if (this.tagMode) {
            return leadingVoiceTag(this.textEditor.getText());
        }
        return this.voiceService.getSelectedVoiceString();
    }

    showStatus(message, type = 'info') {
        this.elements.status.textContent = message;
        this.elements.status.className = 'status ' + type;
        // an uncleared timer from an earlier status would blank this one early
        clearTimeout(this._statusTimer);
        this._statusTimer = setTimeout(() => {
            this.elements.status.className = 'status';
        }, 5000);
    }

    setGenerating(isGenerating) {
        this.playerState.setGenerating(isGenerating);
        this.elements.generateBtn.disabled = isGenerating;
        this.elements.generateBtn.classList.toggle('loading', isGenerating);
        this.elements.generateBtnLoader.style.display = isGenerating ? 'block' : 'none';
        this.elements.generateBtnText.style.visibility = isGenerating ? 'hidden' : 'visible';
        this.elements.cancelBtn.style.display = isGenerating ? 'block' : 'none';
        this.elements.cup.classList.toggle('brewing', isGenerating);
        if (isGenerating) {
            this.elements.cup.classList.remove('done');
        }
    }

    validateInput() {
        const text = this.textEditor.getText().trim();
        // a seeded tag on its own is not something to speak
        const spoken = this.tagMode ? stripVoiceTags(text).trim() : text;
        if (!spoken) {
            this.showStatus('Please enter some text', 'error');
            return false;
        }

        if (!this.requestVoice()) {
            this.showStatus(
                this.tagMode ? 'Start the text with a voice tag' : 'Please select a voice',
                'error'
            );
            return false;
        }
        
        return true;
    }

    async generateSpeech() {
        // Don't check isGenerating state since we want to allow generation after cancel
        if (!this.validateInput()) {
            return;
        }

        const text = this.textEditor.getText().trim();
        const voice = this.requestVoice();
        const speed = this.playerState.getState().speed;
        const allowVoiceTags = this.tagMode;

        this.playerState.setReady(false);
        this.playerState.setPlaying(false);
        this.playerState.setTime(0, 0);
        this.setGenerating(true);
        this._playbackFailed = false;
        this.elements.downloadBtn.classList.remove('ready');

        // Just reset progress bar, don't do full cleanup
        this.waveVisualizer.updateProgress(0, 1);
        
        try {
            console.log('Starting audio generation...', { chars: text.length, voice, speed });

            if (!text || !voice) {
                console.error('Invalid input:', { text, voice, speed });
                throw new Error('Invalid input parameters');
            }
            
            await this.audioService.streamAudio(
                text,
                voice,
                speed,
                (loaded, total) => this.waveVisualizer.updateProgress(loaded, total),
                { allowVoiceTags, voiceAliases: allowVoiceTags ? castAliases(this.cast) : null }
            );
        } catch (error) {
            console.error('Generation error:', error);
            if (error.name !== 'AbortError') {
                this.showStatus('Error generating speech: ' + error.message, 'error');
                this.setGenerating(false);
            }
        }
    }

    downloadAudio() {
        const downloadUrl = this.audioService.getDownloadUrl();
        if (!downloadUrl) {
            console.warn('No download URL available');
            return;
        }

        console.log('Starting download from:', downloadUrl);

        // fallback only: the server's Content-Disposition wins when it's present
        const name = this.audioService.getDownloadName();

        const a = document.createElement('a');
        a.href = downloadUrl;
        if (name) {
            a.download = name;
        }
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new App();
});
