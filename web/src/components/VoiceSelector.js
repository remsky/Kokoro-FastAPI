import { closeOnOutsidePress } from '../dismiss.js';
import { parseVoiceMix } from '../voiceTags.js';

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;')
        .replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export class VoiceSelector {
    constructor(voiceService) {
        this.voiceService = voiceService;
        this.handlers = {};
        this.elements = {
            voiceSearch: document.getElementById('voice-search'),
            voiceDropdown: document.getElementById('voice-dropdown'),
            voiceOptions: document.getElementById('voice-options'),
            selectedVoices: document.getElementById('selected-voices'),
            voiceCast: document.getElementById('voice-cast'),
            voiceCastList: document.getElementById('voice-cast-list'),
            castMenu: document.getElementById('cast-menu'),
            createTagBtn: document.getElementById('create-tag-btn')
        };
        this.menuFor = null;
        this.editing = null;
        this.settleRename = () => {};

        this.setupEventListeners();
        this.setupCastListeners();
    }

    /**
     * In tag mode the mixer keeps working exactly as it does otherwise, but as a
     * staging area: creating a tag moves the mix into the cast below and empties the
     * mixer for the next voice, so the same handful can be placed over and over.
     */
    setTagMode(enabled, handlers = {}) {
        this.handlers = handlers;
        if (this.elements.voiceCast) {
            this.elements.voiceCast.hidden = !enabled;
        }
        if (this.elements.createTagBtn) {
            this.elements.createTagBtn.hidden = !enabled;
        }
        if (!enabled) {
            this.settleRename();
            this.closeCastMenu();
        }
        this.updateCreateTagButton();
    }

    /** Replaces what is staged in the mixer, so a mix can be cleared or sent back to it. */
    setMix(mix) {
        this.voiceService.clearSelectedVoices();
        parseVoiceMix(mix).forEach(({ voice, weight }) => this.voiceService.addVoice(voice, weight));
        this.renderVoiceOptions(this.voiceService.filterVoices(this.elements.voiceSearch.value));
        this.updateSelectedVoicesDisplay();
    }

    renderCast(cast) {
        const list = this.elements.voiceCastList;
        if (!list) {
            return;
        }

        this.closeCastMenu();
        list.innerHTML = cast
            .map((member) => {
                const tagLabel = member.rate
                    ? `[voice:${member.name}] [rate:${member.rate}]`
                    : `[voice:${member.name}]`;
                return `
                <span class="cast-member"
                      data-name="${esc(member.name)}"
                      data-mix="${esc(member.mix)}"
                      title="${member.name === member.mix ? '' : esc(member.mix)}">
                    <button type="button" class="cast-insert-btn cast-name" data-name="${esc(member.name)}"
                            title="Insert ${esc(tagLabel)} at the cursor"
                            aria-label="Insert ${esc(tagLabel)} at the cursor">${esc(member.name)}</button>
                    <input type="number" class="cast-rate" data-name="${esc(member.name)}"
                           min="0.25" max="4" step="0.05" value="${member.rate ?? 1}"
                           title="Speed for this voice, 0.25 to 4, 1 for normal"
                           aria-label="Speed for ${esc(member.name)}">
                    <span class="cast-sep" aria-hidden="true">|</span>
                    <button type="button" class="cast-menu-btn" data-name="${esc(member.name)}"
                            aria-label="Options for ${esc(member.name)}" aria-haspopup="menu" aria-expanded="false">Options</button>
                </span>
            `;
            })
            .join('');
    }

    setupCastListeners() {
        const cast = this.elements.voiceCast;
        const create = this.elements.createTagBtn;

        if (create) {
            create.addEventListener('mousedown', (e) => e.preventDefault());
            create.addEventListener('click', () => this.handlers.onCommit?.());
        }

        if (!cast) {
            return;
        }

        // the caret is where the tag lands, so clicking in here must not steal focus
        cast.addEventListener('mousedown', (e) => {
            if (e.target.classList.contains('cast-rename-input') || e.target.classList.contains('cast-rate')) {
                return;
            }
            e.preventDefault();
        });

        this.elements.voiceCastList.addEventListener('change', (e) => {
            if (e.target.classList.contains('cast-rate')) {
                this.handlers.onRate?.(e.target.dataset.name, e.target.value);
            }
        });

        this.elements.voiceCastList.addEventListener('click', (e) => {
            const menuButton = e.target.closest('.cast-menu-btn');
            if (menuButton) {
                // a click with no pointer behind it came from Enter or Space, so the menu takes focus
                this.toggleCastMenu(menuButton.closest('.cast-member'), e.detail === 0);
                return;
            }

            // only the insert button places a tag, a stray click on the row does nothing
            const insertButton = e.target.closest('.cast-insert-btn');
            if (insertButton) {
                this.closeCastMenu();
                this.handlers.onInsert?.(insertButton.dataset.name);
            }
        });

        this.elements.castMenu.addEventListener('click', (e) => {
            const item = e.target.closest('.cast-menu-item');
            if (!item || item.getAttribute('aria-disabled') === 'true') {
                return;
            }

            const name = this.menuFor;
            this.closeCastMenu();
            // a rename open on another chip lands before this action re-renders the list under it
            this.settleRename();
            if (item.dataset.action === 'rename') {
                this.startRename(name);
                return;
            }
            if (item.dataset.action === 'insert') {
                this.handlers.onInsert?.(name);
                return;
            }
            this.handlers.onMenuAction?.(item.dataset.action, name);
        });

        // the menu is the keyboard route to a chip, so it has to be escapable and walkable
        cast.addEventListener('keydown', (e) => {
            if (!this.menuFor) {
                return;
            }
            if (e.key === 'Escape') {
                this.closeCastMenu({ restoreFocus: true });
                return;
            }
            if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') {
                return;
            }
            e.preventDefault();
            const items = [...this.elements.castMenu.querySelectorAll('.cast-menu-item')];
            const from = Math.max(items.indexOf(document.activeElement), 0);
            const step = e.key === 'ArrowDown' ? 1 : items.length - 1;
            items[(from + step) % items.length]?.focus();
        });

        this.elements.castMenu.addEventListener('focusout', (e) => {
            if (!this.elements.castMenu.contains(e.relatedTarget)) {
                this.closeCastMenu();
            }
        });

        closeOnOutsidePress(cast, () => this.closeCastMenu());
    }

    toggleCastMenu(chip, focusFirstItem = false) {
        const menu = this.elements.castMenu;
        if (!chip || this.menuFor === chip.dataset.name) {
            this.closeCastMenu();
            return;
        }

        const placed = this.handlers.isPlaced?.(chip.dataset.name) !== false;
        const noReset = this.resetBlockedReason(chip.dataset.name, chip.dataset.mix);
        this.menuFor = chip.dataset.name;
        chip.querySelector('.cast-menu-btn')?.setAttribute('aria-expanded', 'true');
        // a member standing for its own mix has no alias to undo, one whose mix is already a member cannot take that name back, and one still spoken cannot leave
        this.setMenuItem('reset', !noReset, noReset);
        this.setMenuItem('strip', placed, 'No tag in the text names this one');
        this.setMenuItem('remove', !placed, 'Tags in the text still name this one, so remove those first');
        menu.hidden = false;
        menu.style.left = `${chip.offsetLeft}px`;
        menu.style.top = `${chip.offsetTop + chip.offsetHeight + 4}px`;
        if (focusFirstItem) {
            menu.querySelector('.cast-menu-item')?.focus();
        }
    }

    /** Why undoing this alias is unavailable, empty when it is, read off the chips already rendered. */
    resetBlockedReason(name, mix) {
        if (name === mix) {
            return 'This name is its own mix, so there is no alias to undo';
        }
        return this.elements.voiceCastList?.querySelector(`.cast-member[data-name="${CSS.escape(mix)}"]`)
            ? `"${mix}" is already in the cast, so this name cannot go back to it`
            : '';
    }

    /** An option that cannot be taken yet greys out and says why, rather than leaving the menu. */
    setMenuItem(action, enabled, reason) {
        const item = this.elements.castMenu.querySelector(`[data-action="${action}"]`);
        item.setAttribute('aria-disabled', String(!enabled));
        item.title = enabled ? '' : reason;
    }

    closeCastMenu({ restoreFocus = false } = {}) {
        const button = this.menuFor && this.elements.voiceCastList
            ?.querySelector(`.cast-menu-btn[data-name="${CSS.escape(this.menuFor)}"]`);
        button?.setAttribute('aria-expanded', 'false');
        this.menuFor = null;
        if (this.elements.castMenu) {
            this.elements.castMenu.hidden = true;
        }
        if (restoreFocus) {
            button?.focus();
        }
    }

    /** Renames in the chip itself, so the name is edited where it is read. */
    startRename(name) {
        // an open rename lands first, otherwise its blur wipes the input this one is about to make
        this.settleRename();

        const chip = this.elements.voiceCastList.querySelector(`.cast-member[data-name="${CSS.escape(name)}"]`);
        const label = chip?.querySelector('.cast-name');
        if (!label) {
            return;
        }

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'cast-rename-input';
        input.value = name;
        input.maxLength = 24;
        input.size = Math.min(Math.max(name.length, 8), 24);
        label.replaceWith(input);
        input.focus();
        input.select();

        let settled = false;
        const commit = (next) => {
            if (settled) {
                return;
            }
            settled = true;
            this.settleRename = () => {};
            this.handlers.onRename?.(name, next);
        };
        this.settleRename = () => commit(input.value);

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                commit(input.value);
            } else if (e.key === 'Escape') {
                commit(name);
            }
        });
        input.addEventListener('blur', () => commit(input.value));
    }

    /** Editing sends a member back to the mixer, so the same button saves it rather than adding another. */
    setEditing(name) {
        this.editing = name;
        this.updateCreateTagButton();
    }

    updateCreateTagButton() {
        const button = this.elements.createTagBtn;
        if (!button) {
            return;
        }

        const mix = this.voiceService.getSelectedVoiceString();
        button.disabled = !mix;
        button.textContent = this.editing ? 'Save mix' : 'Create tag';
        if (this.editing) {
            button.title = mix ? `Retune ${this.editing}` : 'Mix one or more voices first';
            return;
        }
        button.title = mix
            ? `Add [voice:${mix}] to the cast`
            : 'Mix one or more voices first';
    }

    setupEventListeners() {
        // Voice search focus
        this.elements.voiceSearch.addEventListener('focus', () => {
            this.elements.voiceDropdown.classList.add('show');
        });

        // Voice search
        this.elements.voiceSearch.addEventListener('input', (e) => {
            const filteredVoices = this.voiceService.filterVoices(e.target.value);
            this.renderVoiceOptions(filteredVoices);
        });

        // Voice selection - handle clicks on the entire voice option
        this.elements.voiceOptions.addEventListener('mousedown', (e) => {
            e.preventDefault(); // Prevent blur on search input
            
            const voiceOption = e.target.closest('.voice-option');
            if (!voiceOption) return;
            
            const voice = voiceOption.dataset.voice;
            if (!voice) return;

            const isSelected = voiceOption.classList.contains('selected');
            
            if (!isSelected) {
                this.voiceService.addVoice(voice);
            } else {
                this.voiceService.removeVoice(voice);
            }
            
            voiceOption.classList.toggle('selected');
            this.updateSelectedVoicesDisplay();
            
            // Keep focus on search input
            requestAnimationFrame(() => {
                this.elements.voiceSearch.focus();
            });
        });

        // Weight adjustment
        this.elements.selectedVoices.addEventListener('input', (e) => {
            if (e.target.type === 'number') {
                const voice = e.target.dataset.voice;
                let weight = parseFloat(e.target.value);
                
                // Ensure weight is between 0.1 and 10
                weight = Math.max(0.1, Math.min(10, weight));
                e.target.value = weight;
                
                this.voiceService.updateWeight(voice, weight);
                this.updateCreateTagButton();
            }
        });

        // Remove selected voice
        this.elements.selectedVoices.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-voice')) {
                e.preventDefault();
                e.stopPropagation();
                const voice = e.target.dataset.voice;
                this.voiceService.removeVoice(voice);
                this.updateVoiceOptionState(voice, false);
                this.updateSelectedVoicesDisplay();
            }
        });

        closeOnOutsidePress(
            [this.elements.selectedVoices, this.elements.voiceSearch, this.elements.voiceDropdown],
            () => {
                this.elements.voiceDropdown.classList.remove('show');
                this.elements.voiceSearch.blur();
            }
        );
    }

    renderVoiceOptions(voices) {
        this.elements.voiceOptions.innerHTML = voices
            .map(voice => `
                <div class="voice-option ${this.voiceService.getSelectedVoices().includes(voice) ? 'selected' : ''}"
                     data-voice="${esc(voice)}">
                    ${esc(voice)}
                </div>
            `)
            .join('');
    }

    updateSelectedVoicesDisplay() {
        const selectedVoices = this.voiceService.getSelectedVoiceWeights();
        this.elements.selectedVoices.innerHTML = selectedVoices
            .map(({voice, weight}) => `
                <span class="selected-voice-tag">
                    <span class="voice-name">${esc(voice)}</span>
                    <span class="voice-weight">
                        <input type="number"
                               value="${weight}"
                               min="0.1"
                               max="10"
                               step="0.1"
                               data-voice="${esc(voice)}"
                               class="weight-input"
                               title="Voice weight (0.1 to 10)">
                    </span>
                    <span class="remove-voice" data-voice="${esc(voice)}" title="Remove voice">×</span>
                </span>
            `)
            .join('');

        this.updateCreateTagButton();
    }


    updateVoiceOptionState(voice, selected) {
        const voiceOption = this.elements.voiceOptions
            .querySelector(`[data-voice="${CSS.escape(voice)}"]`);
        if (voiceOption) {
            voiceOption.classList.toggle('selected', selected);
        }
    }

    async initialize() {
        try {
            await this.voiceService.loadVoices();
            this.renderVoiceOptions(this.voiceService.getAvailableVoices());
            this.updateSelectedVoicesDisplay();
            return true;
        } catch (error) {
            console.error('Failed to initialize voice selector:', error);
            return false;
        }
    }
}

export default VoiceSelector;