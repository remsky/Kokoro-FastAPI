/** Closes a transient overlay when the next press lands outside every region it owns. */
export function closeOnOutsidePress(inside, close) {
    const zones = [].concat(inside).filter(Boolean);
    document.addEventListener('mousedown', (e) => {
        if (!zones.some((zone) => zone.contains(e.target))) {
            close();
        }
    });
}
