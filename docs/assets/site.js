'use strict';
// Native controls remain available if autoplay is unavailable or JavaScript is off.
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const teaser = document.querySelector('[data-autoplay]');
if (teaser && !reducedMotion.matches) {
  teaser.muted = true;
  teaser.play().catch(() => { /* Poster and native play control remain visible. */ });
}
reducedMotion.addEventListener('change', event => {
  if (event.matches && teaser) teaser.pause();
});
// Starting a comparison pauses any other video. Clips inside each file stay synced.
document.querySelectorAll('video').forEach(video => {
  video.addEventListener('play', () => {
    document.querySelectorAll('video').forEach(other => {
      if (other !== video) other.pause();
    });
  });
});
document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const text = document.getElementById(button.dataset.copy).innerText;
    const status = button.closest('.code-block').querySelector('.copy-status');
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const area = document.createElement('textarea');
        area.value = text; area.style.position = 'fixed'; area.style.opacity = '0';
        document.body.appendChild(area); area.select();
        const success = document.execCommand('copy'); area.remove();
        if (!success) throw new Error('Clipboard unavailable');
      }
      status.textContent = 'Commands copied.';
    } catch (_) {
      status.textContent = 'Select the commands and copy them manually.';
    }
  });
});
