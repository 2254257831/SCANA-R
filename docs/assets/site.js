'use strict';
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
