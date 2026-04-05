/**
 * dev-editor.js — Local-only contenteditable editing system.
 * Only activates on localhost. Adds a floating Edit/Save toggle
 * that makes lecture content editable and POSTs changes to a
 * companion save server on port 8081.
 */
(function () {
  'use strict';

  // Safety: only run on localhost
  const host = window.location.hostname;
  if (host !== 'localhost' && host !== '127.0.0.1' && host !== '0.0.0.0') return;

  const SAVE_URL = 'http://localhost:8081/save';
  const EDITABLE_SELECTORS = 'p, li, h2, h3, h4, .callout, .callout-label';

  // --- Create toggle button ---
  const btn = document.createElement('button');
  btn.textContent = 'Edit';
  btn.id = 'dev-editor-toggle';
  Object.assign(btn.style, {
    position: 'fixed',
    bottom: '24px',
    right: '24px',
    zIndex: '9999',
    padding: '10px 22px',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    fontFamily: 'var(--sans, "Source Sans 3", sans-serif)',
    fontSize: '14px',
    fontWeight: '600',
    letterSpacing: '0.02em',
    background: 'var(--teal, #2dd4bf)',
    color: 'var(--bg-deep, #0a0e17)',
    boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
    transition: 'background 0.2s, transform 0.1s',
  });
  document.body.appendChild(btn);

  // --- Inject editable outline style ---
  const style = document.createElement('style');
  style.textContent = `
    [contenteditable="true"] {
      outline: 2px solid rgba(96, 165, 250, 0.45) !important;
      outline-offset: 2px;
      border-radius: 3px;
    }
    #dev-editor-toggle:hover {
      transform: scale(1.05);
    }
    .dev-delete-btn {
      position: absolute;
      top: -8px;
      right: -8px;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      border: none;
      background: #ef4444;
      color: #fff;
      font-size: 12px;
      line-height: 20px;
      text-align: center;
      cursor: pointer;
      z-index: 9998;
      opacity: 0;
      transition: opacity 0.15s;
      pointer-events: none;
      padding: 0;
    }
    [contenteditable="true"]:hover {
      position: relative;
    }
    [contenteditable="true"]:hover > .dev-delete-btn {
      opacity: 1;
      pointer-events: auto;
    }
    #dev-editor-saved {
      position: fixed;
      bottom: 72px;
      right: 24px;
      z-index: 9999;
      padding: 8px 18px;
      border-radius: 6px;
      background: var(--green, #4ade80);
      color: var(--bg-deep, #0a0e17);
      font-family: var(--sans, sans-serif);
      font-size: 13px;
      font-weight: 600;
      opacity: 0;
      transition: opacity 0.3s;
      pointer-events: none;
    }
  `;
  document.head.appendChild(style);

  // --- Saved toast ---
  const toast = document.createElement('div');
  toast.id = 'dev-editor-saved';
  toast.textContent = 'Saved!';
  document.body.appendChild(toast);

  function showToast(msg, isError) {
    toast.textContent = msg || 'Saved!';
    toast.style.background = isError ? 'var(--rose, #f472b6)' : 'var(--green, #4ade80)';
    toast.style.opacity = '1';
    setTimeout(function () { toast.style.opacity = '0'; }, 1800);
  }

  let editing = false;

  function getArticle() {
    return document.querySelector('.lecture-article') || document.querySelector('article');
  }

  function setEditable(on) {
    const article = getArticle();
    if (!article) return;
    const els = article.querySelectorAll(EDITABLE_SELECTORS);
    els.forEach(function (el) {
      if (on) {
        el.setAttribute('contenteditable', 'true');
        // Add delete button
        if (!el.querySelector('.dev-delete-btn')) {
          el.style.position = 'relative';
          const del = document.createElement('button');
          del.className = 'dev-delete-btn';
          del.textContent = '×';
          del.title = 'Delete this block';
          del.addEventListener('click', function (e) {
            e.stopPropagation();
            e.preventDefault();
            if (confirm('Delete this block?')) {
              el.remove();
            }
          });
          el.appendChild(del);
        }
      } else {
        el.removeAttribute('contenteditable');
        // Remove delete buttons
        const del = el.querySelector('.dev-delete-btn');
        if (del) del.remove();
      }
    });
  }

  function enterEditMode() {
    editing = true;
    setEditable(true);
    btn.textContent = 'Save';
    btn.style.background = 'var(--accent, #f97316)';
  }

  function save() {
    const article = getArticle();
    if (!article) return;

    const html = article.innerHTML;
    const pagePath = window.location.pathname;

    btn.textContent = 'Saving...';
    btn.disabled = true;

    fetch(SAVE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: pagePath, html: html }),
    })
      .then(function (res) {
        if (!res.ok) throw new Error('Server returned ' + res.status);
        setEditable(false);
        editing = false;
        btn.textContent = 'Edit';
        btn.style.background = 'var(--teal, #2dd4bf)';
        btn.disabled = false;
        showToast('Saved!');
      })
      .catch(function (err) {
        btn.textContent = 'Save';
        btn.disabled = false;
        showToast('Error: ' + err.message, true);
      });
  }

  btn.addEventListener('click', function () {
    if (editing) {
      save();
    } else {
      enterEditMode();
    }
  });
})();
