/* ═══════════════════════════════════════════════════════════════
   Diffusion From Scratch — Shared JavaScript
   Sidebar nav, progress bar, ToC, mobile menu, KaTeX rendering

   NOTE: All innerHTML usage in this file operates on hardcoded,
   trusted content defined in index.html's modules array.
   No user input is ever interpolated. This is safe for a static site.
   ═══════════════════════════════════════════════════════════════ */

// ── Scroll progress bar ──
function updateProgress() {
  const scrolled = window.scrollY;
  const total = document.documentElement.scrollHeight - window.innerHeight;
  const pct = total > 0 ? (scrolled / total) * 100 : 0;
  const bar = document.querySelector('.progress-bar .bar');
  if (bar) bar.style.width = pct + '%';
}

// ── Mobile sidebar toggle ──
function toggleSidebar() {
  document.querySelector('.sidebar').classList.toggle('open');
  document.querySelector('.menu-overlay').classList.toggle('open');
}

// ── KaTeX auto-render ──
function renderMath() {
  if (typeof renderMathInElement !== 'undefined') {
    renderMathInElement(document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false }
      ],
      throwOnError: false
    });
  }
}

// ═══════════════════════════════════════════════════════════════
// SVG Icons (trusted, hardcoded)
// ═══════════════════════════════════════════════════════════════

const slideIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>';

const arrowIcon = '<svg class="lecture-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>';

const linkIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>';

// ═══════════════════════════════════════════════════════════════
// INDEX PAGE — Module nav + content builder
// All data comes from the hardcoded modules array in index.html.
// No user-supplied data is ever rendered.
// ═══════════════════════════════════════════════════════════════

function buildIndexPage(modules) {
  const nav = document.getElementById('sidebarNav');
  const content = document.getElementById('content');
  if (!nav || !content) return;

  // Build sidebar nav
  modules.forEach(function(m) {
    var section = document.createElement('div');
    section.className = 'nav-section';
    var a = document.createElement('a');
    a.className = 'nav-item';
    a.href = '#' + m.id;
    a.dataset.id = m.id;
    var numSpan = document.createElement('span');
    numSpan.className = 'nav-num';
    numSpan.textContent = m.num;
    var titleSpan = document.createElement('span');
    titleSpan.textContent = m.title;
    a.appendChild(numSpan);
    a.appendChild(titleSpan);
    section.appendChild(a);
    nav.appendChild(section);
  });

  // Build content — uses innerHTML for trusted, hardcoded module data only
  modules.forEach(function(m) {
    var lectureCards = m.lectures.map(function(l) {
      return '<a class="lecture-card" href="' + l.file + '">' +
        '<div class="lecture-icon">' + slideIcon + '</div>' +
        '<div class="lecture-info">' +
          '<div class="lecture-label">' + l.label + ': ' + (l.title || '') + '</div>' +
          '<div class="lecture-meta">Lecture notes</div>' +
        '</div>' +
        arrowIcon +
      '</a>';
    }).join('');

    var readingPills = '';
    if (m.readings && m.readings.length) {
      readingPills = '<div class="readings"><h4>Papers</h4><div class="reading-pills">' +
        m.readings.map(function(r) {
          if (r.url) {
            return '<a class="reading-pill is-link" href="' + r.url + '" target="_blank" rel="noopener">' + linkIcon + ' ' + r.text + '</a>';
          }
          return '<span class="reading-pill">' + r.text + '</span>';
        }).join('') +
      '</div></div>';
    }

    var prereqsHTML = '';
    if (m.prereqs) {
      prereqsHTML = '<div class="prereqs"><h4>Prerequisites</h4><div class="prereqs-grid">' +
        m.prereqs.map(function(p) {
          return '<a class="prereq-card" href="' + p.url + '" target="_blank" rel="noopener">' +
            '<span class="prereq-icon">' + p.icon + '</span>' +
            '<div><div class="prereq-label">' + p.label + '</div>' +
            '<div class="prereq-sub">' + p.sub + '</div></div></a>';
        }).join('') +
      '</div></div>';
    }

    var moduleEl = document.createElement('section');
    moduleEl.id = m.id;

    if (m.isAssessment) {
      moduleEl.className = 'assessment-section';
      // Trusted hardcoded content — safe innerHTML usage
      moduleEl.innerHTML =
        '<div class="assessment-header">' +
          '<div class="assessment-icon">' + (m.icon || '&#x2713;') + '</div>' +
          '<div><div class="assessment-title">' + m.title + '</div>' +
          '<div class="assessment-subtitle">' + m.subtitle + '</div></div>' +
        '</div>' +
        '<div class="module-body"><div>' +
          '<div class="module-description">' + m.description + '</div>' +
        '</div><div>' +
          '<div class="lectures-panel"><h4>Exercises</h4>' + lectureCards + '</div>' +
        '</div></div>';
    } else {
      moduleEl.className = 'module';
      // Trusted hardcoded content — safe innerHTML usage
      moduleEl.innerHTML =
        '<div class="module-header">' +
          '<div class="module-number">' + m.num + '</div>' +
          '<div class="module-title-block">' +
            '<div class="module-title">' + m.title + '</div>' +
            '<div class="module-subtitle">' + m.subtitle + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="module-body"><div>' +
          '<div class="module-description">' + m.description + '</div>' +
          prereqsHTML +
        '</div><div>' +
          '<div class="lectures-panel"><h4>Lectures</h4>' + lectureCards + '</div>' +
          readingPills +
        '</div></div>';
    }

    content.appendChild(moduleEl);
  });

  // Active nav tracking
  var navItems = document.querySelectorAll('.nav-item');
  var sections = document.querySelectorAll('.module, .assessment-section');

  function updateActiveNav() {
    var current = '';
    sections.forEach(function(s) {
      if (s.getBoundingClientRect().top < 200) current = s.id;
    });
    navItems.forEach(function(item) {
      item.classList.toggle('active', item.dataset.id === current);
    });
  }

  window.addEventListener('scroll', function() {
    updateActiveNav();
    updateProgress();
  }, { passive: true });

  updateActiveNav();

  navItems.forEach(function(item) {
    item.addEventListener('click', function() {
      if (window.innerWidth <= 768) toggleSidebar();
    });
  });
}

// ═══════════════════════════════════════════════════════════════
// LECTURE PAGE — ToC from headings (DOM-safe, no innerHTML)
// ═══════════════════════════════════════════════════════════════

function buildLectureNav() {
  var nav = document.getElementById('sidebarNav');
  var article = document.querySelector('.lecture-article');
  if (!nav || !article) return;

  var headings = article.querySelectorAll('h2, h3');
  headings.forEach(function(h) {
    if (!h.id) {
      h.id = h.textContent.trim().toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/(^-|-$)/g, '');
    }

    var isH3 = h.tagName === 'H3';
    var item = document.createElement('a');
    item.className = isH3 ? 'nav-sub-item' : 'nav-item';
    item.href = '#' + h.id;
    item.dataset.id = h.id;
    item.textContent = h.textContent;

    var section = document.createElement('div');
    section.className = 'nav-section';
    section.appendChild(item);
    nav.appendChild(section);
  });

  function updateActiveToc() {
    var current = '';
    headings.forEach(function(h) {
      if (h.getBoundingClientRect().top < 120) current = h.id;
    });
    nav.querySelectorAll('.nav-item, .nav-sub-item').forEach(function(item) {
      item.classList.toggle('active', item.dataset.id === current);
    });
  }

  window.addEventListener('scroll', function() {
    updateActiveToc();
    updateProgress();
  }, { passive: true });

  updateActiveToc();

  nav.querySelectorAll('.nav-item, .nav-sub-item').forEach(function(item) {
    item.addEventListener('click', function() {
      if (window.innerWidth <= 768) toggleSidebar();
    });
  });
}

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', function() {
  if (document.querySelector('.lecture-article')) {
    buildLectureNav();
  }
  renderMath();
  updateProgress();
});
