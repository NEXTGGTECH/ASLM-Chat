// Copyright NEXTGGTECH. Elastic License 2.0.

import { postJson } from '../main/api.js';
import { escHtml, escapeAttributeValue, markdownToPlainText, timeNow } from '../main/utils.js';
import {
  addCitationSource,
  addSegmentCitationSources,
  addSegmentsCitationSources,
  createCitationRegistry,
  decorateCitationsInHtml as decorateCitationHandlesInHtml,
  normalizeCitationBrackets as normalizeCitationHandleBrackets,
  normalizeCitationSpacing as normalizeCitationHandleSpacing,
  replaceCitationHandlesWithSourceUrls
} from './citations-ui.js';
import { bindCitationPreviewCards } from './citation-preview-ui.js';
import { messageDialog } from './dialogs.js';
import { t } from '../main/i18n.js';

// marked's default GFM `del` allows single-tilde pairs (~x~), which breaks ~ for "approximately".
// GitHub-style strikethrough is ~~ only; install tokenizer override once.
let markedStrikethroughDoubleTildeOnlyInstalled = false;

// Message UI.
// Create helpers for rendering messages, activity timelines, and message actions.
export function createMessagesUi(context, dependencies) {
  const { attachmentUi, browserPortalUi, deepResearchUi, toolInspector } = dependencies;
  const { dom, icons, state } = context;
  const MORE_LABEL = t('messages.more', {}, 'More');
  const HIDE_LABEL = t('messages.hide', {}, 'Hide');
  const SEARCH_BATCH_FIRST_CALL_HOLD_MS = 0;
  const SEARCH_BATCH_MULTI_CALL_SETTLE_MS = 0;
  const PRE_TOOL_TEXT_HOLD_MS = 0;
  const STREAM_TEXT_REVEAL_INITIAL_DELAY_MS = 180;
  const STREAM_TEXT_REVEAL_CHARACTER_MS = 7;
  const STREAM_TEXT_REVEAL_FINAL_CHARACTER_MS = 3;
  const STREAM_TEXT_REVEAL_FRAME_MS = 28;
  const STREAM_TEXT_REVEAL_TAIL_CHARS = 18;
  const WRITE_PREVIEW_COLLAPSED_LINES = 4;
  const WRITE_PREVIEW_EXPANDED_LINES = 80;
  const EDIT_PREVIEW_COLLAPSED_ROWS = 4;
  const EDIT_PREVIEW_EXPANDED_ROWS = 120;
  const SANDBOX_INPUT_PREVIEW_CHARS = 12000;
  const REASONING_CHUNK_TARGET_CHARS = 132;
  const REASONING_CHUNK_MAX_CHARS = 168;
  const REASONING_LABEL_CHANGE_DELAY_MS = 1400;
  const REASONING_LABEL_MAX_TYPING_DURATION_MS = 700;
  const REASONING_LABEL_MAX_CHARACTER_DELAY_MS = 26;
  const REASONING_LABEL_MIN_CHARACTER_DELAY_MS = 10;
  const REASONING_LABEL_TYPING_SPEED_MULTIPLIER = 1.4;
  const REASONING_LABEL_DELETE_SPEED_MULTIPLIER = 3.3;
  const REASONING_LABEL_MIN_CHARS = 12;
  const REASONING_LABEL_MIN_WORDS = 3;
  const REASONING_LABEL_MAX_CHARS = 140;
  const REASONING_PREVIEW_BRACKET_PAIRS = {
    '(': ')',
    '[': ']',
    '{': '}',
    '\uFF08': '\uFF09',
    '\uFF3B': '\uFF3D',
    '\uFF5B': '\uFF5D',
    '\u3010': '\u3011',
    '\u3014': '\u3015',
    '\u3016': '\u3017',
    '\u27E6': '\u27E7'
  };
  const REASONING_PREVIEW_QUOTE_PAIRS = {
    '"': '"',
    '\u00AB': '\u00BB',
    '\u201C': '\u201D',
    '\u201E': '\u201C',
    '\u2018': '\u2019'
  };
  const REASONING_PREVIEW_NAMED_CITATION_HANDLE_SOURCE = String.raw`(?:s\d+|source-(?:[a-z0-9]+-)?\d+|c[a-z0-9]{3,}-\d+)`;
  const REASONING_PREVIEW_BRACKETED_CITATION_HANDLE_SOURCE = String.raw`(?:\d{4,}-\d+|${REASONING_PREVIEW_NAMED_CITATION_HANDLE_SOURCE})`;
  const REASONING_PREVIEW_BRACKETED_CITATION_PATTERN = new RegExp(
    String.raw`[\[\u3010\uFF3B\u3014\u3016\u27E6]\s*${REASONING_PREVIEW_BRACKETED_CITATION_HANDLE_SOURCE}(?:\s*,\s*${REASONING_PREVIEW_BRACKETED_CITATION_HANDLE_SOURCE})*\s*[\]\u3011\uFF3D\u3015\u3017\u27E7]`,
    'gi'
  );
  const REASONING_PREVIEW_INTERNAL_MARKER_PATTERN = new RegExp(
    String.raw`\b(?:source-(?:[a-z0-9]+-)?\d+|c[a-z0-9]{3,}-\d+|turn\d+(?:search|fetch|view|open|click|image|news)\d+)\b`,
    'gi'
  );
  const REASONING_PREVIEW_PRIVATE_MARKER_PATTERN = /\uE200[^\uE201]*(?:\uE201|$)/g;
  const REASONING_PREVIEW_INVISIBLE_PATTERN = /[\u034f\u061c\u180e\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]/g;
  const DOWNLOAD_FILE_ICON = icons.DOWNLOAD_FILE_ICON || '';
  const SHARED_FILE_FOLDER_ICON = icons.SKILLS_FOLDER_ICON || '';
  const OPEN_FILE_LOCATION_LABEL = t('messages.openFileLocation', null, 'Open file location');
  const MEDIA_PLAY_ICON = icons.PLAY_ICON || '&#9658;';
  const MEDIA_PAUSE_ICON = icons.PAUSE_ICON || '&#10073;&#10073;';
  const MEDIA_VOLUME_ICON = icons.VOLUME_ICON || '';
  const MEDIA_VOLUME_MUTED_ICON = icons.VOLUME_MUTED_ICON || MEDIA_VOLUME_ICON;
  const MEDIA_FULLSCREEN_ICON = icons.FULLSCREEN_ICON || '';
  const MEDIA_FULLSCREEN_EXIT_ICON = icons.FULLSCREEN_EXIT_ICON || MEDIA_FULLSCREEN_ICON;
  const MEDIA_POPOUT_ICON = icons.POPOUT_ICON || '';
  const MEDIA_DOCK_ICON = icons.DOCK_ICON || '';
  const MEDIA_CLOSE_ICON = icons.CLOSE_ICON || '&times;';
  const MEDIA_AUDIO_ICON = icons.AUDIO_FILE_ICON || '';
  const MEDIA_VIDEO_ICON = icons.VIDEO_FILE_ICON || '';
  const CODE_SOURCE_ICON = icons.CODE_ICON || '&lt;/&gt;';
  const CODE_PREVIEW_ICON = icons.PLAY_ICON || '&#9658;';
  const MERMAID_ZOOM_OUT_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><path d="M8 11h6M16.5 16.5 21 21"></path></svg>';
  const MERMAID_ZOOM_IN_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><path d="M8 11h6M11 8v6M16.5 16.5 21 21"></path></svg>';
  const MERMAID_FIT_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 3H3v5M16 3h5v5M8 21H3v-5M16 21h5v-5"></path></svg>';
  const MERMAID_DOWNLOAD_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3v12M7 10l5 5 5-5M5 21h14"></path></svg>';
  const loadedSandboxImageSrcs = new Set();
  const sandboxImageLoadStateBySrc = new Map();
  const floatingMediaPlaceholders = new WeakMap();
  const activeMediaFrameIds = new WeakMap();
  const HEAVY_TOOL_ARGUMENT_KEYS = {
    bash: ['stdin'],
    edit: ['content', 'new_str', 'old_str'],
    write: ['content']
  };
  const REASONING_TAG_PAIRS = [
    { start: '<think>', end: '</think>' },
    { start: '<thinking>', end: '</thinking>' },
    { start: '<reasoning>', end: '</reasoning>' },
    { start: '<analysis>', end: '</analysis>' }
  ];
  const REASONING_BOUNDARY_PUNCTUATION_PATTERN = /\p{P}/u;
  const STREAMING_ACTIVITY_MARKERS = [
    '<tool_call>',
    '</tool_call>',
    '<tool_result>',
    '</tool_result>',
    '<tool_activity>',
    '</tool_activity>',
    '<context_compression>',
    '</context_compression>',
    '<think>',
    '</think>',
    '<thinking>',
    '</thinking>',
    '<reasoning>',
    '</reasoning>',
    '<analysis>',
    '</analysis>'
  ];
  let mermaidRenderSeq = 0;

  // A real reasoning tag begins at a text boundary, not in the middle of prose.
  function isReasoningStartBoundary(value, tagIndex) {
    const source = String(value || '');
    let cursor = Number(tagIndex) - 1;
    while (cursor >= 0 && /[ \t\f\v]/.test(source.charAt(cursor))) {
      cursor -= 1;
    }
    if (cursor < 0) {
      return true;
    }
    const previous = source.charAt(cursor);
    return previous === '\n'
      || previous === '\r'
      || previous === '>'
      || REASONING_BOUNDARY_PUNCTUATION_PATTERN.test(previous);
  }

  // Find the next opening tag that is positioned like a control block.
  function findNextReasoningStart(value, fromIndex) {
    const source = String(value || '');
    const lowerSource = source.toLowerCase();
    let best = null;

    REASONING_TAG_PAIRS.forEach(function checkPair(pair) {
      let pos = lowerSource.indexOf(pair.start, fromIndex);
      while (pos !== -1 && !isReasoningStartBoundary(source, pos)) {
        pos = lowerSource.indexOf(pair.start, pos + pair.start.length);
      }
      if (pos !== -1 && (!best || pos < best.pos)) {
        best = { pos, kind: 'thought', pair };
      }
    });
    return best;
  }

  // Composer state.
  // Sync both send buttons with the current generation and attachment state.
  function updateSendButtons() {
    const hasPendingAttachments = state.attachmentState.pending.length > 0;
    const hasBlockedAttachments = state.attachmentState.pending.some(function isBlocked(attachment) {
      return attachment && (attachment.status === 'uploading' || attachment.status === 'error');
    });

    function syncComposerButton($button, $input) {
      const hasDraft = !!String($input.val() || '').trim() || hasPendingAttachments;

      // During generation an empty composer acts as a stop button. As soon as
      // the user types or attaches something, the same button becomes an
      // enabled send button so follow-up messages can be queued instead of
      // feeling locked.
      if (state.isChatGenerating && !hasDraft) {
        $button
          .prop('disabled', false)
          .addClass('stop-btn')
          .html(icons.STOP_ICON)
          .attr('aria-label', t('composer.stopGeneration', {}, 'Stop generation'));
        return;
      }

      $button
        .removeClass('stop-btn')
        .html(icons.SEND_ICON)
        .attr('aria-label', state.isChatGenerating ? t('composer.queueMessage', {}, 'Queue message') : t('composer.sendMessage', {}, 'Send Message'))
        .prop('disabled', !hasDraft || hasBlockedAttachments);
    }

    syncComposerButton(dom.$sendBtn, dom.$chatInput);
    syncComposerButton(dom.$sendBtnConv, dom.$chatInputConv);
  }


  // Show regen buttons only on the latest assistant exchange.
  function updateRegenButtons() {
    dom.$messagesInner.find('.msg-regen-btn').hide();

    const $lastAssistant = dom.$messagesInner.find('.msg.assistant').last();
    if (!$lastAssistant.length) {
      return;
    }

    $lastAssistant.find('.msg-regen-btn').show();

    const $prev = $lastAssistant.prev('.msg.user');
    if ($prev.length) {
      $prev.find('.msg-regen-btn').show();
    }
  }

  // Scroll the message area to the bottom.
  function scrollBottom() {
    dom.$messagesArea.scrollTop(dom.$messagesArea[0].scrollHeight);
  }


  // Markdown rendering.
  // Report whether the character at index is escaped by a backslash run.
  function isEscapedAt(text, index) {
    let backslashCount = 0;
    for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor -= 1) {
      backslashCount += 1;
    }
    return backslashCount % 2 === 1;
  }

  // Find the next unescaped occurrence of one substring.
  function findUnescaped(text, needle, fromIndex) {
    let cursor = Math.max(0, Number(fromIndex) || 0);
    while (cursor < text.length) {
      const index = text.indexOf(needle, cursor);
      if (index === -1) {
        return -1;
      }
      if (!isEscapedAt(text, index)) {
        return index;
      }
      cursor = index + needle.length;
    }
    return -1;
  }

  // Find the earliest LaTeX delimiter starting at one cursor position.
  function nextLatexDelimiter(text, fromIndex) {
    const candidates = [
      { open: '$$', close: '$$', displayMode: true },
      { open: '\\[', close: '\\]', displayMode: true },
      { open: '\\(', close: '\\)', displayMode: false },
      { open: '$', close: '$', displayMode: false }
    ];
    let best = null;
    candidates.forEach(function findDelimiter(delimiter) {
      const index = findUnescaped(text, delimiter.open, fromIndex);
      if (index === -1) {
        return;
      }
      if (delimiter.open === '$') {
        const before = index > 0 ? text[index - 1] : '';
        const after = index + 1 < text.length ? text[index + 1] : '';
        if (/\d/.test(before) || /\s/.test(after || '')) {
          return;
        }
      }
      if (!best || index < best.index || (index === best.index && delimiter.open.length > best.open.length)) {
        best = { ...delimiter, index };
      }
    });
    return best;
  }

  // Render one LaTeX fragment with KaTeX when available.
  function renderLatexSourceToHtml(latexSource, displayMode) {
    if (typeof katex === 'undefined' || !katex.renderToString) {
      return escHtml(latexSource);
    }
    return katex.renderToString(latexSource, {
      displayMode: !!displayMode,
      throwOnError: false,
      strict: 'ignore',
      trust: false,
      output: 'htmlAndMathml'
    });
  }

  // Append rendered LaTeX nodes into one document fragment.
  function appendLatexHtml(fragment, latexSource, displayMode) {
    if (typeof katex === 'undefined' || !katex.renderToString) {
      fragment.appendChild(document.createTextNode(latexSource));
      return;
    }
    const template = document.createElement('template');
    template.innerHTML = renderLatexSourceToHtml(latexSource, displayMode);
    fragment.appendChild(template.content);
  }

  // Replace LaTeX delimiters inside one text node with KaTeX output.
  function replaceLatexInTextNode(textNode) {
    const source = textNode.nodeValue || '';
    const fragment = document.createDocumentFragment();
    let cursor = 0;
    let replaced = false;

    while (cursor < source.length) {
      const delimiter = nextLatexDelimiter(source, cursor);
      if (!delimiter) {
        break;
      }
      const contentStart = delimiter.index + delimiter.open.length;
      const closeIndex = findUnescaped(source, delimiter.close, contentStart);
      if (closeIndex === -1) {
        break;
      }
      const latexSource = source.slice(contentStart, closeIndex).trim();
      if (!latexSource) {
        cursor = closeIndex + delimiter.close.length;
        continue;
      }
      if (delimiter.index > cursor) {
        fragment.appendChild(document.createTextNode(source.slice(cursor, delimiter.index)));
      }
      appendLatexHtml(fragment, latexSource, delimiter.displayMode);
      replaced = true;
      cursor = closeIndex + delimiter.close.length;
    }

    if (!replaced) {
      return;
    }
    if (cursor < source.length) {
      fragment.appendChild(document.createTextNode(source.slice(cursor)));
    }
    textNode.parentNode.replaceChild(fragment, textNode);
  }

  // Render LaTeX delimiters across one HTML string.
  function renderLatexInHtml(html) {
    if (typeof document === 'undefined' || typeof katex === 'undefined') {
      return html;
    }
    const template = document.createElement('template');
    template.innerHTML = String(html || '');
    const ignoredTags = new Set(['CODE', 'PRE', 'SCRIPT', 'STYLE', 'TEXTAREA']);
    const walker = document.createTreeWalker(template.content, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        let parent = node.parentNode;
        while (parent && parent !== template.content) {
          if (ignoredTags.has(parent.nodeName) || (parent.classList && parent.classList.contains('katex'))) {
            return NodeFilter.FILTER_REJECT;
          }
          parent = parent.parentNode;
        }
        return /(?:\$|\\\[|\\\()/.test(node.nodeValue || '')
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      }
    });
    const nodes = [];
    while (walker.nextNode()) {
      nodes.push(walker.currentNode);
    }
    nodes.forEach(replaceLatexInTextNode);
    return template.innerHTML;
  }

  // Heuristically detect whether plain text looks like LaTeX math.
  function looksLikeLatexSource(source) {
    const text = String(source || '').trim();
    if (!text) {
      return false;
    }
    return /\\(?:boxed|mathrm|frac|sqrt|cdot|times|longrightarrow|rightarrow|left|right|text|begin|end|sum|int|alpha|beta|gamma|delta|Delta|Omega)\b/.test(text)
      || (/\\/.test(text) && /[_^]\{?[\w()+\-]+/.test(text));
  }

  // Convert bracket-style display math into $$ blocks before markdown parse.
  function normalizeLooseDisplayLatex(source) {
    return String(source || '')
      .replace(/(^|\n)[ \t]*\\\[\s*\n?([\s\S]*?)\n?[ \t]*\\\](?=\n|$)/g, function normalizeEscapedLatexBlock(match, prefix, body) {
        const latexSource = String(body || '').trim();
        if (!looksLikeLatexSource(latexSource)) {
          return match;
        }
        return `${prefix}$$\n${latexSource}\n$$`;
      })
      .replace(/(^|\n)[ \t]*\[\s*\n([\s\S]*?)\n[ \t]*\](?=\n|$)/g, function normalizeLatexBlock(match, prefix, body) {
      const latexSource = String(body || '').trim();
      if (!looksLikeLatexSource(latexSource)) {
        return match;
      }
      return `${prefix}$$\n${latexSource}\n$$`;
    });
  }

  // Replace display and inline math with placeholders before markdown parse.
  function extractLatexBlocks(source) {
    const blocks = [];
    const text = String(source || '')
      .replace(/(^|\n)[ \t]*\$\$\s*\n?([\s\S]*?)\n?[ \t]*\$\$(?=\n|$)/g, function extractDisplayLatexBlock(match, prefix, body) {
        const latexSource = String(body || '').trim();
        if (!looksLikeLatexSource(latexSource)) {
          return match;
        }
        const token = `ASLMLATEXDISPLAY${blocks.length}TOKEN`;
        blocks.push({ token, latexSource, displayMode: true });
        return `${prefix}\n\n${token}\n\n`;
      })
      .replace(/\\\(([\s\S]*?)\\\)/g, function extractInlineLatexBlock(match, body) {
        const latexSource = String(body || '').trim();
        if (!looksLikeLatexSource(latexSource)) {
          return match;
        }
        const token = `ASLMLATEXINLINE${blocks.length}TOKEN`;
        blocks.push({ token, latexSource, displayMode: false });
        return token;
      });
    return { text, blocks };
  }

  // Swap LaTeX placeholder tokens back to rendered KaTeX HTML.
  function restoreLatexPlaceholders(html, blocks) {
    let result = String(html || '');
    (blocks || []).forEach(function restoreLatexBlock(block) {
      const token = String(block.token || '');
      if (!token) {
        return;
      }
      const rendered = renderLatexSourceToHtml(block.latexSource || '', block.displayMode !== false);
      result = result
        .split(`<p>${token}</p>`)
        .join(rendered)
        .split(token)
        .join(rendered);
    });
    return result;
  }


  // Markdown code-block helpers.
  // Read the raw language class from one fenced code element.
  function markdownRawCodeLanguage(codeEl) {
    const classNames = String((codeEl && codeEl.className) || '');
    const match = classNames.match(/(?:^|\s)language-([a-z0-9_+#.-]+)/i)
      || classNames.match(/(?:^|\s)lang-([a-z0-9_+#.-]+)/i);
    return String(match ? match[1] : '').trim().toLowerCase();
  }

  // Syntax highlighting helpers.
  function markdownCodeLanguage(codeEl) {
    const rawLanguage = markdownRawCodeLanguage(codeEl);
    const displayAliases = {
      htm: 'HTML',
      html: 'HTML',
      javascript: 'JAVASCRIPT',
      js: 'JAVASCRIPT',
      svg: 'SVG'
    };
    if (displayAliases[rawLanguage]) {
      return displayAliases[rawLanguage];
    }
    const normalized = normalizeHighlightLanguage(rawLanguage);
    return normalized === 'plaintext' ? 'Code' : normalized.toUpperCase();
  }

  // Resolve the highlight.js language id for one code element.
  function markdownCodeHighlightLanguage(codeEl) {
    return normalizeHighlightLanguage(markdownRawCodeLanguage(codeEl));
  }

  // Report whether one code block should render as Mermaid.
  function isMermaidCodeLanguage(codeEl) {
    return markdownRawCodeLanguage(codeEl) === 'mermaid';
  }

  // Resolve web languages that can run natively inside a sandboxed browser frame.
  function markdownCodePreviewKind(codeEl) {
    const language = markdownRawCodeLanguage(codeEl);
    const aliases = {
      css: 'css',
      htm: 'html',
      html: 'html',
      javascript: 'javascript',
      js: 'javascript',
      svg: 'svg'
    };
    return aliases[language] || '';
  }

  // Keep only http and https URLs safe for markdown links.
  function safeMarkdownUrl(value) {
    try {
      const parsed = new URL(String(value || '').trim());
      return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.href : '';
    } catch (_error) {
      return '';
    }
  }

  // Turn bare URLs inside inline code spans into links.
  function linkifyInlineCodeUrls(template) {
    template.content.querySelectorAll('code').forEach(function linkifyInlineCodeUrl(codeEl) {
      if (codeEl.closest('pre, a')) {
        return;
      }

      const rawText = String(codeEl.textContent || '').trim();
      if (!rawText || /\s/.test(rawText)) {
        return;
      }

      const url = safeMarkdownUrl(rawText);
      if (!url) {
        return;
      }

      const linkEl = document.createElement('a');
      linkEl.className = 'md-inline-code-link';
      linkEl.href = url;
      linkEl.target = '_blank';
      linkEl.rel = 'noopener noreferrer';
      codeEl.parentNode.insertBefore(linkEl, codeEl);
      linkEl.appendChild(codeEl);
    });
  }

  // Wrap fenced code blocks with copy UI and highlighting.
  function enhanceMarkdownCodeBlocks(html) {
    if (typeof document === 'undefined' || !html) {
      return html;
    }

    const template = document.createElement('template');
    template.innerHTML = String(html || '');
    linkifyInlineCodeUrls(template);
    template.content.querySelectorAll('pre > code').forEach(function wrapMarkdownCodeBlock(codeEl) {
      const preEl = codeEl.parentElement;
      if (!preEl || preEl.closest('.md-code-card, .md-mermaid-card')) {
        return;
      }

      if (isMermaidCodeLanguage(codeEl)) {
        const source = codeEl.textContent || '';
        const cardEl = document.createElement('div');
        cardEl.className = 'md-mermaid-card';
        cardEl.dataset.mermaidState = 'pending';

        const headerEl = document.createElement('div');
        headerEl.className = 'md-mermaid-head';
        headerEl.innerHTML = `
          <span class="md-code-lang">
            <span class="md-code-icon" aria-hidden="true">&lt;&gt;</span>
            <span>MERMAID</span>
          </span>
          <button type="button" class="md-code-copy-btn" title="Copy diagram source" aria-label="Copy diagram source">${icons.COPY_MESSAGE_ICON}</button>
        `;

        const canvasEl = document.createElement('div');
        canvasEl.className = 'md-mermaid-canvas';
        canvasEl.setAttribute('role', 'img');
        canvasEl.setAttribute('aria-label', 'Interactive Mermaid diagram');
        canvasEl.setAttribute('tabindex', '0');

        const statusEl = document.createElement('div');
        statusEl.className = 'md-mermaid-status';
        statusEl.textContent = 'Rendering diagram...';

        preEl.classList.add('md-mermaid-source');
        preEl.parentNode.insertBefore(cardEl, preEl);
        cardEl.appendChild(headerEl);
        cardEl.appendChild(canvasEl);
        cardEl.appendChild(statusEl);
        cardEl.appendChild(preEl);
        codeEl.textContent = source;
        return;
      }

      const language = markdownCodeHighlightLanguage(codeEl);
      const previewKind = markdownCodePreviewKind(codeEl);
      const source = codeEl.textContent || '';
      const safeClassLanguage = language.replace(/[^a-z0-9_-]/gi, '') || 'plaintext';
      codeEl.innerHTML = highlightCode(source, language);
      codeEl.classList.add('hljs', `language-${safeClassLanguage}`);

      const cardEl = document.createElement('div');
      cardEl.className = 'md-code-card';
      if (previewKind) {
        cardEl.classList.add('md-code-card--previewable');
        cardEl.dataset.codePreviewKind = previewKind;
        cardEl.dataset.codeView = 'source';
      }

      const headerEl = document.createElement('div');
      headerEl.className = 'md-code-head';
      headerEl.innerHTML = `
        <span class="md-code-lang">
          <span class="md-code-icon" aria-hidden="true">&lt;/&gt;</span>
          <span>${escHtml(markdownCodeLanguage(codeEl))}</span>
        </span>
        <span class="md-code-actions">
          ${previewKind ? `
            <button type="button" class="md-code-action-btn md-code-view-btn is-active" data-code-action="source" title="View code" aria-label="View code" aria-pressed="true">${CODE_SOURCE_ICON}</button>
            <button type="button" class="md-code-action-btn md-code-view-btn" data-code-action="preview" title="Run preview" aria-label="Run preview" aria-pressed="false">${CODE_PREVIEW_ICON}</button>
          ` : ''}
          <button type="button" class="md-code-copy-btn" title="Copy code" aria-label="Copy code">${icons.COPY_MESSAGE_ICON}</button>
        </span>
      `;

      preEl.parentNode.insertBefore(cardEl, preEl);
      cardEl.appendChild(headerEl);
      cardEl.appendChild(preEl);
      if (previewKind) {
        const previewEl = document.createElement('div');
        previewEl.className = 'md-code-preview';
        previewEl.hidden = true;

        const frameEl = document.createElement('iframe');
        frameEl.className = 'md-code-preview-frame';
        frameEl.setAttribute('sandbox', 'allow-scripts');
        frameEl.setAttribute('referrerpolicy', 'no-referrer');
        frameEl.setAttribute('loading', 'lazy');
        frameEl.setAttribute('title', `${markdownCodeLanguage(codeEl)} rendered preview`);
        previewEl.appendChild(frameEl);
        cardEl.appendChild(previewEl);
      }
    });

    return template.innerHTML;
  }

  // Render one visible text segment as sanitized HTML.
  function renderMarkdownSegment(content, citationSources) {
    // Inline image tokens leave the markdown pipeline as plain markers and re-enter
    // the sanitized HTML as hydratable nodes (same pattern as LaTeX placeholders).
    const imageTokens = extractImageTokenBlocks(
      normalizeCitationHandleSpacing(normalizeCitationHandleBrackets(content))
    );
    const normalizedContent = normalizeLooseDisplayLatex(imageTokens.text);
    const visibleContent = normalizedContent;
    const latexBlocks = extractLatexBlocks(visibleContent);
    if (typeof marked === 'undefined' || typeof DOMPurify === 'undefined') {
      const fallbackHtml = restoreImagePlaceholders(
        restoreLatexPlaceholders(
          decorateCitationHandlesInHtml(escHtml(latexBlocks.text), citationSources),
          latexBlocks.blocks
        ),
        imageTokens.blocks
      );
      return enhanceMarkdownCodeBlocks(renderLatexInHtml(fallbackHtml));
    }

    const html = restoreImagePlaceholders(
      restoreLatexPlaceholders(
        decorateCitationHandlesInHtml(DOMPurify.sanitize(marked.parse(latexBlocks.text)), citationSources),
        latexBlocks.blocks
      ),
      imageTokens.blocks
    );
    return enhanceMarkdownCodeBlocks(renderLatexInHtml(html));
  }

  // Render text cheaply while a message is still streaming.
  function renderPlainTextSegment(content) {
    return escHtml(content);
  }

  // Normalize a language id to a supported highlight.js id.
  function normalizeHighlightLanguage(language) {
    const raw = String(language || '').trim().toLowerCase();
    const aliases = {
      bash: 'bash',
      sh: 'bash',
      shell: 'bash',
      zsh: 'bash',
      ps1: 'powershell',
      py: 'python',
      python: 'python',
      js: 'javascript',
      jsx: 'javascript',
      mjs: 'javascript',
      cjs: 'javascript',
      ts: 'typescript',
      tsx: 'typescript',
      html: 'xml',
      htm: 'xml',
      svg: 'xml',
      xml: 'xml',
      css: 'css',
      scss: 'scss',
      json: 'json',
      jsonl: 'json',
      md: 'markdown',
      markdown: 'markdown',
      yaml: 'yaml',
      yml: 'yaml',
      diff: 'diff'
    };
    const normalized = aliases[raw] || raw || 'plaintext';
    if (typeof hljs !== 'undefined' && hljs.getLanguage && hljs.getLanguage(normalized)) {
      return normalized;
    }
    return 'plaintext';
  }

  // Infer a highlight language from a file path extension.
  function languageFromPath(path, fallback) {
    const cleanPath = String(path || '').split(/[?#]/)[0];
    const baseName = cleanPath.split(/[\\/]/).pop().toLowerCase();
    const specialNames = {
      dockerfile: 'dockerfile',
      makefile: 'makefile',
      'requirements.txt': 'plaintext',
      'package.json': 'json',
      'tsconfig.json': 'json'
    };
    if (specialNames[baseName]) {
      return normalizeHighlightLanguage(specialNames[baseName]);
    }

    const extension = cleanPath.split('.').pop();
    if (!extension || extension === cleanPath) {
      return normalizeHighlightLanguage(fallback || 'plaintext');
    }
    return normalizeHighlightLanguage(extension);
  }

  // Highlight source text with highlight.js when available.
  function highlightCode(code, language) {
    const text = String(code || '');
    const safeLanguage = normalizeHighlightLanguage(language);
    if (typeof hljs === 'undefined' || !hljs.highlight || safeLanguage === 'plaintext') {
      return escHtml(text);
    }

    try {
      return hljs.highlight(text, {
        language: safeLanguage,
        ignoreIllegals: true
      }).value;
    } catch (_error) {
      return escHtml(text);
    }
  }




  // Search source chip helpers.
  function safeExternalUrl(value) {
    const rawValue = String(value || '').trim();
    if (!rawValue) {
      return '';
    }

    try {
      const parsed = new URL(rawValue);
      return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.href : '';
    } catch (_error) {
      return '';
    }
  }

  // Read the display domain from one search source record.
  function faviconDomain(source) {
    const safeSource = source && typeof source === 'object' ? source : {};
    const directDomain = String(safeSource.domain || '').trim();
    if (directDomain) {
      return directDomain.replace(/^www\./i, '');
    }

    try {
      return new URL(String(safeSource.url || '')).hostname.replace(/^www\./i, '');
    } catch (_error) {
      return '';
    }
  }

  // Build the favicon proxy URL for one domain.
  function localFaviconUrlForDomain(domain) {
    const cleanDomain = String(domain || '').trim().replace(/^www\./i, '');
    if (!cleanDomain || /[^a-z0-9.-]/i.test(cleanDomain)) {
      return '';
    }
    return `/api/favicon/?domain=${encodeURIComponent(cleanDomain)}`;
  }

  // Report whether a source record includes extracted preview text.
  function sourceHasExtractedPreview(source) {
    const safeSource = source && typeof source === 'object' ? source : {};
    const hasPreviewField = Object.prototype.hasOwnProperty.call(safeSource, 'preview');
    const hasSnippetField = Object.prototype.hasOwnProperty.call(safeSource, 'snippet');
    if (!hasPreviewField && !hasSnippetField) {
      return true;
    }

    const preview = String(safeSource.preview || '').trim();
    const snippet = String(safeSource.snippet || '').trim();
    return !!preview && preview !== snippet;
  }

  // Resolve the favicon URL shown on one source chip.
  function sourceFaviconUrl(source) {
    const safeSource = source && typeof source === 'object' ? source : {};
    const localFaviconUrl = localFaviconUrlForDomain(faviconDomain(safeSource));
    if (localFaviconUrl) {
      return localFaviconUrl;
    }

    if (!sourceHasExtractedPreview(safeSource)) {
      return '';
    }
    return safeExternalUrl(source.favicon_url);
  }

  // Build CSS variables for a domain-colored chip accent.
  function domainAccentStyle(value, extraStyle) {
    const text = String(value || '').trim().toLowerCase();
    let hash = 0;
    for (let index = 0; index < text.length; index += 1) {
      hash = ((hash << 5) - hash) + text.charCodeAt(index);
      hash |= 0;
    }

    const hue = Math.abs(hash) % 360;
    const base = `--msg-source-accent-bg: hsla(${hue}, 68%, 46%, 0.36); --msg-source-accent-fg: hsl(${hue}, 86%, 82%);`;
    return extraStyle ? `${base} ${extraStyle}` : base;
  }




  // Activity parsing.
  // Parse model output into visible text, thoughts, and tool events.
  function findOpenActivityMarker(rawText, markerPairs) {
    const source = String(rawText || '');
    const lowerSource = source.toLowerCase();
    let best = null;

    (markerPairs || []).forEach(function findMarker(pair) {
      const open = String(pair.open || '').toLowerCase();
      const close = String(pair.close || '').toLowerCase();
      if (!open || !close) {
        return;
      }

      let searchFrom = 0;
      while (searchFrom < lowerSource.length) {
        const openIndex = lowerSource.indexOf(open, searchFrom);
        if (openIndex === -1) {
          break;
        }

        const closeIndex = lowerSource.indexOf(close, openIndex + open.length);
        if (closeIndex === -1) {
          if (!best || openIndex < best.pos) {
            best = { pos: openIndex, open, close };
          }
          break;
        }

        searchFrom = closeIndex + close.length;
      }
    });

    return best;
  }

  // Streaming activity tag helpers.
  function openToolPayloadInfo(rawText) {
    return findOpenActivityMarker(rawText, [
      { open: '<tool_call>', close: '</tool_call>' },
      { open: '<tool_result>', close: '</tool_result>' },
      { open: '<tool_activity>', close: '</tool_activity>' },
      { open: '<context_compression>', close: '</context_compression>' }
    ]);
  }

  // Report whether streaming text has an open tool payload tag.
  function hasOpenToolPayload(rawText) {
    return !!openToolPayloadInfo(rawText);
  }

  // Remove trailing open tool payload markup from streaming text.
  function stripOpenToolPayload(rawText) {
    const source = String(rawText || '');
    const openPayload = openToolPayloadInfo(source);
    return openPayload ? source.slice(0, openPayload.pos) : source;
  }

  // Detect a partial activity tag suffix during streaming.
  function trailingPartialActivityMarkerInfo(rawText) {
    const source = String(rawText || '');
    const lowerSource = source.toLowerCase();
    let best = null;

    if (!lowerSource) {
      return null;
    }

    STREAMING_ACTIVITY_MARKERS.forEach(function checkMarker(marker) {
      const normalizedMarker = String(marker || '').toLowerCase();
      const maxLength = Math.min(normalizedMarker.length - 1, lowerSource.length);

      for (let length = maxLength; length > 0; length -= 1) {
        const suffix = lowerSource.slice(-length);
        if (!normalizedMarker.startsWith(suffix)) {
          continue;
        }

        if (!best || length > best.length) {
          best = {
            length,
            marker: normalizedMarker,
            isTool: /^<\/?(tool|context_compression)/i.test(suffix)
          };
        }
        break;
      }
    });

    return best;
  }

  // Remove a trailing partial activity tag from streaming text.
  function stripTrailingPartialActivityMarker(rawText) {
    const source = String(rawText || '');
    const partialMarker = trailingPartialActivityMarkerInfo(source);
    return partialMarker ? source.slice(0, source.length - partialMarker.length) : source;
  }

  // Parse a trailing JSON shorthand into a shared file payload.
  function sharedFilePayloadFromTrailingShorthand(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return null;
    }
    const keys = Object.keys(value);
    const allowedKeys = new Set(['file', 'path', 'filename', 'name', 'mime_type', 'mimeType', 'type', 'size_bytes', 'sizeBytes']);
    if (!keys.length || keys.some(function hasUnexpectedKey(key) { return !allowedKeys.has(key); })) {
      return null;
    }
    const hasFileKey = Object.prototype.hasOwnProperty.call(value, 'file')
      || Object.prototype.hasOwnProperty.call(value, 'path')
      || Object.prototype.hasOwnProperty.call(value, 'filename');
    if (!hasFileKey) {
      return null;
    }

    const path = String(value.path || value.file || value.filename || '').trim();
    if (!path || path.length > 512 || /[\r\n\x00]/.test(path)) {
      return null;
    }
    const basename = path.split(/[\\/]/).pop() || path;
    if (!/\.[a-z0-9]{1,12}$/i.test(basename)) {
      return null;
    }

    const filename = String(value.filename || value.name || basename || 'download').trim();
    const mimeType = String(value.mime_type || value.mimeType || value.type || '').trim();
    return {
      kind: 'shared_file',
      path,
      filename,
      mime_type: mimeType || '',
      type_label: mimeType || 'File',
      size_bytes: Number(value.size_bytes ?? value.sizeBytes ?? -1)
    };
  }

  // Build one synthetic tool segment for a shared file shorthand.
  function sharedFileSegmentFromTrailingShorthand(sharedFile) {
    const safePath = sharedFile && sharedFile.path ? sharedFile.path : '';
    return {
      type: 'tool',
      alias: `trailing_share_file_${safePath.replace(/[^a-z0-9]+/gi, '_').slice(0, 80) || 'file'}`,
      serverId: 'ui',
      serverName: 'UI',
      toolId: 'share_file',
      toolName: 'Share File',
      arguments: { path: sharedFile.path, filename: sharedFile.filename },
      result: '',
      toolUi: { kind: 'shared_file', status: 'done', file: sharedFile },
      structuredContent: { kind: 'shared_file', file: sharedFile }
    };
  }

  // Convert trailing file shorthands into tool segments.
  function normalizeTrailingSharedFileShorthandSegments(rawSegments) {
    const inputSegments = Array.isArray(rawSegments) ? rawSegments : [];
    const outputSegments = [];
    inputSegments.forEach(function normalizeSegment(segment) {
      if (!segment || segment.type !== 'text') {
        outputSegments.push(segment);
        return;
      }

      const content = String(segment.content || '');
      const inlineRecoveredFiles = [];
      const inlineLines = content.split(/\r?\n/);
      let inFence = false;
      const cleanedInlineLines = inlineLines.map(function cleanInlineLine(rawLine) {
        const line = String(rawLine || '');
        if (/^\s*```/.test(line)) {
          inFence = !inFence;
          return line;
        }
        if (inFence) {
          return line;
        }

        return line.replace(/\{\s*"?(file|path|filename)"?\s*:\s*"([^"\r\n]+)"\s*\}/gi, function replaceShorthand(match, _key, filePath) {
          const sharedFile = sharedFilePayloadFromTrailingShorthand({ file: String(filePath || '').trim() });
          if (sharedFile) {
            inlineRecoveredFiles.push(sharedFile);
            return '';
          }
          return match;
        }).replace(/\s{2,}/g, ' ').replace(/\s+:/g, ':').replace(/\s+$/g, '');
      });
      const cleanedContent = cleanedInlineLines.join('\n');

      const lines = content.split(/\r?\n/);
      let lastIndex = lines.length - 1;
      while (lastIndex >= 0 && !String(lines[lastIndex] || '').trim()) {
        lastIndex -= 1;
      }
      if (lastIndex < 0) {
        inlineRecoveredFiles.forEach(function pushRecoveredInline(sharedFile) {
          outputSegments.push(sharedFileSegmentFromTrailingShorthand(sharedFile));
        });
        return;
      }

      const lastLine = String(lines[lastIndex] || '').trim();
      if (!/^\{[\s\S]*\}$/.test(lastLine)) {
        outputSegments.push(segment);
        return;
      }

      let sharedFile = null;
      try {
        sharedFile = sharedFilePayloadFromTrailingShorthand(JSON.parse(lastLine));
      } catch (_error) {
        sharedFile = null;
      }
      if (!sharedFile) {
        const normalizedText = cleanedContent.trim();
        if (normalizedText) {
          outputSegments.push({ ...segment, content: normalizedText });
        }
        inlineRecoveredFiles.forEach(function pushRecoveredInline(shared) {
          outputSegments.push(sharedFileSegmentFromTrailingShorthand(shared));
        });
        return;
      }

      const keptLines = cleanedInlineLines.slice(0, lastIndex);
      const textBefore = keptLines.join('\n').trim();
      if (textBefore) {
        outputSegments.push({ ...segment, content: textBefore });
      }
      inlineRecoveredFiles.forEach(function pushRecoveredInline(shared) {
        outputSegments.push(sharedFileSegmentFromTrailingShorthand(shared));
      });
      outputSegments.push(sharedFileSegmentFromTrailingShorthand(sharedFile));
    });

    return outputSegments;
  }

  // Timeline parsing.
  function parseMessageTimeline(rawText) {
    const source = String(rawText || '');
    const lowerSource = source.toLowerCase();
    const segments = [];
    const toolSegmentByAlias = {};
    let cursor = 0;

    // Strip control tokens that should never reach the visible transcript.
    function sanitizeVisibleText(value) {
      return String(value || '')
        .replace(/<\|start\|>\s*(assistant|user|system)?\s*(<\|channel\|>\s*(final|analysis|commentary))?\s*(<\|message\|>)?/gi, '')
        .replace(/<\|start\|>/gi, '')
        .replace(/<\|channel\|>\s*(final|analysis|commentary)/gi, '')
        .replace(/<\|message\|>/gi, '')
        .replace(/<\|return\|>/gi, '')
        .replace(/<\|startoftext\|>/gi, '')
        .replace(/<\|im_(start|end)\|>/gi, '')
        .replace(/<\|(assistant|user|system|endoftext)\|>/gi, '');
    }

    // Append a visible text segment if anything readable remains.
    function pushTextSegment(value) {
      const sanitizedValue = sanitizeVisibleText(value);
      if (!sanitizedValue || !sanitizedValue.trim()) {
        return;
      }

      segments.push({ type: 'text', content: sanitizedValue.trim() });
    }

    while (cursor < source.length) {
      const reasoningStart = findNextReasoningStart(source, cursor);
      const toolCallStart = lowerSource.indexOf('<tool_call>', cursor);
      const toolResultStart = lowerSource.indexOf('<tool_result>', cursor);
      const toolActivityStart = lowerSource.indexOf('<tool_activity>', cursor);
      const compressionStart = lowerSource.indexOf('<context_compression>', cursor);
      const candidates = [
        reasoningStart,
        toolCallStart !== -1 ? { pos: toolCallStart, kind: 'tool' } : null,
        toolResultStart !== -1 ? { pos: toolResultStart, kind: 'result' } : null,
        toolActivityStart !== -1 ? { pos: toolActivityStart, kind: 'activity' } : null,
        compressionStart !== -1 ? { pos: compressionStart, kind: 'compression' } : null
      ].filter(Boolean);

      if (candidates.length === 0) {
        pushTextSegment(source.substring(cursor));
        break;
      }

      candidates.sort(function sortCandidates(left, right) {
        return left.pos - right.pos;
      });

      const next = candidates[0];

      if (next.pos > cursor) {
        pushTextSegment(source.substring(cursor, next.pos));
      }

      if (next.kind === 'thought') {
        const contentStart = next.pos + next.pair.start.length;
        const thinkEnd = lowerSource.indexOf(next.pair.end, contentStart);
        if (thinkEnd === -1) {
          const content = sanitizeVisibleText(source.substring(contentStart)).trim();
          if (content) {
            segments.push({ type: 'thought', content });
          }
          break;
        }

        const content = sanitizeVisibleText(source.substring(contentStart, thinkEnd)).trim();
        if (content) {
          segments.push({ type: 'thought', content });
        }

        cursor = thinkEnd + next.pair.end.length;
        continue;
      }

      if (next.kind === 'tool') {
        const openTag = '<tool_call>';
        const closeTag = '</tool_call>';
        const toolEnd = lowerSource.indexOf(closeTag, next.pos + openTag.length);
        if (toolEnd === -1) {
          break;
        }

        const payload = source.substring(next.pos + openTag.length, toolEnd);

        try {
          const parsed = JSON.parse(payload);
          const alias = String(parsed.alias || '').trim();
          const segment = {
            type: 'tool',
            alias,
            serverId: String(parsed.server_id || '').trim(),
            serverName: String(parsed.server_name || '').trim(),
            toolId: String(parsed.tool_id || '').trim(),
            toolName: String(parsed.tool_name || parsed.tool_display_name || '').trim(),
            arguments: parsed.arguments && typeof parsed.arguments === 'object' ? parsed.arguments : {},
            result: null,
            toolUi: parsed.tool_ui && typeof parsed.tool_ui === 'object' ? parsed.tool_ui : null
          };

          segments.push(segment);

          if (alias) {
            toolSegmentByAlias[alias] = segment;
          }
        } catch (_error) {
          // Ignore malformed tool payloads.
        }

        cursor = toolEnd + closeTag.length;
        continue;
      }

      if (next.kind === 'compression') {
        const openTag = '<context_compression>';
        const closeTag = '</context_compression>';
        const compressionEnd = lowerSource.indexOf(closeTag, next.pos + openTag.length);
        if (compressionEnd === -1) {
          break;
        }

        const payload = source.substring(next.pos + openTag.length, compressionEnd);
        try {
          const parsed = JSON.parse(payload);
          segments.push({
            type: 'tool',
            alias: String(parsed.alias || 'context_compression_summary').trim(),
            serverId: String(parsed.server_id || 'system').trim(),
            serverName: String(parsed.server_name || 'System').trim(),
            toolId: String(parsed.tool_id || 'context_compression_summary').trim(),
            toolName: String(parsed.tool_name || parsed.tool_display_name || 'Context Compression').trim(),
            arguments: parsed.arguments && typeof parsed.arguments === 'object' ? parsed.arguments : {},
            result: String(parsed.content || ''),
            toolUi: parsed.tool_ui && typeof parsed.tool_ui === 'object' ? parsed.tool_ui : { kind: 'context_compression', status: 'done' },
            structuredContent: parsed.structured_content && typeof parsed.structured_content === 'object'
              ? parsed.structured_content
              : { kind: 'context_compression' }
          });
        } catch (_error) {
          // Ignore malformed compression payloads.
        }

        cursor = compressionEnd + closeTag.length;
        continue;
      }

      if (next.kind === 'activity') {
        const openTag = '<tool_activity>';
        const closeTag = '</tool_activity>';
        const activityEnd = lowerSource.indexOf(closeTag, next.pos + openTag.length);
        if (activityEnd === -1) {
          break;
        }

        const payload = source.substring(next.pos + openTag.length, activityEnd);
        try {
          const parsed = JSON.parse(payload);
          segments.push({ type: 'tool_activity', event: parsed });
        } catch (_error) {
          // Ignore malformed activity payloads.
        }

        cursor = activityEnd + closeTag.length;
        continue;
      }

      const resultOpenTag = '<tool_result>';
      const resultCloseTag = '</tool_result>';
      const resultEnd = lowerSource.indexOf(resultCloseTag, next.pos + resultOpenTag.length);
      if (resultEnd === -1) {
        break;
      }

      const payload = source.substring(next.pos + resultOpenTag.length, resultEnd);

      try {
        const parsed = JSON.parse(payload);
        const alias = String(parsed.alias || '').trim();
        let target = toolSegmentByAlias[alias];

        if (!target && alias) {
          target = {
            type: 'tool',
            alias,
            serverId: String(parsed.server_id || '').trim(),
            serverName: String(parsed.server_name || '').trim(),
            toolId: String(parsed.tool_id || '').trim(),
            toolName: String(parsed.tool_name || parsed.tool_display_name || '').trim(),
            arguments: parsed.arguments && typeof parsed.arguments === 'object' ? parsed.arguments : {},
            result: null
          };
          segments.push(target);
          toolSegmentByAlias[alias] = target;
        }

        if (target) {
          target.result = String(parsed.content || '');
          if (parsed.tool_ui && typeof parsed.tool_ui === 'object') {
            target.toolUi = parsed.tool_ui;
          }
          if (parsed.structured_content && typeof parsed.structured_content === 'object') {
            target.structuredContent = parsed.structured_content;
          }
        }
      } catch (_error) {
        // Ignore malformed tool results.
      }

      cursor = resultEnd + resultCloseTag.length;
    }

    const normalizedSegments = normalizeTrailingSharedFileShorthandSegments(segments);
    const visibleText = normalizedSegments
      .filter(function onlyText(segment) { return segment.type === 'text'; })
      .map(function mapText(segment) { return segment.content; })
      .join('\n\n')
      .trim();

    return { segments: normalizedSegments, visibleText };
  }

  // Thought state helpers.
  // Read the set of expanded thought indices for one row.

  // Per-message expansion state.
  function getExpandedThoughtIndices($msgRow) {
    const rawValue = String($msgRow.attr('data-expanded-thoughts') || '').trim();
    if (!rawValue) {
      return new Set();
    }

    return new Set(
      rawValue
        .split(',')
        .map(function toNumber(value) { return parseInt(value, 10); })
        .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
    );
  }

  // Persist the expanded thought set back to the row element.
  function setExpandedThoughtIndices($msgRow, expandedIndices) {
    const normalized = Array.from(expandedIndices)
      .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
      .sort(function sortValues(left, right) { return left - right; });

    if (normalized.length === 0) {
      $msgRow.removeAttr('data-expanded-thoughts');
      return;
    }

    $msgRow.attr('data-expanded-thoughts', normalized.join(','));
  }


  // Read the set of expanded search cards for one row.
  function getExpandedSearchIndices($msgRow) {
    const rawValue = String($msgRow.attr('data-expanded-searches') || '').trim();
    if (!rawValue) {
      return new Set();
    }

    return new Set(
      rawValue
        .split(',')
        .map(function toNumber(value) { return parseInt(value, 10); })
        .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
    );
  }

  // Persist expanded search cards back to the row element.
  function setExpandedSearchIndices($msgRow, expandedIndices) {
    const normalized = Array.from(expandedIndices)
      .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
      .sort(function sortValues(left, right) { return left - right; });

    if (normalized.length === 0) {
      $msgRow.removeAttr('data-expanded-searches');
      return;
    }

    $msgRow.attr('data-expanded-searches', normalized.join(','));
  }

  // Read search cards expanded by stable tool keys. This survives stream
  // frames where the numeric tool index can shift while the tool alias stays.
  function getExpandedSearchKeys($msgRow) {
    const existing = $msgRow.data('expandedSearchKeys');
    if (existing instanceof Set) {
      return new Set(existing);
    }

    const rawValue = String($msgRow.attr('data-expanded-search-keys') || '').trim();
    if (!rawValue) {
      return new Set();
    }

    return new Set(
      rawValue
        .split(',')
        .map(function decodeKey(value) {
          try {
            return decodeURIComponent(value);
          } catch (_error) {
            return '';
          }
        })
        .filter(Boolean)
    );
  }

  // Persist expanded search keys on the row and in jQuery data.
  function setExpandedSearchKeys($msgRow, expandedKeys) {
    const normalized = Array.from(expandedKeys)
      .map(function normalizeKey(value) { return String(value || '').trim(); })
      .filter(Boolean)
      .sort();

    const keySet = new Set(normalized);
    $msgRow.data('expandedSearchKeys', keySet);

    if (normalized.length === 0) {
      $msgRow.removeAttr('data-expanded-search-keys');
      return;
    }

    $msgRow.attr('data-expanded-search-keys', normalized.map(encodeURIComponent).join(','));
  }


  // Read the set of expanded write cards for one row.
  function getExpandedWriteIndices($msgRow) {
    const rawValue = String($msgRow.attr('data-expanded-writes') || '').trim();
    if (!rawValue) {
      return new Set();
    }

    return new Set(
      rawValue
        .split(',')
        .map(function toNumber(value) { return parseInt(value, 10); })
        .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
    );
  }

  // Persist expanded write cards back to the row element.
  function setExpandedWriteIndices($msgRow, expandedIndices) {
    const normalized = Array.from(expandedIndices)
      .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
      .sort(function sortValues(left, right) { return left - right; });

    if (normalized.length === 0) {
      $msgRow.removeAttr('data-expanded-writes');
      return;
    }

    $msgRow.attr('data-expanded-writes', normalized.join(','));
  }


  // Read the set of expanded edit cards for one row.
  function getExpandedEditIndices($msgRow) {
    const rawValue = String($msgRow.attr('data-expanded-edits') || '').trim();
    if (!rawValue) {
      return new Set();
    }

    return new Set(
      rawValue
        .split(',')
        .map(function toNumber(value) { return parseInt(value, 10); })
        .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
    );
  }

  // Persist expanded edit cards back to the row element.
  function setExpandedEditIndices($msgRow, expandedIndices) {
    const normalized = Array.from(expandedIndices)
      .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
      .sort(function sortValues(left, right) { return left - right; });

    if (normalized.length === 0) {
      $msgRow.removeAttr('data-expanded-edits');
      return;
    }

    $msgRow.attr('data-expanded-edits', normalized.join(','));
  }


  // Read the set of expanded sandbox cards for one row.
  function getExpandedSandboxIndices($msgRow) {
    const rawValue = String($msgRow.attr('data-expanded-sandboxes') || '').trim();
    if (!rawValue) {
      return new Set();
    }

    return new Set(
      rawValue
        .split(',')
        .map(function toNumber(value) { return parseInt(value, 10); })
        .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
    );
  }

  // Persist expanded sandbox cards back to the row element.
  function setExpandedSandboxIndices($msgRow, expandedIndices) {
    const normalized = Array.from(expandedIndices)
      .filter(function isValid(value) { return Number.isInteger(value) && value >= 0; })
      .sort(function sortValues(left, right) { return left - right; });

    if (normalized.length === 0) {
      $msgRow.removeAttr('data-expanded-sandboxes');
      return;
    }

    $msgRow.attr('data-expanded-sandboxes', normalized.join(','));
  }


  // Activity timeline rendering.
  // Render thoughts, tool calls, and visible text into the assistant timeline.

  // Inline image tokens ([[images: query]]) and image_search galleries.
  // A token written by the model on its own line becomes a native photo row; the
  // row is fetched lazily from the image-search endpoint and cached per query.
  const IMAGE_TOKEN_RE = /^\[\[images:[ \t]+([^\n]{2,140}?)[ \t]*\]\][ \t]*$/gm;
  const imageTokenCache = new Map(); // query -> Promise<images array>

  function extractImageTokenBlocks(content) {
    const blocks = [];
    const text = String(content || '').replace(IMAGE_TOKEN_RE, function replacer(_match, query) {
      const index = blocks.length;
      blocks.push(String(query || '').trim());
      return `%%ASLM-IMG${index}%%`;
    });
    return { text, blocks };
  }

  function restoreImagePlaceholders(html, blocks) {
    if (!Array.isArray(blocks) || blocks.length === 0) {
      return html;
    }
    let restored = html;
    blocks.forEach(function restoreOne(query, index) {
      const marker = `%%ASLM-IMG${index}%%`;
      // The marker may survive inside a <p> block; replace any occurrence with the
      // placeholder node the hydrator upgrades into a photo row.
      const node = `<div class="msg-image-token" data-image-query="${escapeAttributeValue(query)}"></div>`;
      restored = restored.split(marker).join(node);
    });
    return restored;
  }

  function imageTokenLoadingHtml() {
    // Show nothing while the row resolves: the token stays blank until the
    // gallery (or an explicit error hint) replaces it.
    return '';
  }

  function renderImageRowHtml(images, query) {
    const items = (images || []).slice(0, 12).map(function oneThumb(image) {
      const thumb = String((image && (image.thumbnail_url || image.image_url)) || '');
      if (!thumb) {
        return '';
      }
      const page = String((image && image.page_url) || '');
      const title = String((image && image.title) || query || '');
      const openAttrs = page
        ? ` href="${escapeAttributeValue(page)}" target="_blank" rel="noopener noreferrer"`
        : '';
      return `
        <a class="msg-image-thumb"${openAttrs} title="${escapeAttributeValue(title)}">
          <img src="${escapeAttributeValue(thumb)}" alt="${escapeAttributeValue(title)}" loading="lazy" />
        </a>
      `;
    }).filter(Boolean).join('');
    if (!items) {
      return '';
    }
    return `<div class="msg-image-row">${items}</div>`;
  }

  // Resolved rows live briefly in sessionStorage so reopening the same chat
  // does not re-run the search engines; errors are never cached.
  const IMAGE_ROW_CACHE_PREFIX = 'aslm-image-row:';
  const IMAGE_ROW_CACHE_TTL_MS = 30 * 60 * 1000;

  function readStoredImageRow(key) {
    try {
      const raw = window.sessionStorage.getItem(IMAGE_ROW_CACHE_PREFIX + key);
      if (!raw) {
        return null;
      }
      const entry = JSON.parse(raw);
      if (!entry || !Array.isArray(entry.images) || Date.now() - entry.at > IMAGE_ROW_CACHE_TTL_MS) {
        window.sessionStorage.removeItem(IMAGE_ROW_CACHE_PREFIX + key);
        return null;
      }
      return entry.images;
    } catch (_error) {
      return null;
    }
  }

  function storeImageRow(key, images) {
    try {
      window.sessionStorage.setItem(
        IMAGE_ROW_CACHE_PREFIX + key,
        JSON.stringify({ at: Date.now(), images: images })
      );
    } catch (_error) {
      // Full or unavailable sessionStorage — the in-memory cache still applies.
    }
  }

  function fetchImageRow(query) {
    const key = String(query || '').trim();
    if (!key) {
      return Promise.resolve([]);
    }
    if (imageTokenCache.has(key)) {
      return imageTokenCache.get(key);
    }
    const stored = readStoredImageRow(key);
    if (stored) {
      const cachedRequest = Promise.resolve(stored);
      imageTokenCache.set(key, cachedRequest);
      return cachedRequest;
    }
    const request = postJson('/api/image-search/', {
      query: key
    })
      .then(function onImages(data) {
        if (data && data.ok === false && data.error) {
          // Surface the server's reason (e.g. a non-vision model) in the row.
          return [{ error: String(data.error) }];
        }
        const images = (data && Array.isArray(data.images) && data.images) || [];
        if (images.length) {
          storeImageRow(key, images);
        }
        return images;
      })
      .catch(function onImageError(error) {
        // requestJson throws on non-2xx with the server's message preserved.
        const message = String((error && error.message) || '');
        return message ? [{ error: message }] : [];
      });
    imageTokenCache.set(key, request);
    return request;
  }

  function resolveImageToken($token, query) {
    $token.attr('data-image-state', 'loading');
    $token.html(imageTokenLoadingHtml());
    fetchImageRow(query).then(function fillToken(images) {
      const errorHint = images && images.length === 1 && images[0] && images[0].error
        ? String(images[0].error)
        : '';
      const html = errorHint ? '' : renderImageRowHtml(images, query);
      $token.attr('data-image-state', html ? 'done' : 'empty');
      // Found rows render the gallery; explicit errors show their reason;
      // an empty result collapses to nothing.
      $token.html(
        html
        || (errorHint ? `<span class="msg-image-row-hint is-empty">${escHtml(errorHint)}</span>` : '')
      );
    });
  }

  function hydrateImageTokens($root) {
    $root.find('.msg-image-token:not([data-image-state])').each(function hydrateToken() {
      const $token = window.jQuery(this);
      const query = String($token.attr('data-image-query') || '').trim();
      if (!query) {
        $token.attr('data-image-state', 'empty');
        return;
      }
      resolveImageToken($token, query);
    });
  }

  function isImageSearchToolSegment(segment) {
    const toolId = String(segment.toolId || '').toLowerCase();
    const alias = String(segment.alias || '').toLowerCase();
    return !!(segment && segment.toolUi && segment.toolUi.kind === 'image_search')
      || toolId === 'image_search'
      || alias.endsWith('__image_search');
  }

  function imageSearchImagesFromSegment(segment) {
    const candidates = [
      segment && segment.toolUi ? segment.toolUi.images : null,
      segment && segment.structuredContent ? segment.structuredContent.images : null,
      segment && segment.structuredContent && segment.structuredContent.ui
        ? segment.structuredContent.ui.images : null
    ];
    for (let index = 0; index < candidates.length; index += 1) {
      if (Array.isArray(candidates[index]) && candidates[index].length > 0) {
        return candidates[index];
      }
    }
    return [];
  }

  // Search and read-page tool rendering.
  function isSearchToolSegment(segment) {
    const toolId = String(segment.toolId || '').toLowerCase();
    const alias = String(segment.alias || '').toLowerCase();
    return toolId === 'web_search'
      || toolId === 'web_search_rich'
      || alias.endsWith('__web_search')
      || alias.endsWith('__web_search_rich')
      || isImageSearchToolSegment(segment)
      || !!(segment.toolUi && segment.toolUi.compact);
  }

  // Source-bearing orchestrator tools (for example Deep Research) must feed
  // the same citation registry even though they are not rendered as search cards.
  function hasCitationSources(segment) {
    return !!(
      segment
      && segment.type === 'tool'
      && searchSourcesFromSegment(segment).some(function hasCitableSource(source) {
        return !!String((source && (source.id || source.url)) || '').trim();
      })
    );
  }

  // Read the backend-normalized request attached during tool preflight or execution.
  function normalizedSearchRequestFromSegment(segment) {
    const candidates = [
      segment && segment.toolUi && segment.toolUi.search_request,
      segment && segment.structuredContent && segment.structuredContent.search_request,
      segment && segment.structuredContent && segment.structuredContent.ui
        && segment.structuredContent.ui.search_request
    ];
    return candidates.find(function findRequest(candidate) {
      return candidate && typeof candidate === 'object' && Array.isArray(candidate.queries);
    }) || null;
  }

  // Return one display row per normalized query; legacy raw data is only a fallback.
  function searchQueryEntriesFromSegment(segment) {
    const request = normalizedSearchRequestFromSegment(segment);
    if (request) {
      return request.queries.map(function normalizePreparedQuery(item) {
        const query = item && typeof item === 'object' ? item : {};
        return {
          query: String(query.compiled_query || '').replace(/\s+/g, ' ').trim(),
          vertical: String(query.vertical || 'web').trim().toLowerCase()
        };
      }).filter(function keepPreparedQuery(item) { return !!item.query; });
    }

    const args = segment && segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const raw = args.query || args.q || '';
    if (Array.isArray(raw)) {
      return raw.map(function normalizeLegacyQuery(value) {
        return { query: formatSearchQueryValue(value).trim(), vertical: '' };
      }).filter(function keepLegacyQuery(item) { return !!item.query; });
    }
    const query = formatSearchQueryValue(raw).trim();
    if (query) {
      return [{ query, vertical: '' }];
    }

    const rejectedArguments = segment && segment.toolUi
      && segment.toolUi.rejected_arguments
      && typeof segment.toolUi.rejected_arguments === 'object'
      ? segment.toolUi.rejected_arguments
      : null;
    if (!rejectedArguments) {
      return [];
    }

    return ['web', 'academic', 'shopping', 'onion'].flatMap(function mapRejectedVertical(vertical) {
      const parts = [];
      collectRejectedSearchStrings(rejectedArguments[vertical], parts);
      const rejectedQuery = parts.join(' | ').replace(/\s+/g, ' ').trim();
      return rejectedQuery ? [{ query: rejectedQuery, vertical }] : [];
    });
  }

  // Recover the model's original query text from a rejected call so the UI can
  // identify the malformed request instead of falling back to "Bad query: query".
  function collectRejectedSearchStrings(value, output) {
    if (typeof value === 'string') {
      const text = value.replace(/\s+/g, ' ').trim();
      if (text) {
        output.push(text);
      }
      return;
    }
    if (Array.isArray(value)) {
      value.forEach(function collectRejectedArrayItem(item) {
        collectRejectedSearchStrings(item, output);
      });
      return;
    }
    if (!value || typeof value !== 'object') {
      return;
    }
    ['text', 'query', 'q', 'raw_query', 'item', 'items'].forEach(function collectRejectedKey(key) {
      if (Object.prototype.hasOwnProperty.call(value, key)) {
        collectRejectedSearchStrings(value[key], output);
      }
    });
  }

  // Prefer the backend-normalized activity description over raw model arguments.
  function searchDescriptionFromSegment(segment) {
    const request = normalizedSearchRequestFromSegment(segment);
    const toolUi = segment && segment.toolUi && typeof segment.toolUi === 'object'
      ? segment.toolUi
      : {};
    const args = segment && segment.arguments && typeof segment.arguments === 'object'
      ? segment.arguments
      : {};
    return String((request && request.description) || toolUi.description || args.description || '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  // Instant-mode search activity deliberately exposes descriptions, never query text.
  function isInstantSearchFlowSegment(segment) {
    const toolUi = segment && segment.toolUi && typeof segment.toolUi === 'object' ? segment.toolUi : {};
    const structured = segment && segment.structuredContent && typeof segment.structuredContent === 'object'
      ? segment.structuredContent
      : {};
    const structuredUi = structured.ui && typeof structured.ui === 'object' ? structured.ui : {};
    const request = normalizedSearchRequestFromSegment(segment);
    const args = segment && segment.arguments && typeof segment.arguments === 'object'
      ? segment.arguments
      : {};
    const compactInstantShape = Object.prototype.hasOwnProperty.call(args, 'query')
      && Object.prototype.hasOwnProperty.call(args, 'description')
      && !['web', 'shopping', 'academic', 'onion', 'effort', 'call_description'].some(function hasAdvancedField(key) {
        return Object.prototype.hasOwnProperty.call(args, key);
      });
    return toolUi.instant_mode === true
      || structuredUi.instant_mode === true
      || !!(request && request.schema_mode === 'instant')
      || compactInstantShape;
  }

  // Extract a stable query summary from one tool segment.
  function searchQueryFromSegment(segment) {
    return searchQueryEntriesFromSegment(segment).map(function mapEntry(item) {
      return item.query;
    }).join(' | ');
  }

  // Append one query fragment without duplicates.
  function appendUniqueSearchPart(parts, value) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    if (text && !parts.includes(text)) {
      parts.push(text);
    }
  }

  // Flatten list-like search arguments into query parts.
  function appendSearchList(parts, value) {
    if (!Array.isArray(value)) {
      return;
    }
    value.forEach(function appendItem(item) {
      appendUniqueSearchPart(parts, item);
    });
  }

  // Format one search argument value for display.
  function formatSearchQueryValue(value) {
    if (value === null || value === undefined) {
      return '';
    }
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      const text = String(value).replace(/\s+/g, ' ').trim();
      if (text && (text[0] === '{' || text[0] === '[')) {
        try {
          const parsed = JSON.parse(text);
          return formatSearchQueryValue(parsed);
        } catch (_error) {
          // Fall through to plain text for malformed partial JSON from streaming.
        }
      }
      return text;
    }
    if (Array.isArray(value)) {
      return value.map(formatSearchQueryValue).filter(Boolean).join(', ');
    }
    if (typeof value !== 'object') {
      return String(value || '').replace(/\s+/g, ' ').trim();
    }

    const plan = value.query && typeof value.query === 'object' && !Array.isArray(value.query)
      ? value.query
      : value;
    const parts = [];
    ['entities', 'model_identifiers', 'terms'].forEach(function appendKey(key) {
      appendSearchList(parts, plan[key]);
    });
    appendSearchList(parts, (plan.exact_phrases || []).map(function quotePhrase(phrase) {
      return `"${String(phrase || '').replace(/\s+/g, ' ').trim()}"`;
    }));
    appendSearchList(parts, (plan.required_terms || []).map(function requireTerm(term) {
      const text = String(term || '').replace(/\s+/g, ' ').trim();
      return text ? `+${text}` : '';
    }));
    if (plan.intent && plan.intent !== 'general') {
      appendUniqueSearchPart(parts, plan.intent);
    }
    appendUniqueSearchPart(parts, plan.region);

    const numericMin = parseInt(plan.numeric_min ?? plan.price_min ?? 0, 10) || 0;
    const numericMax = parseInt(plan.numeric_max ?? plan.price_max ?? 0, 10) || 0;
    if (numericMin && numericMax) {
      appendUniqueSearchPart(parts, `${numericMin}-${numericMax}`);
    } else if (numericMin) {
      appendUniqueSearchPart(parts, numericMin);
    } else if (numericMax) {
      appendUniqueSearchPart(parts, numericMax);
    }
    appendUniqueSearchPart(parts, plan.unit || plan.currency);

    if (plan.year) {
      appendUniqueSearchPart(parts, plan.year);
    }
    appendSearchList(parts, (plan.site_include || []).map(function includeDomain(domain) {
      return `site:${String(domain || '').replace(/^site:/i, '').trim()}`;
    }));
    appendSearchList(parts, (plan.site_exclude || []).map(function excludeDomain(domain) {
      return `-site:${String(domain || '').replace(/^-?site:/i, '').trim()}`;
    }));
    appendSearchList(parts, (plan.excluded_terms || []).map(function excludeTerm(term) {
      const text = String(term || '').replace(/\s+/g, ' ').trim();
      return text ? `-${text}` : '';
    }));

    if (parts.length === 0 && plan.raw_query) {
      appendUniqueSearchPart(parts, plan.raw_query);
    }
    return parts.join(' ').replace(/\s+/g, ' ').trim();
  }

  // Build a stable key for one search tool segment.
  function searchSegmentKey(segment, fallbackIndex) {
    const safeSegment = segment && typeof segment === 'object' ? segment : {};
    const alias = String(safeSegment.alias || '').trim();
    if (alias) {
      return `alias:${alias}`;
    }

    const query = searchQueryFromSegment(safeSegment).toLowerCase();
    const toolId = String(safeSegment.toolId || safeSegment.toolName || '').trim().toLowerCase();
    if (query) {
      return `query:${toolId}:${query}`;
    }

    if (Number.isInteger(fallbackIndex) && fallbackIndex >= 0) {
      return `index:${fallbackIndex}`;
    }

    return '';
  }

  // Report whether one segment is a read-page tool call.
  function isReadPageToolSegment(segment) {
    const toolId = String(segment.toolId || '').toLowerCase();
    const alias = String(segment.alias || '').toLowerCase();
    const uiKind = segment.toolUi && segment.toolUi.kind ? String(segment.toolUi.kind).toLowerCase() : '';
    return toolId === 'read_page'
      || alias.endsWith('__read_page')
      || uiKind === 'read_page';
  }

  // Report whether one segment is a write-file tool call.
  function isWriteToolSegment(segment) {
    const toolId = String(segment.toolId || '').toLowerCase();
    const alias = String(segment.alias || '').toLowerCase();
    const toolName = String(segment.toolName || '').toLowerCase();
    return toolId === 'write'
      || alias.endsWith('__write')
      || toolName === 'write'
      || toolName === 'write file';
  }

  // Report whether one segment is an edit-file tool call.
  function isEditToolSegment(segment) {
    const toolId = String(segment.toolId || '').toLowerCase();
    const alias = String(segment.alias || '').toLowerCase();
    const toolName = String(segment.toolName || '').toLowerCase();
    return toolId === 'edit'
      || alias.endsWith('__edit')
      || toolName === 'edit'
      || toolName === 'edit file';
  }

  // Build a minimal citation source record from one URL.
  function sourceFromUrl(value, rank) {
    const rawUrl = String(value || '').trim();
    if (!rawUrl) {
      return null;
    }

    let domain = '';
    try {
      domain = new URL(rawUrl).hostname.replace(/^www\./i, '');
    } catch (_error) {
      domain = rawUrl.replace(/^https?:\/\//i, '').split('/')[0].replace(/^www\./i, '');
    }

    const parts = domain.split('.').filter(Boolean);
    const label = parts.length >= 2 ? parts[parts.length - 2] : (parts[0] || domain);
    const displayDomain = label.replace(/-/g, ' ').replace(/\b\w/g, function titleCase(letter) {
      return letter.toUpperCase();
    });

    return {
      rank: rank || 0,
      url: rawUrl,
      domain,
      display_domain: displayDomain || domain,
      favicon_url: localFaviconUrlForDomain(domain),
    };
  }

  // Normalize one search result into a chip record.
  function normalizeSearchSourceItem(item, rank) {
    if (typeof item === 'string') {
      return sourceFromUrl(item, rank);
    }

    if (!item || typeof item !== 'object') {
      return null;
    }

    const rawUrl = String(item.url || item.link || item.href || item.source_url || '').trim();
    const fromUrl = rawUrl ? sourceFromUrl(rawUrl, rank) : null;
    const domain = String(
      item.domain
      || item.display_domain
      || item.host
      || (fromUrl ? fromUrl.domain : '')
      || ''
    ).trim();

    if (!rawUrl && !domain) {
      return null;
    }

    return {
      ...(fromUrl || {}),
      ...item,
      rank: item.rank || rank || (fromUrl ? fromUrl.rank : 0),
      url: rawUrl || item.url || (fromUrl ? fromUrl.url : ''),
      domain: domain || (fromUrl ? fromUrl.domain : ''),
      display_domain: String(item.display_domain || item.displayDomain || '').trim()
        || (fromUrl ? fromUrl.display_domain : domain),
      favicon_url: item.favicon_url || item.faviconUrl || (fromUrl ? fromUrl.favicon_url : ''),
    };
  }

  // Flatten nested containers into source candidate arrays.
  function collectSourceCandidates(container) {
    if (Array.isArray(container)) {
      return container;
    }

    if (!container || typeof container !== 'object') {
      return [];
    }

    return [
      container.sources,
      container.source_chips,
      container.sourceChips,
      container.results,
      container.items,
      container.documents,
      container.data
    ]
      .filter(Array.isArray)
      .flat();
  }

  // Extract search result sources from one tool segment.
  function searchSourcesFromSegment(segment) {
    const structured = segment.structuredContent && typeof segment.structuredContent === 'object'
      ? segment.structuredContent
      : null;
    const toolUi = segment.toolUi && typeof segment.toolUi === 'object' ? segment.toolUi : null;
    const compact = segment.toolUi && segment.toolUi.compact && typeof segment.toolUi.compact === 'object'
      ? segment.toolUi.compact
      : null;
    const resultObject = parseToolResultObject(segment);

    return dedupeSearchSources([
      ...collectSourceCandidates(structured),
      ...collectSourceCandidates(toolUi),
      ...collectSourceCandidates(compact),
      ...collectSourceCandidates(resultObject)
    ].map(normalizeSearchSourceItem));
  }

  // Extract read-page sources from one tool segment.
  function readPageSourcesFromSegment(segment) {
    const structured = segment.structuredContent && typeof segment.structuredContent === 'object'
      ? segment.structuredContent
      : null;
    const uiSources = segment.toolUi && Array.isArray(segment.toolUi.sources) ? segment.toolUi.sources : [];
    if (uiSources.length > 0) {
      return uiSources;
    }
    if (structured && Array.isArray(structured.sources) && structured.sources.length > 0) {
      return structured.sources;
    }

    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const rawUrls = Array.isArray(args.url) ? args.url : [args.url || args.urls];
    return rawUrls
      .flatMap(function flattenUrls(item) { return Array.isArray(item) ? item : [item]; })
      .map(function mapUrl(item, index) { return sourceFromUrl(item, index + 1); })
      .filter(Boolean);
  }

  // Register search sources from one segment.
  function addSearchSourcesToCitationRegistry(registry, segment) {
    addSegmentCitationSources(registry, segment);
  }

  // Register search sources from every segment.
  function addAllSearchSourcesToCitationRegistry(registry, segments) {
    addSegmentsCitationSources(
      registry,
      (Array.isArray(segments) ? segments : []).filter(function onlySourceBearingToolSegment(segment) {
        return hasCitationSources(segment);
      })
    );
  }

  // Render one inline search source chip.
  function renderSourceChip(chip) {
    const source = chip && typeof chip === 'object' ? chip : {};
    const domain = String(source.display_domain || source.domain || '').trim();
    if (!domain) {
      return '';
    }
    const faviconUrl = sourceFaviconUrl(source);
    const sourceUrl = String(source.url || '').trim();
    const fallbackLetter = domain.charAt(0).toUpperCase();
    const imgHtml = faviconUrl
      ? `<img class="msg-search-chip-favicon" src="${escapeAttributeValue(faviconUrl)}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='inline-flex';">`
      : '';
    const fallbackStyle = domainAccentStyle(domain, faviconUrl ? 'display:none;' : '');
    const tagName = sourceUrl ? 'a' : 'span';
    const linkAttrs = sourceUrl
      ? ` href="${escapeAttributeValue(sourceUrl)}" target="_blank" rel="noopener noreferrer"`
      : '';
    return `
      <${tagName} class="msg-search-chip" title="${escapeAttributeValue(source.title || domain)}"${linkAttrs}>
        ${imgHtml}<span class="msg-search-chip-fallback" style="${escapeAttributeValue(fallbackStyle)}">${escHtml(fallbackLetter)}</span>
        <span class="msg-search-chip-domain">${escHtml(domain)}</span>
      </${tagName}>
    `;
  }

  // Remove duplicate search sources by URL and id.
  function dedupeSearchSources(sources) {
    const seen = {};
    return (Array.isArray(sources) ? sources : []).filter(function keepFirstSource(source) {
      if (!source || typeof source !== 'object') {
        return false;
      }

      const key = String(source.url || source.source_id || source.id || source.domain || source.display_domain || '').trim().toLowerCase();
      if (!key || seen[key]) {
        return false;
      }

      seen[key] = true;
      return true;
    });
  }

  // Collect every source attached to one assistant response, including direct
  // search results and citation blocks embedded in tool output.
  function messageSourcesFromSegments(segments, citationRegistry) {
    const safeSegments = Array.isArray(segments) ? segments : [];
    const directSources = [];
    safeSegments.forEach(function collectMessageSource(segment) {
      if (!segment || segment.type !== 'tool') {
        return;
      }
      directSources.push(...searchSourcesFromSegment(segment));
      if (isReadPageToolSegment(segment)) {
        directSources.push(...readPageSourcesFromSegment(segment));
      }
    });

    const messageSourceRegistry = createCitationRegistry();
    addSegmentsCitationSources(
      messageSourceRegistry,
      safeSegments.filter(function onlyToolSourceSegments(segment) {
        return segment && segment.type === 'tool';
      })
    );
    const registeredSources = [
      ...(citationRegistry && typeof citationRegistry === 'object' ? Object.values(citationRegistry) : []),
      ...Object.values(messageSourceRegistry)
    ];
    return dedupeSearchSources(
      [...directSources, ...registeredSources]
        .map(function normalizeMessageSource(source, index) {
          return normalizeSearchSourceItem(source, index + 1);
        })
        .filter(Boolean)
    );
  }

  // Render the compact favicon stack shown inside the Sources action button.
  function renderMessageSourcesButtonContents(sources) {
    const label = t('sources.title', null, 'Sources');
    const iconsHtml = (Array.isArray(sources) ? sources : []).slice(0, 2).map(function renderButtonSource(source) {
      const domain = String(source.display_domain || source.domain || source.url || '').trim();
      const faviconUrl = sourceFaviconUrl(source);
      const fallbackLetter = (domain.charAt(0) || '?').toUpperCase();
      const imgHtml = faviconUrl
        ? `<img class="msg-sources-btn-favicon" src="${escapeAttributeValue(faviconUrl)}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='inline-flex';">`
        : '';
      const fallbackStyle = domainAccentStyle(domain, faviconUrl ? 'display:none;' : '');
      return `<span class="msg-sources-btn-icon">${imgHtml}<span class="msg-sources-btn-fallback" style="${escapeAttributeValue(fallbackStyle)}">${escHtml(fallbackLetter)}</span></span>`;
    }).join('');

    return `<span class="msg-sources-btn-icons" aria-hidden="true">${iconsHtml || icons.GLOBE_ICON}</span><span class="msg-sources-btn-label">${escHtml(label)}</span>`;
  }

  // Refresh one row's Sources action after timeline rendering or after the
  // action bar is attached at the end of a streamed response.
  function syncMessageSourcesButton($msgRow) {
    if (!$msgRow || !$msgRow.length) {
      return;
    }
    const sources = $msgRow.data('messageSources') || [];
    const hasSources = Array.isArray(sources) && sources.length > 0;
    const $button = $msgRow.find('.msg-sources-btn').first();
    if ($button.length) {
      const label = t('sources.title', null, 'Sources');
      $button
        .prop('hidden', !hasSources)
        .attr('title', label)
        .attr('aria-label', label)
        .html(hasSources ? renderMessageSourcesButtonContents(sources) : '');
    }

    if ($activeSourcesMessageRow && $activeSourcesMessageRow[0] === $msgRow[0]) {
      if (hasSources) {
        syncSourcesDrawer($msgRow);
      } else {
        closeSourcesDrawer();
      }
    }
  }

  function setMessageSources($msgRow, sources) {
    $msgRow.data('messageSources', Array.isArray(sources) ? sources : []);
    syncMessageSourcesButton($msgRow);
  }

  // Render source chips with a More overflow control.
  function renderSearchSourcesWithOverflow(sources, maxVisible) {
    const items = Array.isArray(sources) ? sources : [];
    const visibleLimit = Number.isInteger(maxVisible) && maxVisible > 0 ? maxVisible : 3;
    const rendered = items.map(function mapRenderedSource(source) {
      return {
        source,
        html: renderSourceChip(source)
      };
    }).filter(function filterRenderableSource(item) {
      return !!String(item.html || '').trim();
    });

    return {
      visibleHtml: rendered.slice(0, visibleLimit).map(function mapVisible(item) { return item.html; }).join(''),
      hiddenHtml: rendered.slice(visibleLimit).map(function mapHidden(item) { return item.html; }).join(''),
      hiddenCount: Math.max(0, rendered.length - visibleLimit)
    };
  }

  // Render a canonical request produced by backend preflight without reparsing its schema.
  function renderPreparedSearchToolCard(segment, toolSegmentIndex, options, entries) {
    const renderOptions = options || {};
    const hasResult = segment.result !== null && segment.result !== undefined;
    const status = segment.toolUi && segment.toolUi.status ? String(segment.toolUi.status) : '';
    const isRejected = status === 'rejected' || /^INVALID_SEARCH_PLAN:/i.test(String(segment.result || '').trim());
    const sources = searchSourcesFromSegment(segment);
    const renderedSources = renderSearchSourcesWithOverflow(sources, 3);
    const isExpanded = !!renderOptions.expanded;
    const instantMode = isInstantSearchFlowSegment(segment);
    const displayEntries = instantMode
      ? [{ query: searchDescriptionFromSegment(segment) || (hasResult ? 'Searched sources' : 'Searching sources') }]
      : entries;
    const queriesHtml = displayEntries.map(function renderPreparedQuery(entry) {
      const pendingDot = hasResult || isRejected ? '' : '<span class="msg-search-pending-dot"></span>';
      return `<div class="msg-search-batch-query${isRejected ? ' is-error' : ''}"><span class="msg-search-batch-query-text">${escHtml(entry.query)}</span>${pendingDot}</div>`;
    }).join('');
    const moreCount = renderedSources.hiddenCount;
    const moreButtonAttrs = `type="button" data-search-more-count="${moreCount}" aria-expanded="${isExpanded ? 'true' : 'false'}"`;
    const collapsedMoreHtml = moreCount > 0
      ? `<button class="msg-search-chip msg-search-chip--more msg-search-chip--more-collapsed" ${moreButtonAttrs}><span class="msg-search-chip-domain">${escHtml(`${MORE_LABEL} ${moreCount}`)}</span></button>`
      : '';
    const expandedMoreHtml = moreCount > 0
      ? `<button class="msg-search-chip msg-search-chip--more msg-search-chip--more-expanded" ${moreButtonAttrs}><span class="msg-search-chip-domain">${escHtml(HIDE_LABEL)}</span></button>`
      : '';
    const iconHtml = instantMode
      ? (icons.TOOL_SEARCH_ICON || icons.WEB_SEARCH_ICON || icons.GLOBE_ICON)
      : (icons.WEB_SEARCH_ICON || icons.TOOL_SEARCH_ICON || icons.GLOBE_ICON);
    const searchKey = searchSegmentKey(segment, toolSegmentIndex);
    const searchKeyAttr = searchKey ? ` data-search-key="${escapeAttributeValue(searchKey)}"` : '';

    return `
      <div class="msg-search-card msg-search-card--batch${hasResult ? ' is-done' : ' is-pending'}${isRejected ? ' is-error' : ''}${isExpanded ? ' is-expanded' : ''}" data-tool-segment-index="${toolSegmentIndex}"${searchKeyAttr}>
        <div class="msg-search-batch-head">
          <span class="msg-search-icon msg-search-batch-icon">${iconHtml}</span>
          <div class="msg-search-batch-queries">${queriesHtml}</div>
        </div>
        ${renderedSources.visibleHtml || collapsedMoreHtml ? `<div class="msg-search-chips msg-search-chips--batch">${renderedSources.visibleHtml}${collapsedMoreHtml}</div>` : ''}
        ${renderedSources.hiddenHtml || expandedMoreHtml ? `<div class="msg-search-extra-chips msg-search-extra-chips--batch">${renderedSources.hiddenHtml}${expandedMoreHtml}</div>` : ''}
      </div>
    `;
  }

  // Render one search tool activity card.
  function renderSearchToolCard(segment, toolSegmentIndex, options) {
    const renderOptions = options || {};
    if (isImageSearchToolSegment(segment)) {
      return renderImageSearchToolCard(segment, toolSegmentIndex, renderOptions);
    }
    const normalizedRequest = normalizedSearchRequestFromSegment(segment);
    const preparedEntries = searchQueryEntriesFromSegment(segment);
    if (normalizedRequest && preparedEntries.length > 0) {
      return renderPreparedSearchToolCard(
        segment,
        toolSegmentIndex,
        renderOptions,
        preparedEntries
      );
    }
    const hasResult = segment.result !== null && segment.result !== undefined;
    const compact = segment.toolUi && segment.toolUi.compact && typeof segment.toolUi.compact === 'object'
      ? segment.toolUi.compact
      : null;
    const instantMode = isInstantSearchFlowSegment(segment);
    const query = instantMode ? '' : searchQueryFromSegment(segment);
    const instantDescription = instantMode ? searchDescriptionFromSegment(segment) : '';
    const sources = searchSourcesFromSegment(segment);
    const renderedSources = renderSearchSourcesWithOverflow(sources, 3);
    const isExpanded = !!renderOptions.expanded;
    const status = segment.toolUi && segment.toolUi.status ? String(segment.toolUi.status) : '';
    const isRejected = status === 'rejected' || /^BAD_QUERY:/i.test(String(segment.result || '').trim());
    let label = '';
    if (instantMode) {
      label = instantDescription || (hasResult ? 'Searched sources' : 'Searching sources');
    } else if (renderOptions.compactLabel && query) {
      label = isRejected ? `Bad query: ${query}` : query;
    } else if (isRejected) {
      label = `Bad query: ${query || 'query'}`;
    } else if (hasResult) {
      label = sources.length > 0 ? `Searched for ${query || 'sources'}` : `No sources found for ${query || 'sources'}`;
    } else {
      label = compact && compact.label ? String(compact.label) : `Searching for ${query || 'sources'}`;
    }
    const chipsHtml = renderedSources.visibleHtml;
    const hiddenChipsHtml = renderedSources.hiddenHtml;
    const moreCount = renderedSources.hiddenCount;
    const moreButtonAttrs = `type="button" data-search-more-count="${moreCount}" aria-expanded="${isExpanded ? 'true' : 'false'}"`;
    const collapsedMoreHtml = moreCount > 0
      ? `<button class="msg-search-chip msg-search-chip--more msg-search-chip--more-collapsed" ${moreButtonAttrs}><span class="msg-search-chip-domain">${escHtml(`${MORE_LABEL} ${moreCount}`)}</span></button>`
      : '';
    const expandedMoreHtml = moreCount > 0
      ? `<button class="msg-search-chip msg-search-chip--more msg-search-chip--more-expanded" ${moreButtonAttrs}><span class="msg-search-chip-domain">${escHtml(HIDE_LABEL)}</span></button>`
      : '';
    const pendingHtml = hasResult ? '' : '<span class="msg-search-pending-dot"></span>';
    const iconHtml = instantMode
      ? (icons.TOOL_SEARCH_ICON || icons.WEB_SEARCH_ICON || icons.GLOBE_ICON)
      : (icons.WEB_SEARCH_ICON || icons.TOOL_SEARCH_ICON || icons.GLOBE_ICON);
    const searchKey = searchSegmentKey(segment, toolSegmentIndex);
    const searchKeyAttr = searchKey ? ` data-search-key="${escapeAttributeValue(searchKey)}"` : '';

    return `
      <div class="msg-search-card${hasResult ? ' is-done' : ' is-pending'}${isRejected ? ' is-error' : ''}${isExpanded ? ' is-expanded' : ''}${renderOptions.compactLabel ? ' msg-search-card--batch-item' : ''}" data-tool-segment-index="${toolSegmentIndex}"${searchKeyAttr}>
        <div class="msg-search-line">
          ${renderOptions.hideIcon ? '' : `<span class="msg-search-icon">${iconHtml}</span>`}
          <span class="msg-search-label">${escHtml(label)}</span>
          ${pendingHtml}
        </div>
        ${chipsHtml || collapsedMoreHtml ? `<div class="msg-search-chips">${chipsHtml}${collapsedMoreHtml}</div>` : ''}
        ${hiddenChipsHtml || expandedMoreHtml ? `<div class="msg-search-extra-chips">${hiddenChipsHtml}${expandedMoreHtml}</div>` : ''}
      </div>
    `;
  }

  // Render one image_search tool call as a compact card with a thumbnail row.
  function renderImageSearchToolCard(segment, toolSegmentIndex, renderOptions) {
    const hasResult = segment.result !== null && segment.result !== undefined;
    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const query = String(args.query || (segment.toolUi && segment.toolUi.query) || '').trim();
    const images = imageSearchImagesFromSegment(segment);
    const isExpanded = !!renderOptions.expanded;
    const status = segment.toolUi && segment.toolUi.status ? String(segment.toolUi.status) : '';
    const isRejected = status === 'rejected';
    const label = isRejected
      ? `Bad query: ${query || 'query'}`
      : hasResult
        ? (images.length > 0 ? `Images for ${query || 'query'}` : `No images found for ${query || 'query'}`)
        : `Searching images for ${query || 'query'}`;
    const pendingHtml = hasResult ? '' : '<span class="msg-search-pending-dot"></span>';
    const iconHtml = icons.WEB_SEARCH_ICON || icons.TOOL_SEARCH_ICON || icons.GLOBE_ICON;
    const galleryHtml = images.length > 0
      ? `<div class="msg-image-gallery">${renderImageRowHtml(images, query)}</div>`
      : '';
    const searchKey = searchSegmentKey(segment, toolSegmentIndex);
    const searchKeyAttr = searchKey ? ` data-search-key="${escapeAttributeValue(searchKey)}"` : '';
    return `
      <div class="msg-search-card msg-search-card--images${hasResult ? ' is-done' : ' is-pending'}${isRejected ? ' is-error' : ''}${isExpanded ? ' is-expanded' : ''}${renderOptions.compactLabel ? ' msg-search-card--batch-item' : ''}" data-tool-segment-index="${toolSegmentIndex}"${searchKeyAttr}>
        <div class="msg-search-line">
          ${renderOptions.hideIcon ? '' : `<span class="msg-search-icon">${iconHtml}</span>`}
          <span class="msg-search-label">${escHtml(label)}</span>
          ${pendingHtml}
        </div>
        ${galleryHtml}
      </div>
    `;
  }

  // Render a batch of related search tool cards.
  function renderSearchToolGroup(searchItems, options) {
    const renderOptions = options || {};
    if (!Array.isArray(searchItems) || searchItems.length === 0) {
      return '';
    }

    if (searchItems.length === 1) {
      return renderSearchToolCard(searchItems[0].segment, searchItems[0].index, renderOptions);
    }

    const instantMode = searchItems.some(function hasInstantSearch(item) {
      return isInstantSearchFlowSegment(item.segment);
    });
    const iconHtml = instantMode
      ? (icons.TOOL_SEARCH_ICON || icons.WEB_SEARCH_ICON || icons.GLOBE_ICON)
      : (icons.WEB_SEARCH_ICON || icons.TOOL_SEARCH_ICON || icons.GLOBE_ICON);
    const activePendingIndex = searchItems.findIndex(function findPendingSearch(item) {
      return item.segment.result === null || item.segment.result === undefined;
    });
    const hasPending = activePendingIndex !== -1;
    const instantDescription = instantMode
      ? searchItems.map(function mapInstantDescription(item) {
          return searchDescriptionFromSegment(item.segment);
        }).find(Boolean) || (hasPending ? 'Searching sources' : 'Searched sources')
      : '';
    const queriesHtml = instantMode
      ? `<div class="msg-search-batch-query${hasPending ? ' is-active' : ''}"><span class="msg-search-batch-query-text">${escHtml(instantDescription)}</span>${hasPending ? '<span class="msg-search-pending-dot"></span>' : ''}</div>`
      : searchItems.flatMap(function renderBatchQuery(item, itemIndex) {
      const entries = searchQueryEntriesFromSegment(item.segment);
      const itemStatus = item.segment.toolUi && item.segment.toolUi.status ? String(item.segment.toolUi.status) : '';
      const itemRejected = itemStatus === 'rejected' || /^BAD_QUERY:/i.test(String(item.segment.result || '').trim());
      const compact = item.segment.toolUi && item.segment.toolUi.compact && typeof item.segment.toolUi.compact === 'object'
        ? item.segment.toolUi.compact
        : null;
      const displayEntries = entries.length > 0
        ? entries
        : [{ query: compact && compact.label ? String(compact.label).replace(/^Searching for\s+/i, '') : 'sources', vertical: '' }];
      return displayEntries.map(function renderEntry(entry) {
        const label = itemRejected ? `Bad query: ${entry.query}` : entry.query;
        const activeClass = itemIndex === activePendingIndex ? ' is-active' : '';
        const activeDot = itemIndex === activePendingIndex ? '<span class="msg-search-pending-dot"></span>' : '';
        return `<div class="msg-search-batch-query${activeClass}"><span class="msg-search-batch-query-text">${escHtml(label)}</span>${activeDot}</div>`;
      });
    }).join('');
    const combinedSources = dedupeSearchSources(searchItems.flatMap(function collectSources(item) {
      return searchSourcesFromSegment(item.segment);
    }));
    const renderedSources = renderSearchSourcesWithOverflow(combinedSources, 3);
    const chipsHtml = renderedSources.visibleHtml;
    const hiddenChipsHtml = renderedSources.hiddenHtml;
    const moreCount = renderedSources.hiddenCount;
    const firstIndex = Number.isInteger(searchItems[0].index) ? searchItems[0].index : 0;
    const isExpanded = renderOptions.expanded === undefined ? false : !!renderOptions.expanded;
    const searchKey = searchItems
      .map(function mapSearchItemKey(item) { return searchSegmentKey(item.segment, item.index); })
      .filter(Boolean)
      .join('|');
    const searchKeyAttr = searchKey ? ` data-search-key="${escapeAttributeValue(searchKey)}"` : '';
    const moreButtonAttrs = `type="button" data-search-more-count="${moreCount}" aria-expanded="${isExpanded ? 'true' : 'false'}"`;
    const collapsedMoreHtml = moreCount > 0
      ? `<button class="msg-search-chip msg-search-chip--more msg-search-chip--more-collapsed" ${moreButtonAttrs}><span class="msg-search-chip-domain">${escHtml(`${MORE_LABEL} ${moreCount}`)}</span></button>`
      : '';
    const expandedMoreHtml = moreCount > 0
      ? `<button class="msg-search-chip msg-search-chip--more msg-search-chip--more-expanded" ${moreButtonAttrs}><span class="msg-search-chip-domain">${escHtml(HIDE_LABEL)}</span></button>`
      : '';

    return `
      <div class="msg-search-card msg-search-card--batch${hasPending ? ' is-pending' : ' is-done'}${isExpanded ? ' is-expanded' : ''}" data-tool-segment-index="${firstIndex}"${searchKeyAttr}>
        <div class="msg-search-batch-head">
          <span class="msg-search-icon msg-search-batch-icon">${iconHtml}</span>
          <div class="msg-search-batch-queries">${queriesHtml}</div>
        </div>
        ${chipsHtml || collapsedMoreHtml ? `<div class="msg-search-chips msg-search-chips--batch">${chipsHtml}${collapsedMoreHtml}</div>` : ''}
        ${hiddenChipsHtml || expandedMoreHtml ? `<div class="msg-search-extra-chips msg-search-extra-chips--batch">${hiddenChipsHtml}${expandedMoreHtml}</div>` : ''}
      </div>
    `;
  }

  // Render one read-page tool activity card.
  function renderReadPageToolCard(readItems) {
    const items = Array.isArray(readItems) ? readItems : [];
    const firstItem = items[0] || {};
    const sourcesByUrl = {};
    items.forEach(function collectSources(item) {
      readPageSourcesFromSegment(item.segment).forEach(function collectSource(source) {
        const key = String(source.url || source.domain || '').trim();
        if (key && !sourcesByUrl[key]) {
          sourcesByUrl[key] = source;
        }
      });
    });

    const sources = Object.keys(sourcesByUrl).map(function mapSource(key) { return sourcesByUrl[key]; });
    const chipsHtml = sources.map(renderSourceChip).join('');
    const resultCount = sources.length;
    const label = resultCount === 1 ? 'Read source:' : 'Read sources:';
    const status = firstItem.segment && firstItem.segment.toolUi && firstItem.segment.toolUi.status
      ? String(firstItem.segment.toolUi.status)
      : '';
    const iconHtml = icons.WEB_SEARCH_ICON || icons.TOOL_SEARCH_ICON || icons.GLOBE_ICON;
    const dataIndex = Number.isInteger(firstItem.index) ? ` data-tool-segment-index="${firstItem.index}"` : '';

    return `
      <div class="msg-read-page-card${status === 'error' ? ' is-error' : ' is-done'}"${dataIndex}>
        <div class="msg-read-page-line">
          <span class="msg-read-page-icon">${iconHtml}</span>
          <span class="msg-read-page-label">${escHtml(label)}</span>
        </div>
        ${chipsHtml ? `<div class="msg-read-page-chips">${chipsHtml}</div>` : ''}
      </div>
    `;
  }

  // Read the target path from one write tool segment.
  function writePathFromSegment(segment) {
    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    return String(args.path || args.file || args.filename || '').trim();
  }

  // Read file content from one write tool segment.
  function writeContentFromSegment(segment) {
    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    if (args.content !== undefined && args.content !== null) {
      return String(args.content);
    }
    if (args.text !== undefined && args.text !== null) {
      return String(args.text);
    }
    if (args.input !== undefined && args.input !== null) {
      return String(args.input);
    }
    return '';
  }

  // Render collapsed or expanded write preview lines.
  function renderWritePreviewLines(content, isExpanded, path) {
    const lines = String(content || '').split(/\r?\n/);
    const maxLines = isExpanded ? WRITE_PREVIEW_EXPANDED_LINES : WRITE_PREVIEW_COLLAPSED_LINES;
    const visibleLines = lines.slice(0, maxLines);
    const language = languageFromPath(path, 'plaintext');
    const rowsHtml = visibleLines.map(function renderLine(line, index) {
      const isFadeLine = !isExpanded && index === WRITE_PREVIEW_COLLAPSED_LINES - 1 && lines.length > WRITE_PREVIEW_COLLAPSED_LINES;
      const lineClass = isFadeLine ? ' msg-write-line--fade' : '';
      const lineText = isFadeLine && !line ? '...' : line;
      return `<div class="msg-write-line${lineClass}">${lineText ? highlightCode(lineText, language) : ' '}</div>`;
    }).join('');
    if (isExpanded && lines.length > maxLines) {
      return `${rowsHtml}<div class="msg-write-line msg-write-line--fade">... ${escHtml(String(lines.length - maxLines))} more lines omitted</div>`;
    }
    return rowsHtml;
  }

  // Render one write-file tool activity card.
  function renderWriteToolCard(segment, toolSegmentIndex, options) {
    const renderOptions = options || {};
    const content = writeContentFromSegment(segment);
    const path = writePathFromSegment(segment);
    const isExpanded = !!renderOptions.expanded;
    const lineCount = textLineCount(content);
    const byteCount = utf8ByteLength(content);
    const hasMore = lineCount > WRITE_PREVIEW_COLLAPSED_LINES;
    const dataIndex = Number.isInteger(toolSegmentIndex) ? ` data-write-segment-index="${toolSegmentIndex}"` : '';
    const label = path ? `Write ${path}` : 'Write';
    const summary = lineCount > 0
      ? `${lineCount} ${lineCount === 1 ? 'line' : 'lines'}${byteCount > 8192 ? ` · ${Math.round(byteCount / 1024)} KB` : ''}`
      : toolStatusText(segment);
    const language = languageFromPath(path, 'plaintext');

    return `
      <div class="msg-write-card${toolStatusClass(segment)}${isExpanded ? ' is-expanded' : ''}${hasMore ? ' has-more' : ''}"${dataIndex} role="button" tabindex="0" aria-expanded="${isExpanded ? 'true' : 'false'}">
        <span class="msg-write-head">
          ${icons.TOOL_MAKE_FILE_ICON || ''}
          <span class="msg-write-title">${escHtml(label)}</span>
          <span class="msg-write-summary">${escHtml(summary)}</span>
        </span>
        <span class="msg-write-preview msg-code-block language-${escapeAttributeValue(language)}" data-language="${escapeAttributeValue(language)}">${content ? renderWritePreviewLines(content, isExpanded, path) : '<span class="msg-write-line msg-write-line--empty">No content.</span>'}</span>
      </div>
    `;
  }

  // Parse one tool segment result JSON into an object.
  function parseToolResultObject(segment) {
    const rawResult = segment && segment.result !== null && segment.result !== undefined
      ? String(segment.result)
      : '';
    if (!rawResult) {
      return {};
    }

    try {
      const parsed = JSON.parse(rawResult);
      if (!parsed || typeof parsed !== 'object') {
        return {};
      }
      return parsed.result && typeof parsed.result === 'object' ? parsed.result : parsed;
    } catch (_error) {
      return {};
    }
  }

  // Shared file and compression cards.
  function formatByteSize(bytes) {
    const size = Number(bytes);
    if (!Number.isFinite(size) || size < 0) {
      return '';
    }
    if (size < 1024) {
      return `${Math.round(size)} B`;
    }
    if (size < 1024 * 1024) {
      return `${Math.round(size / 1024)} KB`;
    }
    return `${(size / (1024 * 1024)).toFixed(size < 10 * 1024 * 1024 ? 1 : 0)} MB`;
  }

  // Normalize one shared file payload for UI rendering.
  function normalizeSharedFilePayload(candidate) {
    if (!candidate || typeof candidate !== 'object') {
      return null;
    }
    const file = candidate.kind === 'shared_file'
      ? candidate
      : (candidate.file && typeof candidate.file === 'object' ? candidate.file : null);
    if (!file || typeof file !== 'object') {
      return null;
    }
    function asScalarText(value) {
      if (value === null || value === undefined) {
        return '';
      }
      if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
        return String(value).trim();
      }
      if (typeof value === 'object') {
        const preferredKeys = ['filename', 'name', 'path', 'file', 'value', 'label'];
        for (let index = 0; index < preferredKeys.length; index += 1) {
          const key = preferredKeys[index];
          if (Object.prototype.hasOwnProperty.call(value, key)) {
            const nested = asScalarText(value[key]);
            if (nested) {
              return nested;
            }
          }
        }
      }
      return '';
    }

    const path = asScalarText(file.path || file.file);
    const downloadUrl = asScalarText(file.download_url || file.downloadUrl || file.content_url || file.contentUrl || file.url);
    if (!path && !downloadUrl) {
      return null;
    }
    const filename = asScalarText(file.filename || file.name) || String(path || downloadUrl).split(/[\\/]/).pop() || 'download';
    return {
      path,
      filename,
      mimeType: asScalarText(file.mime_type || file.mimeType || file.type),
      typeLabel: asScalarText(file.type_label || file.typeLabel || file.mime_type || file.mimeType) || 'File',
      sizeBytes: Number(file.size_bytes ?? file.sizeBytes ?? -1),
      downloadUrl,
      modelContext: asScalarText(file.model_context || file.modelContext),
      render: file.render && typeof file.render === 'object' ? file.render : null
    };
  }

  // Extract a shared file payload from one tool segment.
  function sharedFileFromSegment(segment) {
    const candidates = [
      segment && segment.structuredContent,
      segment && segment.toolUi,
      segment && segment.toolUi && segment.toolUi.file,
      parseToolResultObject(segment)
    ];

    for (let index = 0; index < candidates.length; index += 1) {
      const normalized = normalizeSharedFilePayload(candidates[index]);
      if (normalized) {
        return normalized;
      }
    }
    return null;
  }

  // Report whether one segment represents a shared file.
  function isSharedFileToolSegment(segment) {
    if (isSandboxShareFileToolSegment(segment)) {
      return Boolean(sharedFileFromSegment(segment));
    }
    if (sharedFileFromSegment(segment)) {
      return true;
    }
    return /share[_-\s]?file|download[_-\s]?file/.test(toolIdentityText(segment));
  }

  // Report whether one segment is a context compression marker.
  function isCompressionContextSegment(segment) {
    const alias = String(segment && (segment.alias || segment.toolId || segment.toolName || '') || '').trim().toLowerCase();
    const toolUi = segment && segment.toolUi && typeof segment.toolUi === 'object' ? segment.toolUi : {};
    const structured = segment && segment.structuredContent && typeof segment.structuredContent === 'object'
      ? segment.structuredContent
      : {};
    return alias === 'context_compression_summary'
      || String(toolUi.kind || '').toLowerCase() === 'context_compression'
      || String(structured.kind || '').toLowerCase() === 'context_compression';
  }

  // Read display text from one compression context segment.
  function compressionContextText(segment) {
    return String(segment && segment.result ? segment.result : '').trim();
  }

  // Render one context compression activity card.
  function renderCompressionContextCard(segment, toolSegmentIndex) {
    const text = compressionContextText(segment);
    if (!text) {
      return '';
    }
    const dataIndex = Number.isInteger(toolSegmentIndex) ? ` data-tool-segment-index="${toolSegmentIndex}"` : '';
    return `
      <div class="msg-compression-context"${dataIndex}>
        <div class="msg-compression-context-line">
          <span class="msg-compression-context-divider"></span>
          <button type="button" class="msg-compression-context-btn">Compressed context active</button>
          <span class="msg-compression-context-divider"></span>
        </div>
        <pre class="msg-compression-context-json msg-code-block language-json" style="display:none;"><code>${highlightCode(text, 'json')}</code></pre>
      </div>
    `;
  }

  // Build the pending row shown during compression.
  function buildCompressionPendingRow() {
    return $(`
      <div class="msg msg-compression-marker msg-compression-marker--pending" data-message-key="context-compression-pending">
        <div class="msg-compression-marker-body">
          <div class="msg-compression-context is-pending">
            <div class="msg-compression-context-line">
              <span class="msg-compression-context-divider"></span>
              <span class="msg-compression-context-pending">
                <span class="msg-compression-context-spinner" aria-hidden="true"></span>
                Compressing context...
              </span>
              <span class="msg-compression-context-divider"></span>
            </div>
          </div>
        </div>
      </div>
    `);
  }

  // Append a context compression pending row to the chat.
  function appendCompressionPending() {
    const existing = dom.$messagesInner.find('.msg-compression-marker--pending').last();
    if (existing.length) {
      scrollBottom();
      return existing;
    }
    const $row = buildCompressionPendingRow();
    dom.$messagesInner.append($row);
    scrollBottom();
    return $row;
  }

  // Remove one compression pending row from the chat.
  function removeCompressionPending($row) {
    const target = $row && $row.length ? $row : dom.$messagesInner.find('.msg-compression-marker--pending');
    target.remove();
  }

  // Report whether segments contain only compression markers.
  function isCompressionOnlyActivitySegments(segments) {
    return Array.isArray(segments)
      && segments.length > 0
      && segments.every(function onlyCompression(segment) {
        return segment && segment.type === 'tool' && isCompressionContextSegment(segment);
      });
  }

  // Resolve the download URL for one shared file card.
  function sharedFileDownloadUrl(file, options) {
    if (!file) {
      return '';
    }
    if (file.downloadUrl) {
      const downloadUrl = String(file.downloadUrl || '').trim();
      if (!(options && options.preview) || !downloadUrl) {
        return downloadUrl;
      }

      // The backend intentionally serves the same shared-file endpoint as an
      // attachment for downloads and inline for previews. Preserve external
      // or signed URLs, but opt our own endpoint into inline rendering so
      // large animated GIF/SVG files keep working when no base64 preview fits.
      try {
        const baseUrl = typeof window !== 'undefined' && window.location
          ? window.location.href
          : 'http://localhost/';
        const parsedUrl = new URL(downloadUrl, baseUrl);
        const baseOrigin = new URL(baseUrl).origin;
        if (parsedUrl.origin === baseOrigin && parsedUrl.pathname === '/api/shared-file/download/') {
          parsedUrl.searchParams.set('preview', '1');
          return /^https?:\/\//i.test(downloadUrl)
            ? parsedUrl.toString()
            : `${parsedUrl.pathname}${parsedUrl.search}${parsedUrl.hash}`;
        }
      } catch (_error) {
        // Keep malformed or non-standard URLs unchanged.
      }
      return downloadUrl;
    }
    if (!file.path) {
      return '';
    }
    const params = new URLSearchParams({
      path: file.path,
      name: file.filename || 'download'
    });
    if (options && options.preview) {
      params.set('preview', '1');
    }
    return `/api/shared-file/download/?${params.toString()}`;
  }

  // Build image preview metadata for one shared file.
  function sharedFileImageRender(file) {
    const render = file && file.render && typeof file.render === 'object' ? file.render : null;
    if (render && String(render.type || '').toLowerCase() === 'image') {
      const src = sandboxImageDataUrl({
        mime: render.mime_type || file.mimeType,
        preview: render.preview
      });
      if (src) {
        return {
          src,
          width: Number(render.width),
          height: Number(render.height),
          mimeType: String(render.mime_type || file.mimeType || '').trim()
        };
      }
    }

    const mimeType = String(file && file.mimeType ? file.mimeType : '').trim().toLowerCase();
    const filename = String(file && file.filename ? file.filename : '').trim().toLowerCase();
    const looksLikeImage = mimeType.startsWith('image/')
      || /\.(png|jpe?g|gif|svg|webp|bmp|avif)$/i.test(filename);
    if (!looksLikeImage) {
      return null;
    }

    return {
      src: sharedFileDownloadUrl(file, { preview: true }),
      width: NaN,
      height: NaN,
      mimeType: mimeType || 'image'
    };
  }

  // Attachment card helpers.
  function normalizeAttachmentCardFile(file, options) {
    const renderOptions = options || {};
    if (!file) {
      return null;
    }
    if (renderOptions.source === 'shared_file') {
      return file;
    }
    return attachmentUi.normalizeAttachment(file);
  }

  // Read the filename from one attachment record.
  function attachmentFilename(file) {
    return String((file && (file.filename || file.name)) || 'File').trim() || 'File';
  }

  // Read the MIME type from one attachment record.
  function attachmentMimeType(file) {
    return String((file && (file.mimeType || file.mime_type || file.type)) || '').trim().toLowerCase();
  }

  // Build a short type label for one attachment.
  function attachmentTypeLabel(file) {
    const rawLabel = String((file && (file.typeLabel || file.type_label || file.mimeType || file.mime_type)) || '').trim();
    if (rawLabel && rawLabel.toLowerCase() !== 'file') {
      return rawLabel;
    }
    const displayKind = inferredAttachmentDisplayKind(file);
    if (displayKind === 'audio') {
      return 'Audio';
    }
    if (displayKind === 'video') {
      return 'Video';
    }
    if (displayKind === 'image') {
      return 'Image';
    }
    return rawLabel || 'File';
  }

  // Read the byte size from one attachment record.
  function attachmentSizeBytes(file) {
    return Number(file && (file.sizeBytes ?? file.size_bytes ?? file.size ?? -1));
  }

  // Resolve the content URL for one attachment card.
  function attachmentSourceUrl(file, options) {
    const renderOptions = options || {};
    if (!file) {
      return '';
    }
    if (renderOptions.source === 'shared_file') {
      return sharedFileDownloadUrl(file, { preview: true });
    }
    return String(file.dataUrl || file.previewDataUrl || file.contentUrl || '').trim();
  }

  // Resolve the download URL for one attachment card.
  function attachmentDownloadUrl(file, options) {
    const renderOptions = options || {};
    if (!file) {
      return '';
    }
    if (renderOptions.source === 'shared_file') {
      return sharedFileDownloadUrl(file);
    }
    return String(file.contentUrl || (!/^data:/i.test(file.dataUrl || '') ? file.dataUrl : '') || '').trim();
  }

  // Read the display kind from one attachment record.
  function attachmentDisplayKind(file) {
    return String((file && (file.displayKind || file.display_kind || file.kind)) || '').trim().toLowerCase();
  }

  // Media attachment rendering.
  function isAudioAttachment(file) {
    const mimeType = attachmentMimeType(file);
    const filename = attachmentFilename(file).toLowerCase();
    const displayKind = attachmentDisplayKind(file);
    return displayKind === 'audio'
      || mimeType.startsWith('audio/')
      || /\.(mp3|wav|ogg|oga|m4a|aac|flac|opus)$/i.test(filename);
  }

  // Report whether one attachment should render as video.
  function isVideoAttachment(file) {
    const mimeType = attachmentMimeType(file);
    const filename = attachmentFilename(file).toLowerCase();
    const displayKind = attachmentDisplayKind(file);
    return displayKind === 'video'
      || mimeType.startsWith('video/')
      || /\.(mp4|webm|mov|m4v|ogv|avi|mkv)$/i.test(filename);
  }

  // Report whether one attachment should render as an image.
  function isImageAttachment(file) {
    const mimeType = attachmentMimeType(file);
    const filename = attachmentFilename(file).toLowerCase();
    const displayKind = attachmentDisplayKind(file);
    return displayKind === 'image'
      || mimeType.startsWith('image/')
      || /\.(png|jpe?g|gif|svg|webp|bmp|avif)$/i.test(filename);
  }

  // Build type and size metadata text for media cards.
  function mediaMetaText(file) {
    const typeLabel = attachmentTypeLabel(file);
    const sizeText = formatByteSize(attachmentSizeBytes(file));
    return [typeLabel, sizeText].filter(Boolean).join(' / ');
  }

  // Render one download button for an attachment card.
  function renderAttachmentDownloadButton(file, options, className) {
    const href = attachmentDownloadUrl(file, options);
    if (!href) {
      return '';
    }
    return `
      <a class="${className || 'msg-media-download'}" href="${escapeAttributeValue(href)}" download="${escapeAttributeValue(attachmentFilename(file))}" title="Download" aria-label="Download ${escapeAttributeValue(attachmentFilename(file))}">
        ${DOWNLOAD_FILE_ICON}
      </a>
    `;
  }

  // Build the file-manager action embedded before download in shared-file cards.
  function renderSharedFileRevealButton(file) {
    const path = String(file && file.path ? file.path : '').trim();
    if (!path) {
      return '';
    }
    const filename = attachmentFilename(file);
    const ariaLabel = `${OPEN_FILE_LOCATION_LABEL}: ${filename}`;
    return `
      <span class="msg-shared-file-reveal" role="button" tabindex="0" data-shared-file-path="${escapeAttributeValue(path)}" title="${escapeAttributeValue(OPEN_FILE_LOCATION_LABEL)}" aria-label="${escapeAttributeValue(ariaLabel)}">
        ${SHARED_FILE_FOLDER_ICON}
      </span>
    `;
  }

  // Render the inline reveal/download group used by shared media cards.
  function renderAttachmentActions(file, options, downloadClassName) {
    const downloadButton = renderAttachmentDownloadButton(file, options, downloadClassName);
    const revealButton = options && options.source === 'shared_file'
      ? renderSharedFileRevealButton(file)
      : '';
    if (!revealButton) {
      return downloadButton;
    }
    return `<span class="msg-shared-file-actions msg-media-actions">${revealButton}${downloadButton}</span>`;
  }

  // Render one image attachment card.
  function renderImageAttachmentCard(file, options) {
    const renderOptions = options || {};
    const sharedRender = renderOptions.source === 'shared_file' ? sharedFileImageRender(file) : null;
    const src = sharedRender ? sharedRender.src : attachmentSourceUrl(file, renderOptions);
    if (!src) {
      return renderGenericAttachmentCard(file, renderOptions);
    }
    const filename = attachmentFilename(file);
    const width = Number(sharedRender && sharedRender.width);
    const height = Number(sharedRender && sharedRender.height);
    const hasDimensions = Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0;
    const details = [
      formatByteSize(attachmentSizeBytes(file)),
      hasDimensions ? `${Math.round(width)}x${Math.round(height)}` : '',
      (sharedRender && sharedRender.mimeType) || attachmentTypeLabel(file)
    ].filter(Boolean).join(' / ');
    const href = attachmentDownloadUrl(file, renderOptions) || src;
    const tagName = href ? 'a' : 'span';
    const hrefAttr = href ? ` href="${escapeAttributeValue(href)}" download="${escapeAttributeValue(filename)}"` : '';

    const cardHtml = `
      <${tagName} class="msg-attachment-card msg-image-card${renderOptions.source === 'shared_file' ? ' msg-shared-image-card' : ''}"${hrefAttr}>
        <span class="msg-shared-image-preview">
          <img class="msg-shared-image" src="${escapeAttributeValue(src)}" alt="${escapeAttributeValue(filename ? `Preview for ${filename}` : 'Attached image')}" loading="lazy">
        </span>
        <span class="msg-shared-image-footer">
          <span class="msg-shared-image-meta">
            <span class="msg-shared-image-name">${escHtml(filename)}</span>
            <span class="msg-shared-image-details">${escHtml(details)}</span>
          </span>
          <span class="msg-shared-file-actions msg-shared-image-actions">
            ${renderOptions.source === 'shared_file' ? renderSharedFileRevealButton(file) : ''}
            ${href ? `<span class="msg-shared-image-download" aria-hidden="true">${DOWNLOAD_FILE_ICON}</span>` : ''}
          </span>
        </span>
      </${tagName}>
    `;
    return cardHtml;
  }

  // Render one generic file attachment chip.
  function renderGenericAttachmentCard(file, options) {
    const renderOptions = options || {};
    const filename = attachmentFilename(file);
    const href = attachmentDownloadUrl(file, renderOptions);
    const tagName = href ? 'a' : 'div';
    const hrefAttr = href ? ` href="${escapeAttributeValue(href)}" download="${escapeAttributeValue(filename)}"` : '';
    const badgeLabel = inferredAttachmentDisplayKind(file).toUpperCase();

    const cardHtml = `
      <${tagName} class="msg-file-chip msg-attachment-card msg-shared-file-card"${hrefAttr}>
        <span class="msg-shared-file-badge" aria-hidden="true">${escHtml(badgeLabel.slice(0, 6) || 'FILE')}</span>
        <span class="msg-shared-file-main">
          <span class="msg-file-name">${escHtml(filename)}</span>
          <span class="msg-file-meta">${escHtml(mediaMetaText(file))}</span>
        </span>
        <span class="msg-shared-file-actions">
          ${renderOptions.source === 'shared_file' ? renderSharedFileRevealButton(file) : ''}
          ${href ? `<span class="msg-shared-file-download" aria-hidden="true">${DOWNLOAD_FILE_ICON}</span>` : ''}
        </span>
      </${tagName}>
    `;
    return cardHtml;
  }

  // Infer image, audio, video, or file from metadata.
  function inferredAttachmentDisplayKind(file) {
    let displayKind = attachmentDisplayKind(file);
    if (!displayKind || displayKind === 'file') {
      if (isAudioAttachment(file)) {
        displayKind = 'audio';
      } else if (isVideoAttachment(file)) {
        displayKind = 'video';
      } else if (isImageAttachment(file)) {
        displayKind = 'image';
      } else if (isUrlAttachment(file)) {
        displayKind = 'url';
      } else {
        displayKind = 'file';
      }
    }
    return displayKind;
  }

  function isUrlAttachment(file) {
    const kind = attachmentDisplayKind(file);
    const mime = attachmentMimeType(file);
    const name = attachmentFilename(file);
    return kind === 'url' || /url|x-url|uri-list/i.test(mime) || /^https?:\/\//i.test(name);
  }

  // Render one compact chip for a user upload.
  function renderUserUploadAttachmentChip(file) {
    const filename = attachmentFilename(file);
    const badgeLabel = inferredAttachmentDisplayKind(file).toUpperCase();

    return `
      <div class="msg-upload-file-chip msg-attachment-card">
        <span class="msg-upload-file-badge" aria-hidden="true">${escHtml(badgeLabel.slice(0, 6) || 'FILE')}</span>
        <span class="msg-upload-file-main">
          <span class="msg-upload-file-name">${escHtml(filename)}</span>
          <span class="msg-upload-file-meta">${escHtml(mediaMetaText(file))}</span>
        </span>
      </div>
    `;
  }

  // Render URL attachment (fixed "URL" label, address as description, truncated to fit).
  function renderUrlAttachmentCard(file) {
    const url = file.url || file.name || file.contentUrl || attachmentFilename(file);
    const short = (url.length > 55) ? url.slice(0, 32) + '...' + url.slice(-18) : url;
    return `
      <div class="msg-upload-file-chip msg-attachment-card msg-url-card">
        <span class="msg-upload-file-badge" aria-hidden="true">URL</span>
        <span class="msg-upload-file-main">
          <span class="msg-upload-file-name" title="${escapeAttributeValue(url)}">${escHtml(short)}</span>
          <span class="msg-upload-file-meta">URL</span>
        </span>
      </div>
    `;
  }

  // Render one audio attachment player card.
  function renderAudioAttachmentCard(file, options) {
    const renderOptions = options || {};
    const src = attachmentSourceUrl(file, renderOptions);
    const filename = attachmentFilename(file);
    const disabledClass = src ? '' : ' is-unavailable';

    return `
      <div class="msg-attachment-card msg-media-card msg-audio-card${disabledClass}" data-media-card data-media-type="audio">
        ${src ? `<audio class="msg-media-native" preload="metadata" src="${escapeAttributeValue(src)}"></audio>` : ''}
        <div class="msg-audio-control">
          <button class="msg-media-play-btn" type="button" data-media-action="toggle" aria-label="Play audio"${src ? '' : ' disabled'}>
            ${MEDIA_PLAY_ICON}
          </button>
        </div>
        <div class="msg-media-main">
          <div class="msg-media-kicker">
            <span class="msg-media-type-pill">AUDIO</span>
          </div>
          <div class="msg-media-name">${escHtml(filename)}</div>
          <div class="msg-media-meta">${escHtml(mediaMetaText(file))}</div>
          <div class="msg-audio-progress-row">
            <span class="msg-media-time msg-audio-time" data-media-time>0:00 / --:--</span>
            <input class="msg-media-range" type="range" min="0" max="1000" value="0" step="1" data-media-action="seek" aria-label="Audio progress"${src ? '' : ' disabled'}>
          </div>
        </div>
        ${renderAttachmentActions(file, renderOptions, 'msg-media-download')}
      </div>
    `;
  }

  // Render one video attachment player card.
  function renderVideoAttachmentCard(file, options) {
    const renderOptions = options || {};
    const src = attachmentSourceUrl(file, renderOptions);
    const filename = attachmentFilename(file);
    const disabledClass = src ? '' : ' is-unavailable';

    return `
      <div class="msg-attachment-card msg-media-card msg-video-card${disabledClass}" data-media-card data-media-type="video">
        <div class="msg-video-viewport">
          ${src ? `<video class="msg-media-native" preload="metadata" src="${escapeAttributeValue(src)}"></video>` : '<div class="msg-video-empty">Video preview unavailable</div>'}
          <button class="msg-video-center-play" type="button" data-media-action="toggle" aria-label="Play video"${src ? '' : ' disabled'}>
            ${MEDIA_PLAY_ICON}
          </button>
          <div class="msg-video-controls">
            <button class="msg-media-icon-btn" type="button" data-media-action="toggle" aria-label="Play video"${src ? '' : ' disabled'}>${MEDIA_PLAY_ICON}</button>
            <span class="msg-media-time" data-media-current>0:00</span>
            <input class="msg-media-range" type="range" min="0" max="1000" value="0" step="1" data-media-action="seek" aria-label="Video progress"${src ? '' : ' disabled'}>
            <span class="msg-media-time" data-media-duration>--:--</span>
            <button class="msg-media-icon-btn" type="button" data-media-action="mute" aria-label="Mute video"${src ? '' : ' disabled'}>${MEDIA_VOLUME_ICON}</button>
            <button class="msg-media-icon-btn" type="button" data-media-action="fullscreen" aria-label="Fullscreen"${src ? '' : ' disabled'}>${MEDIA_FULLSCREEN_ICON}</button>
            <button class="msg-media-icon-btn msg-media-popout-btn" type="button" data-media-action="popout" aria-label="Open mini player"${src ? '' : ' disabled'}>${MEDIA_POPOUT_ICON}</button>
            <button class="msg-media-icon-btn msg-media-dock-btn" type="button" data-media-action="dock" aria-label="Return to message">${MEDIA_DOCK_ICON}</button>
            <button class="msg-media-icon-btn msg-media-close-btn" type="button" data-media-action="close-floating" aria-label="Close mini player">${MEDIA_CLOSE_ICON}</button>
          </div>
        </div>
        <div class="msg-media-info">
          <span class="msg-media-type-badge" aria-hidden="true">${MEDIA_VIDEO_ICON || 'VIDEO'}</span>
          <span class="msg-media-copy">
            <span class="msg-media-name">${escHtml(filename)}</span>
            <span class="msg-media-meta">${escHtml(mediaMetaText(file))}</span>
          </span>
          ${renderAttachmentActions(file, renderOptions, 'msg-media-download')}
        </div>
      </div>
    `;
  }

  // Render the correct attachment card for one file record.
  function renderAttachmentCard(file, options) {
    const renderOptions = options || {};
    const normalized = normalizeAttachmentCardFile(file, renderOptions);
    if (!normalized) {
      return '';
    }
    if (renderOptions.source === 'upload' && renderOptions.side === 'user') {
      if (isUrlAttachment(normalized)) {
        return renderUrlAttachmentCard(normalized);
      }
      return renderUserUploadAttachmentChip(normalized);
    }
    if (isUrlAttachment(normalized)) {
      return renderUrlAttachmentCard(normalized);
    }
    if (isAudioAttachment(normalized)) {
      return renderAudioAttachmentCard(normalized, renderOptions);
    }
    if (isVideoAttachment(normalized)) {
      return renderVideoAttachmentCard(normalized, renderOptions);
    }
    if (isImageAttachment(normalized)) {
      return renderImageAttachmentCard(normalized, renderOptions);
    }
    return renderGenericAttachmentCard(normalized, renderOptions);
  }

  // Render message attachments.
  function renderMessageAttachments(attachments, options) {
    const cards = (attachments || [])
      .map(function renderOneAttachment(attachment) {
        return renderAttachmentCard(attachment, options);
      })
      .filter(function hasCard(card) {
        return !!String(card || '').trim();
      });
    if (!cards.length) {
      return '';
    }
    return `<div class="msg-attachments msg-attachments--${escapeAttributeValue((options && options.side) || 'message')}">${cards.join('')}</div>`;
  }

  // Format seconds as m:ss for media controls.
  function formatMediaTime(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) {
      return '--:--';
    }
    const whole = Math.floor(value);
    const minutes = Math.floor(whole / 60);
    const remainingSeconds = whole % 60;
    return `${minutes}:${String(remainingSeconds).padStart(2, '0')}`;
  }

  // Handle media element from card.
  function mediaElementFromCard(card) {
    return card ? card.querySelector('audio, video') : null;
  }

  // Handle buffered progress percent.
  function bufferedProgressPercent(media, duration, progressPercent) {
    const buffered = media && media.buffered;
    if (!buffered || !Number.isFinite(duration) || duration <= 0) {
      return progressPercent;
    }

    let bufferedEnd = 0;
    const currentTime = Number(media.currentTime) || 0;
    for (let index = 0; index < buffered.length; index += 1) {
      const start = buffered.start(index);
      const end = buffered.end(index);
      if (start <= currentTime && currentTime <= end) {
        bufferedEnd = Math.max(bufferedEnd, end);
      } else {
        bufferedEnd = Math.max(bufferedEnd, end);
      }
    }
    return Math.max(progressPercent, Math.min(100, (bufferedEnd / duration) * 100));
  }

  // Handle stop media frame sync.
  function stopMediaFrameSync(card) {
    const frameId = activeMediaFrameIds.get(card);
    if (frameId) {
      window.cancelAnimationFrame(frameId);
      activeMediaFrameIds.delete(card);
    }
  }

  // Handle start media frame sync.
  function startMediaFrameSync(card) {
    if (!card || activeMediaFrameIds.has(card)) {
      return;
    }

    function syncFrame() {
      const media = mediaElementFromCard(card);
      syncMediaCard(card);
      if (media && !media.paused && !media.ended) {
        activeMediaFrameIds.set(card, window.requestAnimationFrame(syncFrame));
      } else {
        activeMediaFrameIds.delete(card);
      }
    }

    activeMediaFrameIds.set(card, window.requestAnimationFrame(syncFrame));
  }

  // Sync media card.
  function syncMediaCard(card) {
    const media = mediaElementFromCard(card);
    if (!card || !media) {
      return;
    }
    const isPlaying = !media.paused && !media.ended;
    const duration = Number(media.duration);
    const currentTime = Number(media.currentTime);
    const hasDuration = Number.isFinite(duration) && duration > 0;
    const progressValue = hasDuration ? Math.max(0, Math.min(1000, Math.round((currentTime / duration) * 1000))) : 0;
    const progressPercent = progressValue / 10;
    const bufferedPercent = bufferedProgressPercent(media, duration, progressPercent);
    const playIcon = isPlaying ? MEDIA_PAUSE_ICON : MEDIA_PLAY_ICON;

    card.classList.toggle('is-playing', isPlaying);
    card.style.setProperty('--media-progress', `${progressPercent}%`);
    card.style.setProperty('--media-buffered', `${bufferedPercent}%`);
    card.querySelectorAll('[data-media-action="toggle"]').forEach(function syncToggle(button) {
      button.innerHTML = playIcon;
      button.setAttribute('aria-label', isPlaying ? 'Pause media' : 'Play media');
    });
    card.querySelectorAll('[data-media-action="mute"]').forEach(function syncMute(button) {
      button.innerHTML = media.muted || media.volume === 0 ? MEDIA_VOLUME_MUTED_ICON : MEDIA_VOLUME_ICON;
      button.setAttribute('aria-label', media.muted || media.volume === 0 ? 'Unmute media' : 'Mute media');
    });
    card.querySelectorAll('[data-media-action="seek"]').forEach(function syncRange(range) {
      if (document.activeElement !== range) {
        range.value = String(progressValue);
      }
      range.disabled = !hasDuration;
    });
    card.querySelectorAll('[data-media-current]').forEach(function syncCurrent(node) {
      node.textContent = formatMediaTime(currentTime);
    });
    card.querySelectorAll('[data-media-duration]').forEach(function syncDuration(node) {
      node.textContent = formatMediaTime(duration);
    });
    card.querySelectorAll('[data-media-time]').forEach(function syncCombined(node) {
      node.textContent = `${formatMediaTime(currentTime)} / ${formatMediaTime(duration)}`;
    });
    syncFullscreenControls(card);
  }

  // Handle preview media seek.
  function previewMediaSeek(card, value) {
    const media = mediaElementFromCard(card);
    const duration = Number(media && media.duration);
    if (!card || !Number.isFinite(duration) || duration <= 0) {
      return;
    }
    const progressValue = Math.max(0, Math.min(1000, Number(value) || 0));
    const progressPercent = progressValue / 10;
    const seekTime = (progressValue / 1000) * duration;
    card.style.setProperty('--media-progress', `${progressPercent}%`);
    card.style.setProperty('--media-buffered', `${bufferedProgressPercent(media, duration, progressPercent)}%`);
    card.querySelectorAll('[data-media-current]').forEach(function syncCurrent(node) {
      node.textContent = formatMediaTime(seekTime);
    });
    card.querySelectorAll('[data-media-time]').forEach(function syncCombined(node) {
      node.textContent = `${formatMediaTime(seekTime)} / ${formatMediaTime(duration)}`;
    });
  }

  // Handle force pause media.
  function forcePauseMedia(card, media) {
    if (!card || !media) {
      return;
    }
    card._mediaWantsPaused = true;
    stopMediaFrameSync(card);
    media.pause();
    syncMediaCard(card);
  }

  // Handle force play media.
  function forcePlayMedia(card, media) {
    if (!card || !media) {
      return;
    }
    card._mediaWantsPaused = false;
    const playResult = media.play();
    if (playResult && typeof playResult.then === 'function') {
      playResult
        .then(function afterPlay() {
          if (!card._mediaWantsPaused && !media.paused && !media.ended) {
            startMediaFrameSync(card);
          }
        })
        .catch(function ignorePlayError() {
          syncMediaCard(card);
        });
    } else {
      startMediaFrameSync(card);
    }
    syncMediaCard(card);
  }

  // Toggle media card.
  function toggleMediaCard(card) {
    const media = mediaElementFromCard(card);
    if (!media) {
      return;
    }
    if (media.paused || media.ended) {
      forcePlayMedia(card, media);
    } else {
      forcePauseMedia(card, media);
    }
  }

  // Handle commit media seek.
  function commitMediaSeek(range) {
    const card = range && range.closest('[data-media-card]');
    const media = mediaElementFromCard(card);
    if (!media || !Number.isFinite(media.duration) || media.duration <= 0) {
      return;
    }
    const shouldResume = !card._mediaWantsPaused && !media.paused && !media.ended;
    media.currentTime = (Number(range.value) / 1000) * media.duration;
    if (shouldResume) {
      forcePlayMedia(card, media);
    } else {
      syncMediaCard(card);
    }
  }

  // Sync fullscreen controls.
  function syncFullscreenControls(card) {
    if (!card) {
      return;
    }
    const viewport = card.querySelector('.msg-video-viewport');
    const isFullscreen = !!viewport && document.fullscreenElement === viewport;
    card.classList.toggle('is-fullscreen', isFullscreen);
    card.querySelectorAll('[data-media-action="fullscreen"]').forEach(function syncFullscreen(button) {
      button.innerHTML = isFullscreen ? MEDIA_FULLSCREEN_EXIT_ICON : MEDIA_FULLSCREEN_ICON;
      button.setAttribute('aria-label', isFullscreen ? 'Exit fullscreen' : 'Fullscreen');
      button.setAttribute('title', isFullscreen ? 'Exit fullscreen' : 'Fullscreen');
    });
  }

  // Handle ensure floating media root.
  function ensureFloatingMediaRoot() {
    let root = document.getElementById('floating-media-root');
    if (!root) {
      root = document.createElement('div');
      root.id = 'floating-media-root';
      document.body.appendChild(root);
    }
    return root;
  }

  // Open floating video card.
  function openFloatingVideoCard(card) {
    if (!card || card.classList.contains('is-floating')) {
      return;
    }
    const placeholder = document.createElement('div');
    placeholder.className = 'video-inline-placeholder';
    placeholder.setAttribute('aria-hidden', 'true');
    placeholder.style.height = `${Math.max(96, Math.round(card.getBoundingClientRect().height || 0))}px`;
    card.replaceWith(placeholder);
    floatingMediaPlaceholders.set(card, placeholder);
    ensureFloatingMediaRoot().appendChild(card);
    card.classList.add('is-floating');
  }

  // Handle dock floating video card.
  function dockFloatingVideoCard(card) {
    if (!card || !card.classList.contains('is-floating')) {
      return;
    }
    const placeholder = floatingMediaPlaceholders.get(card);
    if (placeholder && placeholder.parentNode) {
      placeholder.replaceWith(card);
    } else {
      dom.$messagesInner.append(card);
    }
    floatingMediaPlaceholders.delete(card);
    card.classList.remove('is-floating');
  }

  // Close floating video card.
  function closeFloatingVideoCard(card) {
    const media = mediaElementFromCard(card);
    forcePauseMedia(card, media);
    dockFloatingVideoCard(card);
  }

  // Media player events.
  function bindAttachmentMediaEvents() {
    $(document)
      .on('click.mediaAttachments', '[data-media-card] [data-media-action="toggle"]', function onMediaToggle(event) {
        event.preventDefault();
        event.stopPropagation();
        const card = this.closest('[data-media-card]');
        toggleMediaCard(card);
      })
      .on('input.mediaAttachments', '[data-media-card] [data-media-action="seek"]', function onMediaSeekPreview() {
        previewMediaSeek(this.closest('[data-media-card]'), this.value);
      })
      .on('change.mediaAttachments', '[data-media-card] [data-media-action="seek"]', function onMediaSeek() {
        commitMediaSeek(this);
      })
      .on('pointerup.mediaAttachments', '[data-media-card] [data-media-action="seek"]', function onMediaSeekPointerUp() {
        commitMediaSeek(this);
      })
      .on('keydown.mediaAttachments', '[data-media-card] [data-media-action="seek"]', function onMediaSeekKey(event) {
        if (event.key === 'Enter' || event.key === ' ') {
          commitMediaSeek(this);
        }
      })
      .on('click.mediaAttachments', '[data-media-card] [data-media-action="mute"]', function onMediaMute(event) {
        event.preventDefault();
        const card = this.closest('[data-media-card]');
        const media = mediaElementFromCard(card);
        if (!media) {
          return;
        }
        media.muted = !media.muted;
        syncMediaCard(card);
      })
      .on('click.mediaAttachments', '[data-media-card] [data-media-action="fullscreen"]', function onMediaFullscreen(event) {
        event.preventDefault();
        const card = this.closest('[data-media-card]');
        const target = card && card.querySelector('.msg-video-viewport');
        if (!target) {
          return;
        }
        if (document.fullscreenElement === target && document.exitFullscreen) {
          document.exitFullscreen().catch(function ignoreFullscreenExitError() {});
          return;
        }
        if (target.requestFullscreen) {
          target.requestFullscreen()
            .then(function afterFullscreen() { syncFullscreenControls(card); })
            .catch(function ignoreFullscreenError() {});
        }
      })
      .on('click.mediaAttachments', '[data-media-card] [data-media-action="popout"]', function onMediaPopout(event) {
        event.preventDefault();
        openFloatingVideoCard(this.closest('.msg-video-card'));
      })
      .on('click.mediaAttachments', '[data-media-card] [data-media-action="dock"]', function onMediaDock(event) {
        event.preventDefault();
        dockFloatingVideoCard(this.closest('.msg-video-card'));
      })
      .on('click.mediaAttachments', '[data-media-card] [data-media-action="close-floating"]', function onMediaClose(event) {
        event.preventDefault();
        closeFloatingVideoCard(this.closest('.msg-video-card'));
      })
      .on('loadedmetadata.mediaAttachments progress.mediaAttachments canplay.mediaAttachments waiting.mediaAttachments seeking.mediaAttachments seeked.mediaAttachments play.mediaAttachments pause.mediaAttachments volumechange.mediaAttachments ended.mediaAttachments', '[data-media-card] audio, [data-media-card] video', function onMediaStateChange(event) {
        const card = this.closest('[data-media-card]');
        if (event.type === 'play') {
          card._mediaWantsPaused = false;
          startMediaFrameSync(card);
        } else if (event.type === 'pause' || event.type === 'ended') {
          card._mediaWantsPaused = true;
          stopMediaFrameSync(card);
        }
        syncMediaCard(card);
      })
      .on('timeupdate.mediaAttachments', '[data-media-card] audio, [data-media-card] video', function onMediaTimeUpdate() {
        // Throttle progress updates to one repaint per animation frame per card
        // to avoid hammering conic-gradient / CSS custom-property recalculations.
        const card = this.closest('[data-media-card]');
        if (!card || card._rafPending) { return; }
        card._rafPending = true;
        requestAnimationFrame(function rafSync() {
          card._rafPending = false;
          syncMediaCard(card);
        });
      });
    document.addEventListener('fullscreenchange', function onMediaFullscreenChange() {
      document.querySelectorAll('[data-media-card]').forEach(syncFullscreenControls);
    });
  }

  // Render shared image file card.
  function renderSharedImageFileCard(file) {
    return renderAttachmentCard(file, {
      side: 'assistant',
      source: 'shared_file',
      mode: 'message'
    });
  }

  // Render shared file card.
  function renderSharedFileCard(segment) {
    const file = sharedFileFromSegment(segment);
    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const fallbackPath = String(args.path || args.file || args.filename || '').trim();
    const normalized = file || normalizeSharedFilePayload({
      kind: 'shared_file',
      path: fallbackPath,
      filename: args.filename || fallbackPath
    });
    if (!normalized) {
      return renderReasoningToolRow(segment);
    }

    return renderAttachmentCard(normalized, {
      side: 'assistant',
      source: 'shared_file',
      mode: 'message'
    });
  }

  // Collect pinned shared file cards.
  function collectPinnedSharedFileCards(segments) {
    const safeSegments = Array.isArray(segments) ? segments : [];
    const cards = [];
    const seen = {};
    for (let index = 0; index < safeSegments.length; index += 1) {
      const segment = safeSegments[index];
      if (!segment || segment.type !== 'tool' || !isSharedFileToolSegment(segment)) {
        continue;
      }
      const file = sharedFileFromSegment(segment);
      const dedupeKey = file
        ? `${String(file.path || '').trim().toLowerCase()}::${String(file.filename || '').trim().toLowerCase()}`
        : `segment-${index}`;
      if (seen[dedupeKey]) {
        continue;
      }
      seen[dedupeKey] = true;
      cards.push(renderSharedFileCard(segment));
    }
    return cards.filter(function filterCard(card) {
      return !!String(card || '').trim();
    });
  }

  // Report whether image view tool segment.
  function isImageViewToolSegment(segment) {
    const identity = toolIdentityText(segment);
    if (/view[_-\s]?image|read[_-\s]?image/.test(identity)) {
      return true;
    }
    const result = parseToolResultObject(segment);
    return result && result.kind === 'image';
  }

  // Handle image result from segment.
  function imageResultFromSegment(segment) {
    const candidates = [
      segment && segment.structuredContent,
      segment && segment.toolUi && segment.toolUi.image,
      parseToolResultObject(segment)
    ];

    for (let index = 0; index < candidates.length; index += 1) {
      const result = candidates[index];
      if (!result || typeof result !== 'object') {
        continue;
      }
      const preview = result.preview && typeof result.preview === 'object' ? result.preview : null;
      if (result.kind === 'image' || (preview && preview.type === 'inline_base64')) {
        return result;
      }
    }
    return null;
  }

  // Handle sandbox image data url.
  function sandboxImageDataUrl(imageResult) {
    const preview = imageResult && imageResult.preview && typeof imageResult.preview === 'object'
      ? imageResult.preview
      : null;
    if (!preview || preview.type !== 'inline_base64') {
      return '';
    }

    let mime = String(preview.mime_type || imageResult.mime || 'image/png').trim().toLowerCase();
    if (!mime || mime === 'image') {
      mime = 'image/png';
    } else if (!mime.includes('/')) {
      mime = `image/${mime}`;
    }
    const dataBase64 = String(preview.data_base64 || '').replace(/\s+/g, '');
    if (!/^image\/[a-z0-9.+-]+$/i.test(mime) || !dataBase64) {
      return '';
    }
    return `data:${mime};base64,${dataBase64}`;
  }

  // Handle edit mode from segment.
  function editModeFromSegment(segment) {
    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const mode = String(args.mode || '').trim().toLowerCase();
    if (mode === 'lines' || mode === 'line' || (args.range !== undefined && String(args.range || '').trim())) {
      return 'lines';
    }
    return 'match';
  }

  // Handle edit path from segment.
  function editPathFromSegment(segment, result) {
    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    return String((result && (result.p || result.path)) || args.path || args.file || args.filename || '').trim();
  }

  // Parse unified diff rows.
  function parseUnifiedDiffRows(diffText) {
    const rows = [];
    const lines = String(diffText || '').split(/\r?\n/);
    let oldLine = null;
    let newLine = null;

    lines.forEach(function parseDiffLine(line) {
      if (!line || line.startsWith('--- ') || line.startsWith('+++ ')) {
        return;
      }

      const hunkMatch = /^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@/.exec(line);
      if (hunkMatch) {
        oldLine = parseInt(hunkMatch[1], 10);
        newLine = parseInt(hunkMatch[2], 10);
        rows.push({ type: 'hunk', oldNo: '', newNo: '', text: line });
        return;
      }

      if (oldLine === null || newLine === null) {
        return;
      }

      const marker = line.charAt(0);
      const text = line.slice(1);
      if (marker === '-') {
        rows.push({ type: 'delete', oldNo: oldLine, newNo: '', text });
        oldLine += 1;
        return;
      }
      if (marker === '+') {
        rows.push({ type: 'add', oldNo: '', newNo: newLine, text });
        newLine += 1;
        return;
      }
      if (marker === ' ') {
        rows.push({ type: 'context', oldNo: oldLine, newNo: newLine, text });
        oldLine += 1;
        newLine += 1;
      }
    });

    return rows;
  }

  // Handle fallback edit rows.
  function fallbackEditRows(segment) {
    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const mode = editModeFromSegment(segment);
    const rows = [];

    if (mode === 'lines') {
      const rangeLabel = args.range !== undefined ? String(Array.isArray(args.range) ? args.range.join(':') : args.range) : '';
      if (rangeLabel) {
        rows.push({ type: 'hunk', oldNo: '', newNo: '', text: `range ${rangeLabel}` });
      }
      String(args.content || args.new_str || '')
        .split(/\r?\n/)
        .forEach(function addLine(line, index) {
          rows.push({ type: 'add', oldNo: '', newNo: index + 1, text: line });
        });
      return rows;
    }

    String(args.old_str || '')
      .split(/\r?\n/)
      .forEach(function deleteLine(line, index) {
        rows.push({ type: 'delete', oldNo: index + 1, newNo: '', text: line });
      });
    String(args.new_str || '')
      .split(/\r?\n/)
      .forEach(function addLine(line, index) {
        rows.push({ type: 'add', oldNo: '', newNo: index + 1, text: line });
      });
    return rows;
  }

  // Handle edit rows from segment.
  function editRowsFromSegment(segment, result) {
    const rows = parseUnifiedDiffRows(result && result.ud ? result.ud : '');
    return rows.length > 0 ? rows : fallbackEditRows(segment);
  }

  // Render edit rows.
  function renderEditRows(rows, isExpanded, language) {
    const safeRows = Array.isArray(rows) ? rows : [];
    const maxRows = isExpanded ? EDIT_PREVIEW_EXPANDED_ROWS : EDIT_PREVIEW_COLLAPSED_ROWS;
    const visibleRows = safeRows.slice(0, maxRows);
    const codeLanguage = normalizeHighlightLanguage(language);
    const rowsHtml = visibleRows.map(function renderEditRow(row, index) {
      const isFadeLine = !isExpanded && index === EDIT_PREVIEW_COLLAPSED_ROWS - 1 && safeRows.length > EDIT_PREVIEW_COLLAPSED_ROWS;
      const rowClass = `msg-edit-row is-${row.type || 'context'}${isFadeLine ? ' msg-edit-row--fade' : ''}`;
      const lineNo = row.type === 'delete'
        ? row.oldNo
        : (row.newNo !== undefined && row.newNo !== '' ? row.newNo : row.oldNo);
      return `
        <span class="${rowClass}">
          <span class="msg-edit-gutter">${escHtml(lineNo === undefined ? '' : lineNo)}</span>
          <span class="msg-edit-code">${row.type === 'hunk' ? escHtml(row.text || ' ') : highlightCode(row.text || ' ', codeLanguage)}</span>
        </span>
      `;
    }).join('');
    if (isExpanded && safeRows.length > maxRows) {
      return `${rowsHtml}<span class="msg-edit-row is-context msg-edit-row--fade"><span class="msg-edit-gutter"></span><span class="msg-edit-code">... ${escHtml(String(safeRows.length - maxRows))} more rows omitted</span></span>`;
    }
    return rowsHtml;
  }

  // Render one edit-file tool activity card.
  function renderEditToolCard(segment, toolSegmentIndex, options) {
    const renderOptions = options || {};
    const result = parseToolResultObject(segment);
    const rows = editRowsFromSegment(segment, result);
    const isExpanded = !!renderOptions.expanded;
    const path = editPathFromSegment(segment, result);
    const mode = editModeFromSegment(segment);
    const language = languageFromPath(path, 'plaintext');
    const dataIndex = Number.isInteger(toolSegmentIndex) ? ` data-edit-segment-index="${toolSegmentIndex}"` : '';
    const label = path ? `Edit ${path}` : 'Edit';
    const summaryParts = [mode];
    if (result.r) {
      summaryParts.push(`range ${result.r}`);
    } else if (result.rep !== undefined) {
      summaryParts.push(`${result.rep} replaced`);
    }
    if (result.d !== undefined) {
      summaryParts.push(`${Number(result.d) >= 0 ? '+' : ''}${result.d} lines`);
    } else if (rows.length > EDIT_PREVIEW_EXPANDED_ROWS) {
      summaryParts.push(`${rows.length} rows`);
    }

    return `
      <div class="msg-edit-card${toolStatusClass(segment)}${isExpanded ? ' is-expanded' : ''}"${dataIndex} role="button" tabindex="0" aria-expanded="${isExpanded ? 'true' : 'false'}">
        <span class="msg-edit-head">
          ${icons.TOOL_EDIT_FILE_ICON || ''}
          <span class="msg-edit-title">${escHtml(label)}</span>
          <span class="msg-edit-summary">${escHtml(summaryParts.join(' · '))}</span>
        </span>
        <span class="msg-edit-preview msg-code-block language-${escapeAttributeValue(language)}" data-language="${escapeAttributeValue(language)}">${rows.length ? renderEditRows(rows, isExpanded, language) : '<span class="msg-edit-row is-context"><span class="msg-edit-gutter"></span><span class="msg-edit-code">No diff.</span></span>'}</span>
      </div>
    `;
  }

  // Handle truncate inline text.
  function truncateInlineText(value, maxLength) {
    const text = String(value || '').replace(/\s+/g, ' ').trim();
    const limit = maxLength || 140;
    return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
  }

  // Handle compact tool value.
  function compactToolValue(value) {
    if (value === null || value === undefined) {
      return '';
    }

    if (typeof value === 'string') {
      return truncateInlineText(value, 150);
    }

    try {
      return truncateInlineText(JSON.stringify(value), 150);
    } catch (_error) {
      return truncateInlineText(String(value), 150);
    }
  }

  // Handle utf8 byte length.
  function utf8ByteLength(value) {
    const text = String(value || '');
    if (typeof TextEncoder !== 'undefined') {
      return new TextEncoder().encode(text).length;
    }
    return unescape(encodeURIComponent(text)).length;
  }

  // Handle text line count.
  function textLineCount(value) {
    const text = String(value || '');
    return text ? text.split(/\r?\n/).length : 0;
  }

  // Handle truncate text preview.
  function truncateTextPreview(value, maxChars) {
    const text = String(value || '');
    const limit = Math.max(0, Number(maxChars) || 0);
    if (!limit || text.length <= limit) {
      return { text, truncated: false, omittedChars: 0 };
    }
    return {
      text: text.slice(0, limit),
      truncated: true,
      omittedChars: text.length - limit
    };
  }

  // Handle heavy tool keys for segment.
  function heavyToolKeysForSegment(segment) {
    if (isWriteToolSegment(segment)) {
      return HEAVY_TOOL_ARGUMENT_KEYS.write;
    }
    if (isEditToolSegment(segment)) {
      return HEAVY_TOOL_ARGUMENT_KEYS.edit;
    }
    if (isSandboxPythonToolSegment(segment)) {
      return HEAVY_TOOL_ARGUMENT_KEYS.bash;
    }
    if (isSandboxToolSegment(segment)) {
      return HEAVY_TOOL_ARGUMENT_KEYS.bash;
    }
    return [];
  }

  // Handle tool identity text.
  function toolIdentityText(segment) {
    const safeSegment = segment && typeof segment === 'object' ? segment : {};
    return [
      safeSegment.alias,
      safeSegment.serverId,
      safeSegment.serverName,
      safeSegment.toolId,
      safeSegment.toolName
    ].join(' ').toLowerCase();
  }

  // Handle tool server id.
  function toolServerId(segment) {
    return String(segment && segment.serverId || '').trim().toLowerCase();
  }

  // Report whether mcp sandbox tool segment.
  function isMcpSandboxToolSegment(segment) {
    const serverId = toolServerId(segment);
    if (serverId === 'sandbox') {
      return true;
    }
    const alias = String(segment && segment.alias || '').trim().toLowerCase();
    return alias.startsWith('sandbox__');
  }

  // Report whether sandbox python tool segment.
  function isSandboxPythonToolSegment(segment) {
    if (!isMcpSandboxToolSegment(segment)) {
      return false;
    }
    const toolId = String(segment && segment.toolId || '').trim().toLowerCase();
    return toolId === 'sandbox_python' || toolId === 'python';
  }

  // Report whether sandbox share file tool segment.
  function isSandboxShareFileToolSegment(segment) {
    if (!isMcpSandboxToolSegment(segment)) {
      return false;
    }
    const toolId = String(segment && segment.toolId || '').trim().toLowerCase();
    return toolId === 'share_file';
  }

  // Report whether sandbox tool segment.
  function isSandboxToolSegment(segment) {
    if (isMcpSandboxToolSegment(segment)) {
      const toolId = String(segment && segment.toolId || '').trim().toLowerCase();
      if (['bash', 'write', 'edit', 'view_image', 'share_file', 'sandbox_python', 'python'].includes(toolId)) {
        return true;
      }
    }
    return /sandbox|bash|shell|exec|deep[-_\s]?think|container/.test(toolIdentityText(segment));
  }

  // Parse sandbox result.
  function parseSandboxResult(segment) {
    const rawResult = segment && segment.result !== null && segment.result !== undefined
      ? String(segment.result)
      : '';
    if (!rawResult) {
      return {
        ok: null,
        exitCode: null,
        stdout: '',
        stderr: '',
        raw: ''
      };
    }

    try {
      const parsed = JSON.parse(rawResult);
      const envelope = parsed && typeof parsed === 'object' ? parsed : {};
      const result = envelope.result && typeof envelope.result === 'object' ? envelope.result : envelope;
      const error = envelope.error && typeof envelope.error === 'object' ? envelope.error : null;
      return {
        ok: envelope.ok === undefined ? null : !!envelope.ok,
        exitCode: result.exit_code !== undefined && result.exit_code !== null ? result.exit_code : null,
        stdout: result.stdout !== undefined && result.stdout !== null ? String(result.stdout) : '',
        stderr: result.stderr !== undefined && result.stderr !== null
          ? String(result.stderr)
          : (error && error.message ? String(error.message) : ''),
        raw: rawResult
      };
    } catch (_error) {
      return {
        ok: null,
        exitCode: null,
        stdout: rawResult,
        stderr: '',
        raw: rawResult
      };
    }
  }

  // Handle sandbox input text.
  function sandboxInputText(segment) {
    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const command = args.command || args.cmd || args.code || args.input || '';
    const stdin = args.stdin !== undefined && args.stdin !== null ? String(args.stdin) : '';
    if (stdin) {
      return `${command}\n\n${stdin}`;
    }
    return String(command || '');
  }

  // Handle sandbox input preview text.
  function sandboxInputPreviewText(segment) {
    const preview = truncateTextPreview(sandboxInputText(segment), SANDBOX_INPUT_PREVIEW_CHARS);
    if (!preview.truncated) {
      return preview.text;
    }
    return `${preview.text}\n\n... ${preview.omittedChars} more characters omitted`;
  }

  // Return the short model-provided intent for a sandbox command.
  function sandboxHeaderDetail(segment) {
    const args = segment && segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const description = String(args.description || '').trim().replace(/\s+/g, ' ');
    if (description) {
      return description.split(' ').slice(0, 4).join(' ');
    }
    return reasoningToolDetail(segment);
  }

  // Return the complete model-provided description for a compact sandbox call.
  function sandboxDescription(segment) {
    const args = segment && segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    return String(args.description || '').trim().replace(/\s+/g, ' ');
  }

  // Handle sandbox language.
  function sandboxLanguage(segment) {
    const identity = toolIdentityText(segment);
    if (/python/.test(identity)) {
      return 'python';
    }
    if (/bash|shell|exec|sandbox|container/.test(identity)) {
      return 'bash';
    }
    return 'plaintext';
  }

  // Render sandbox stream block.
  function renderSandboxStreamBlock(label, content, streamClass, language) {
    const text = String(content || '');
    if (!text) {
      return '';
    }

    const safeLanguage = String(language || 'plaintext').replace(/[^a-z0-9_-]/gi, '') || 'plaintext';
    return `
      <div class="msg-sandbox-section ${streamClass}">
        <div class="msg-sandbox-section-label">${escHtml(label)}</div>
        <pre class="msg-sandbox-pre msg-code-block language-${safeLanguage}" data-language="${escapeAttributeValue(safeLanguage)}"><code>${highlightCode(text, safeLanguage)}</code></pre>
      </div>
    `;
  }

  // Handle sandbox status class.
  function sandboxStatusClass(segment, result) {
    if (result && (result.ok === false || (result.exitCode !== null && result.exitCode !== undefined && Number(result.exitCode) !== 0))) {
      return ' is-error';
    }
    return toolStatusClass(segment);
  }

  // Render sandbox image tool block.
  function renderSandboxImageToolBlock(segment, toolSegmentIndex) {
    const image = imageResultFromSegment(segment);
    if (!image) {
      return '';
    }

    const status = toolStatusText(segment);
    const dataIndex = Number.isInteger(toolSegmentIndex) ? ` data-tool-segment-index="${toolSegmentIndex}"` : '';
    const path = String(image.path || reasoningToolDetail(segment) || '').trim();
    const mime = String(image.mime || (image.preview && image.preview.mime_type) || '').trim();
    const src = sandboxImageDataUrl(image);
    const width = Number(image.width);
    const height = Number(image.height);
    const hasDimensions = Number.isFinite(width) && width > 0 && Number.isFinite(height) && height > 0;
    const metaParts = [
      hasDimensions ? `${Math.round(width)}x${Math.round(height)}` : '',
      mime,
      formatByteSize(image.size_bytes)
    ].filter(Boolean);
    const title = path ? `Viewed ${path}` : 'Viewed image';
    const altText = path ? `Image preview for ${path}` : 'Image preview';
    const preview = image.preview && typeof image.preview === 'object' ? image.preview : {};
    const previewUnavailableClass = src ? '' : ' is-preview-unavailable';
    const previewWithheldClass = preview.type === 'text_placeholder' || (image.vision_gate && image.vision_gate.allowed === false)
      ? ' is-preview-withheld'
      : '';
    const imageLoadState = src ? (sandboxImageLoadStateBySrc.get(src) || '') : '';
    const imageLoadedClass = src && (loadedSandboxImageSrcs.has(src) || imageLoadState === 'loaded') ? ' is-loaded' : '';
    const imageErrorClass = imageLoadState === 'error' ? ' is-load-error' : '';
    const imageHtml = src
      ? `<div class="msg-sandbox-image-loader" aria-hidden="true"></div><img class="msg-sandbox-image" src="${escapeAttributeValue(src)}" alt="${escapeAttributeValue(altText)}" loading="lazy" data-sandbox-image-src="${escapeAttributeValue(src)}">`
      : '<div class="msg-sandbox-image-empty">Preview unavailable</div>';

    return `
      <div class="msg-sandbox-card msg-sandbox-image-card${previewUnavailableClass}${previewWithheldClass}${toolStatusClass(segment)}"${dataIndex}>
        <div class="msg-sandbox-head">
          ${icons.TOOL_IMAGE_VIEW_ICON || icons.TOOL_BASH_ICON || '<span class="msg-reasoning-tool-icon is-sandbox" aria-hidden="true">S</span>'}
          <span class="msg-sandbox-title">Image</span>
          <span class="msg-sandbox-detail">${escHtml(title)}</span>
          <span class="msg-reasoning-tool-status">${escHtml(status)}</span>
        </div>
        <div class="msg-sandbox-image-frame${src ? ' is-loading' : ''}${imageLoadedClass}${imageErrorClass}">${imageHtml}</div>
        ${metaParts.length ? `<div class="msg-sandbox-image-meta">${escHtml(metaParts.join(' В· '))}</div>` : ''}
      </div>
    `;
  }

  // Render sandbox tool block.
  function renderSandboxToolBlock(segment, toolSegmentIndex) {
    const status = toolStatusText(segment);
    const detail = sandboxHeaderDetail(segment);
    const result = parseSandboxResult(segment);
    const language = sandboxLanguage(segment);
    const inputText = sandboxInputPreviewText(segment) || detail;
    const hasResult = segment.result !== null && segment.result !== undefined;
    const outputText = result.stdout || (!hasResult ? 'Running...' : '');
    const exitCodeText = result.exitCode !== null && result.exitCode !== undefined ? `exit ${result.exitCode}` : status;
    const dataIndex = Number.isInteger(toolSegmentIndex) ? ` data-tool-segment-index="${toolSegmentIndex}"` : '';
    const stderrHtml = renderSandboxStreamBlock('stderr', result.stderr, 'is-stderr', 'plaintext');
    const stdoutHtml = renderSandboxStreamBlock('stdout', outputText, 'is-stdout', 'plaintext');

    if (result.ok !== false && isImageViewToolSegment(segment)) {
      const imageHtml = renderSandboxImageToolBlock(segment, toolSegmentIndex);
      if (imageHtml) {
        return imageHtml;
      }
    }

    return `
      <div class="msg-sandbox-card${sandboxStatusClass(segment, result)}"${dataIndex}>
        <div class="msg-sandbox-head">
          ${icons.TOOL_BASH_ICON || '<span class="msg-reasoning-tool-icon is-sandbox" aria-hidden="true">S</span>'}
          <span class="msg-sandbox-title">Sandbox</span>
          ${detail ? `<span class="msg-sandbox-detail msg-sandbox-command-description">${escHtml(detail)}</span>` : ''}
          <span class="msg-reasoning-tool-status">${escHtml(exitCodeText)}</span>
        </div>
        ${renderSandboxStreamBlock('stdin', inputText, 'is-stdin', language)}
        ${stdoutHtml || stderrHtml ? `${stdoutHtml}${stderrHtml}` : renderSandboxStreamBlock('stdout', 'No output.', 'is-stdout is-empty', 'plaintext')}
      </div>
    `;
  }

  // Render a sandbox call as a clean, collapsed line outside reasoning mode.
  function renderCollapsedSandboxToolBlock(segment, toolSegmentIndex, isExpanded) {
    const description = sandboxDescription(segment) || 'Sandbox';
    const dataIndex = Number.isInteger(toolSegmentIndex) ? ` data-sandbox-segment-index="${toolSegmentIndex}"` : '';
    return `
      <div class="msg-sandbox-collapsible${isExpanded ? ' is-expanded' : ''}"${dataIndex}>
        <button type="button" class="msg-sandbox-toggle" aria-expanded="${isExpanded ? 'true' : 'false'}">
          <span class="msg-sandbox-toggle-label">${escHtml(description)}</span>
          <span class="msg-sandbox-toggle-chevron" aria-hidden="true">&gt;</span>
        </button>
        <div class="msg-sandbox-collapsible-content"${isExpanded ? '' : ' hidden'}>
          ${renderSandboxToolBlock(segment, toolSegmentIndex)}
        </div>
      </div>
    `;
  }

  // Handle remember loaded sandbox images.
  function rememberLoadedSandboxImages($root) {
    const root = $root && $root.length ? $root : dom.$messagesInner;
    root.find('.msg-sandbox-image-frame.is-loaded .msg-sandbox-image[data-sandbox-image-src]').each(function rememberImage() {
      const src = String(this.getAttribute('data-sandbox-image-src') || '');
      if (src) {
        loadedSandboxImageSrcs.add(src);
        sandboxImageLoadStateBySrc.set(src, 'loaded');
      }
    });
  }

  // Sync sandbox image frames for src.
  function syncSandboxImageFramesForSrc(src) {
    const normalizedSrc = String(src || '').trim();
    if (!normalizedSrc) {
      return;
    }

    const stateValue = sandboxImageLoadStateBySrc.get(normalizedSrc) || '';
    if (!stateValue) {
      return;
    }

    const $roots = dom.$messagesInner.add($('#reasoningDrawerBody'));
    $roots.find('.msg-sandbox-image[data-sandbox-image-src]').each(function syncFrame() {
      const imageSrc = String(this.getAttribute('data-sandbox-image-src') || this.currentSrc || this.src || '');
      if (imageSrc !== normalizedSrc) {
        return;
      }

      const frame = this.closest('.msg-sandbox-image-frame');
      if (!frame) {
        return;
      }

      if (stateValue === 'loaded') {
        frame.classList.add('is-loaded');
        frame.classList.remove('is-load-error');
      } else if (stateValue === 'error') {
        frame.classList.add('is-load-error');
      }
    });
  }

  // Handle ensure sandbox image preload.
  function ensureSandboxImagePreload(src) {
    const normalizedSrc = String(src || '').trim();
    if (!normalizedSrc) {
      return;
    }

    const knownState = sandboxImageLoadStateBySrc.get(normalizedSrc);
    if (knownState === 'loading' || knownState === 'loaded' || knownState === 'error') {
      return;
    }

    sandboxImageLoadStateBySrc.set(normalizedSrc, 'loading');
    const preloadImage = new window.Image();
    preloadImage.onload = function onPreloadLoad() {
      loadedSandboxImageSrcs.add(normalizedSrc);
      sandboxImageLoadStateBySrc.set(normalizedSrc, 'loaded');
      syncSandboxImageFramesForSrc(normalizedSrc);
    };
    preloadImage.onerror = function onPreloadError() {
      sandboxImageLoadStateBySrc.set(normalizedSrc, 'error');
      syncSandboxImageFramesForSrc(normalizedSrc);
    };
    preloadImage.src = normalizedSrc;
  }

  // Handle mark sandbox image loaded.
  function markSandboxImageLoaded(imageEl) {
    if (!imageEl) {
      return;
    }
    const src = String(imageEl.getAttribute('data-sandbox-image-src') || imageEl.currentSrc || imageEl.src || '');
    if (src) {
      loadedSandboxImageSrcs.add(src);
      sandboxImageLoadStateBySrc.set(src, 'loaded');
    }
    const frame = imageEl.closest('.msg-sandbox-image-frame');
    if (frame) {
      frame.classList.add('is-loaded');
      frame.classList.remove('is-load-error');
    }
  }

  // Handle mark sandbox image error.
  function markSandboxImageError(imageEl) {
    const src = String((imageEl && imageEl.getAttribute && imageEl.getAttribute('data-sandbox-image-src')) || (imageEl && imageEl.currentSrc) || (imageEl && imageEl.src) || '');
    if (src) {
      sandboxImageLoadStateBySrc.set(src, 'error');
    }
    const frame = imageEl && imageEl.closest ? imageEl.closest('.msg-sandbox-image-frame') : null;
    if (frame) {
      frame.classList.add('is-load-error');
    }
  }

  // Handle hydrate sandbox images.
  function hydrateSandboxImages($root) {
    const root = $root && $root.length ? $root : dom.$messagesInner;
    root.find('.msg-sandbox-image[data-sandbox-image-src]').each(function hydrateImage() {
      const imageEl = this;
      const src = String(imageEl.getAttribute('data-sandbox-image-src') || imageEl.currentSrc || imageEl.src || '');
      const stateValue = src ? (sandboxImageLoadStateBySrc.get(src) || '') : '';
      if (stateValue === 'loaded') {
        markSandboxImageLoaded(imageEl);
        return;
      }
      if (stateValue === 'error') {
        markSandboxImageError(imageEl);
        return;
      }
      if (src && loadedSandboxImageSrcs.has(src)) {
        markSandboxImageLoaded(imageEl);
        return;
      }
      if (src) {
        ensureSandboxImagePreload(src);
      }
      if (imageEl.complete && imageEl.naturalWidth > 0) {
        markSandboxImageLoaded(imageEl);
        return;
      }
      if (imageEl.dataset.sandboxLoadBound === '1') {
        return;
      }
      imageEl.dataset.sandboxLoadBound = '1';
      imageEl.addEventListener('load', function onSandboxImageLoad() {
        markSandboxImageLoaded(imageEl);
      }, { once: true });
      imageEl.addEventListener('error', function onSandboxImageError() {
        markSandboxImageError(imageEl);
      }, { once: true });
    });
  }

  // Handle hydrate shared image cards.
  function hydrateSharedImageCards($root) {
    const root = $root && $root.length ? $root : dom.$messagesInner;
    root.find('.msg-shared-image').each(function hydrateSharedImage() {
      const imageEl = this;
      if (imageEl.dataset.sharedImageHydrated === '1') {
        return;
      }
      imageEl.dataset.sharedImageHydrated = '1';

      const applyDimensions = function applyDimensions() {
        const card = imageEl.closest('.msg-shared-image-card');
        const details = card ? card.querySelector('.msg-shared-image-details') : null;
        const width = Number(imageEl.naturalWidth);
        const height = Number(imageEl.naturalHeight);
        if (!details || !Number.isFinite(width) || width <= 0 || !Number.isFinite(height) || height <= 0) {
          return;
        }
        const base = String(details.dataset.sharedImageDetails || details.textContent || '').trim();
        const dimensionText = `${Math.round(width)}x${Math.round(height)}`;
        if (base.includes(dimensionText)) {
          return;
        }
        details.textContent = base ? `${base} / ${dimensionText}` : dimensionText;
      };

      if (imageEl.complete) {
        applyDimensions();
      } else {
        imageEl.addEventListener('load', applyDimensions, { once: true });
      }
    });
  }

  // Handle sanitize mermaid svg.
  function sanitizeMermaidSvg(svg) {
    if (typeof DOMPurify !== 'undefined' && DOMPurify.sanitize) {
      return DOMPurify.sanitize(String(svg || ''), {
        USE_PROFILES: { svg: true, svgFilters: true }
      });
    }
    return String(svg || '');
  }

  // Handle read root css var.
  function readRootCssVar(name, fallback) {
    if (typeof document === 'undefined' || !document.documentElement) {
      return fallback;
    }
    var raw = window.getComputedStyle(document.documentElement).getPropertyValue(name);
    raw = (raw || '').trim();
    return raw || fallback;
  }

  // Build mermaid theme variables.
  function buildMermaidThemeVariables() {
    return {
      background: 'transparent',
      mainBkg: readRootCssVar('--surface-secondary', '#202326'),
      primaryColor: readRootCssVar('--surface-secondary', '#202326'),
      primaryTextColor: readRootCssVar('--c-text', '#f4f5f6'),
      primaryBorderColor: readRootCssVar('--c-border', '#5f6b76'),
      nodeTextColor: readRootCssVar('--c-text', '#f4f5f6'),
      labelTextColor: readRootCssVar('--c-text', '#f4f5f6'),
      edgeLabelBackground: readRootCssVar('--surface-tertiary', '#343638'),
      lineColor: readRootCssVar('--c-gray-1', '#8fa3b8'),
      secondaryColor: readRootCssVar('--surface-tertiary', '#253241'),
      tertiaryColor: readRootCssVar('--surface-primary', '#17201b'),
      fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif'
    };
  }

  // Parse css color to rgb.
  function parseCssColorToRgb(value) {
    const color = String(value || '').trim().toLowerCase();
    if (!color || color === 'none' || color === 'transparent') {
      return null;
    }

    const hex = color.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      const raw = hex[1];
      const expanded = raw.length === 3
        ? raw.split('').map(function expandHex(part) { return part + part; }).join('')
        : raw;
      return {
        r: parseInt(expanded.slice(0, 2), 16),
        g: parseInt(expanded.slice(2, 4), 16),
        b: parseInt(expanded.slice(4, 6), 16)
      };
    }

    const rgb = color.match(/^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i);
    if (rgb) {
      return {
        r: Number(rgb[1]),
        g: Number(rgb[2]),
        b: Number(rgb[3])
      };
    }

    return null;
  }

  // Handle relative luminance.
  function relativeLuminance(rgb) {
    if (!rgb) {
      return 0;
    }
    const channels = [rgb.r, rgb.g, rgb.b].map(function normalizeChannel(value) {
      const channel = Math.max(0, Math.min(255, Number(value) || 0)) / 255;
      return channel <= 0.03928
        ? channel / 12.92
        : Math.pow((channel + 0.055) / 1.055, 2.4);
    });
    return (0.2126 * channels[0]) + (0.7152 * channels[1]) + (0.0722 * channels[2]);
  }

  // Handle contrast ratio.
  function contrastRatio(firstRgb, secondRgb) {
    const first = relativeLuminance(firstRgb);
    const second = relativeLuminance(secondRgb);
    const lighter = Math.max(first, second);
    const darker = Math.min(first, second);
    return (lighter + 0.05) / (darker + 0.05);
  }

  // Read svg shape fill from app state.
  function getSvgShapeFill(shapeEl) {
    if (!shapeEl) {
      return null;
    }
    const computedFill = typeof window !== 'undefined' && window.getComputedStyle
      ? window.getComputedStyle(shapeEl).fill
      : '';
    return parseCssColorToRgb(computedFill)
      || parseCssColorToRgb(shapeEl.getAttribute('fill'))
      || parseCssColorToRgb((shapeEl.getAttribute('style') || '').match(/fill\s*:\s*([^;]+)/i)?.[1]);
  }

  // Handle apply mermaid label contrast.
  function applyMermaidLabelContrast(svgRoot) {
    if (!svgRoot || !svgRoot.querySelectorAll) {
      return;
    }

    svgRoot.querySelectorAll('g.node').forEach(function applyNodeLabelContrast(nodeEl) {
      const shapeEl = nodeEl.querySelector('rect, polygon, circle, ellipse, path');
      const fill = getSvgShapeFill(shapeEl);
      if (!fill) {
        return;
      }

      const darkText = parseCssColorToRgb(readRootCssVar('--surface-secondary', '#202326'))
        || { r: 32, g: 35, b: 38 };
      const lightText = parseCssColorToRgb(readRootCssVar('--c-text', '#f4f5f6'))
        || { r: 244, g: 245, b: 246 };
      const textColor = contrastRatio(fill, darkText) >= contrastRatio(fill, lightText)
        ? readRootCssVar('--surface-secondary', '#202326')
        : readRootCssVar('--c-text', '#f4f5f6');
      nodeEl.querySelectorAll('text, tspan').forEach(function applySvgTextColor(textEl) {
        textEl.setAttribute('fill', textColor);
        textEl.style.fill = textColor;
      });
      nodeEl.querySelectorAll('.nodeLabel, .md-mermaid-katex-label').forEach(function applyHtmlTextColor(labelEl) {
        labelEl.style.color = textColor;
      });
    });
  }

  // Render mermaid latex labels.
  function renderMermaidLatexLabels(svgRoot) {
    if (!svgRoot || !svgRoot.querySelectorAll || typeof document === 'undefined') {
      return;
    }

    svgRoot.querySelectorAll('text').forEach(function replaceLatexSvgText(textEl) {
      const source = String(textEl.textContent || '').trim();
      if (!source || !/(?:\$|\\\[|\\\()/.test(source) || textEl.closest('foreignObject')) {
        return;
      }

      let bbox = null;
      try {
        bbox = textEl.getBBox();
      } catch (error) {
        bbox = null;
      }
      if (!bbox || !Number.isFinite(bbox.x) || !Number.isFinite(bbox.y)) {
        return;
      }

      const parentEl = textEl.parentNode;
      if (!parentEl) {
        return;
      }

      const textAnchor = String(textEl.getAttribute('text-anchor') || '').toLowerCase();
      const width = Math.max(bbox.width + 40, 80);
      const height = Math.max(bbox.height + 18, 30);
      const centerX = bbox.x + (bbox.width / 2);
      let x = centerX - (width / 2);
      if (textAnchor === 'start') {
        x = bbox.x - 8;
      } else if (textAnchor === 'end') {
        x = bbox.x + bbox.width - width + 8;
      }

      const foreignObject = document.createElementNS('http://www.w3.org/2000/svg', 'foreignObject');
      foreignObject.classList.add('md-mermaid-katex-foreign');
      foreignObject.setAttribute('x', String(x));
      foreignObject.setAttribute('y', String(bbox.y - 8));
      foreignObject.setAttribute('width', String(width));
      foreignObject.setAttribute('height', String(height));
      foreignObject.setAttribute('requiredExtensions', 'http://www.w3.org/1999/xhtml');

      const labelEl = document.createElement('div');
      labelEl.setAttribute('xmlns', 'http://www.w3.org/1999/xhtml');
      labelEl.className = 'md-mermaid-katex-label';
      labelEl.innerHTML = renderLatexInHtml(escHtml(source));

      foreignObject.appendChild(labelEl);
      parentEl.insertBefore(foreignObject, textEl.nextSibling);
      textEl.setAttribute('aria-hidden', 'true');
      textEl.style.display = 'none';
    });
  }

  // Handle enhance mermaid svg.
  function enhanceMermaidSvg(canvasEl) {
    if (!canvasEl || !canvasEl.querySelector) {
      return;
    }
    const svgRoot = canvasEl.querySelector('svg');
    if (!svgRoot) {
      return;
    }
    renderMermaidLatexLabels(svgRoot);
    applyMermaidLabelContrast(svgRoot);
  }

  // Read the intrinsic Mermaid SVG size used by pan, zoom, and PNG export.
  function mermaidSvgSize(svgRoot) {
    if (!svgRoot) {
      return null;
    }

    const viewBox = svgRoot.viewBox && svgRoot.viewBox.baseVal;
    if (viewBox && Number.isFinite(viewBox.width) && viewBox.width > 0
      && Number.isFinite(viewBox.height) && viewBox.height > 0) {
      return { width: viewBox.width, height: viewBox.height };
    }

    const rawViewBox = String(svgRoot.getAttribute('viewBox') || '').trim().split(/[\s,]+/).map(Number);
    if (rawViewBox.length === 4 && rawViewBox.every(Number.isFinite) && rawViewBox[2] > 0 && rawViewBox[3] > 0) {
      return { width: rawViewBox[2], height: rawViewBox[3] };
    }

    try {
      const bounds = svgRoot.getBBox();
      if (bounds && bounds.width > 0 && bounds.height > 0) {
        return { width: bounds.width, height: bounds.height };
      }
    } catch (_error) {
      // Ignore SVG implementations without getBBox support.
    }

    return null;
  }

  // Apply the current viewport matrix and keep the zoom indicator in sync.
  function applyMermaidViewportTransform(canvasEl) {
    const viewport = canvasEl && canvasEl._mermaidViewport;
    if (!viewport) {
      return;
    }
    viewport.stage.style.transform = `translate3d(${viewport.x}px, ${viewport.y}px, 0) scale(${viewport.scale})`;
    if (viewport.zoomValue) {
      viewport.zoomValue.textContent = `${Math.round(viewport.scale * 100)}%`;
    }
  }

  // Fit the full diagram into its viewport.
  function fitMermaidViewport(canvasEl) {
    const viewport = canvasEl && canvasEl._mermaidViewport;
    if (!viewport || !canvasEl.clientWidth || !canvasEl.clientHeight) {
      return;
    }
    const padding = canvasEl.clientWidth < 520 ? 28 : 44;
    const availableWidth = Math.max(1, canvasEl.clientWidth - (padding * 2));
    const availableHeight = Math.max(1, canvasEl.clientHeight - (padding * 2));
    viewport.scale = Math.max(
      viewport.minScale,
      Math.min(viewport.maxScale, availableWidth / viewport.width, availableHeight / viewport.height)
    );
    viewport.x = (canvasEl.clientWidth - (viewport.width * viewport.scale)) / 2;
    viewport.y = (canvasEl.clientHeight - (viewport.height * viewport.scale)) / 2;
    viewport.hasInteracted = false;
    applyMermaidViewportTransform(canvasEl);
  }

  // Zoom around one viewport point, preserving the content below the pointer.
  function zoomMermaidViewport(canvasEl, factor, clientX, clientY) {
    const viewport = canvasEl && canvasEl._mermaidViewport;
    if (!viewport) {
      return;
    }
    const nextScale = Math.max(viewport.minScale, Math.min(viewport.maxScale, viewport.scale * factor));
    if (Math.abs(nextScale - viewport.scale) < 0.001) {
      return;
    }
    const bounds = canvasEl.getBoundingClientRect();
    const pointX = Number.isFinite(clientX) ? clientX - bounds.left : bounds.width / 2;
    const pointY = Number.isFinite(clientY) ? clientY - bounds.top : bounds.height / 2;
    const diagramX = (pointX - viewport.x) / viewport.scale;
    const diagramY = (pointY - viewport.y) / viewport.scale;
    viewport.x = pointX - (diagramX * nextScale);
    viewport.y = pointY - (diagramY * nextScale);
    viewport.scale = nextScale;
    viewport.hasInteracted = true;
    applyMermaidViewportTransform(canvasEl);
  }

  // Draw the thematic grid used behind Mermaid diagrams into an export canvas.
  function drawMermaidExportBackground(context, canvasEl, width, height) {
    const styles = window.getComputedStyle(canvasEl);
    context.fillStyle = styles.backgroundColor || readRootCssVar('--surface-secondary', '#202326');
    context.fillRect(0, 0, width, height);
    context.save();
    context.globalAlpha = 0.34;
    context.strokeStyle = readRootCssVar('--c-border', '#5f6b76');
    context.lineWidth = 1;
    for (let x = 0.5; x < width; x += 24) {
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
    }
    for (let y = 0.5; y < height; y += 24) {
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(width, y);
      context.stroke();
    }
    context.restore();
  }

  // Export the complete diagram (independent of current pan/zoom) as a PNG.
  function downloadMermaidPng(canvasEl, buttonEl) {
    const viewport = canvasEl && canvasEl._mermaidViewport;
    if (!viewport || !viewport.svg) {
      return;
    }

    const padding = 32;
    const logicalWidth = viewport.width + (padding * 2);
    const logicalHeight = viewport.height + (padding * 2);
    const exportScale = Math.max(0.25, Math.min(2, 4096 / logicalWidth, 4096 / logicalHeight));
    const exportCanvas = document.createElement('canvas');
    exportCanvas.width = Math.max(1, Math.round(logicalWidth * exportScale));
    exportCanvas.height = Math.max(1, Math.round(logicalHeight * exportScale));
    const context = exportCanvas.getContext('2d');
    if (!context) {
      return;
    }
    context.scale(exportScale, exportScale);
    drawMermaidExportBackground(context, canvasEl, logicalWidth, logicalHeight);

    const svgClone = viewport.svg.cloneNode(true);
    svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    svgClone.setAttribute('width', String(viewport.width));
    svgClone.setAttribute('height', String(viewport.height));
    svgClone.style.maxWidth = 'none';
    svgClone.style.width = `${viewport.width}px`;
    svgClone.style.height = `${viewport.height}px`;
    const exportStyles = document.createElementNS('http://www.w3.org/2000/svg', 'style');
    exportStyles.textContent = '.md-mermaid-katex-label{display:flex;align-items:center;justify-content:center;width:100%;height:100%;font-family:Inter,Arial,sans-serif;font-weight:600;text-align:center}.md-mermaid-katex-label .katex{font-size:1em}';
    svgClone.insertBefore(exportStyles, svgClone.firstChild);

    const svgBlob = new Blob([new XMLSerializer().serializeToString(svgClone)], { type: 'image/svg+xml;charset=utf-8' });
    const svgUrl = URL.createObjectURL(svgBlob);
    const image = new Image();
    if (buttonEl) {
      buttonEl.disabled = true;
      buttonEl.classList.add('is-busy');
      buttonEl.dataset.mermaidExportState = 'working';
    }

    function finishExport(state) {
      URL.revokeObjectURL(svgUrl);
      if (buttonEl) {
        buttonEl.disabled = false;
        buttonEl.classList.remove('is-busy');
        buttonEl.dataset.mermaidExportState = state || 'error';
      }
    }

    image.onload = function onMermaidExportLoaded() {
      try {
        context.drawImage(image, padding, padding, viewport.width, viewport.height);
        exportCanvas.toBlob(function saveMermaidPng(pngBlob) {
          if (!pngBlob) {
            finishExport('error');
            return;
          }
          const pngUrl = URL.createObjectURL(pngBlob);
          const link = document.createElement('a');
          link.href = pngUrl;
          link.download = `mermaid-diagram-${Date.now()}.png`;
          document.body.appendChild(link);
          link.click();
          link.remove();
          setTimeout(function revokeMermaidPngUrl() { URL.revokeObjectURL(pngUrl); }, 1000);
          finishExport('ready');
        }, 'image/png');
      } catch (_error) {
        finishExport('error');
      }
    };
    image.onerror = function onMermaidExportError() { finishExport('error'); };
    image.src = svgUrl;
  }

  // Wrap a rendered Mermaid SVG in the interactive pan/zoom viewport.
  function initializeMermaidViewport(canvasEl) {
    if (!canvasEl || canvasEl._mermaidViewport) {
      return;
    }
    const svgRoot = canvasEl.querySelector('svg');
    const size = mermaidSvgSize(svgRoot);
    if (!svgRoot || !size) {
      return;
    }

    const stage = document.createElement('div');
    stage.className = 'md-mermaid-stage';
    stage.style.width = `${size.width}px`;
    stage.style.height = `${size.height}px`;
    svgRoot.setAttribute('width', String(size.width));
    svgRoot.setAttribute('height', String(size.height));
    svgRoot.parentNode.insertBefore(stage, svgRoot);
    stage.appendChild(svgRoot);

    const controls = document.createElement('div');
    controls.className = 'md-mermaid-controls';
    controls.setAttribute('role', 'toolbar');
    controls.setAttribute('aria-label', 'Diagram controls');
    controls.innerHTML = `
      <button type="button" class="md-mermaid-control" data-mermaid-action="zoom-out" title="Zoom out" aria-label="Zoom out">${MERMAID_ZOOM_OUT_ICON}</button>
      <span class="md-mermaid-zoom-value" aria-live="polite">100%</span>
      <button type="button" class="md-mermaid-control" data-mermaid-action="zoom-in" title="Zoom in" aria-label="Zoom in">${MERMAID_ZOOM_IN_ICON}</button>
      <span class="md-mermaid-control-separator" aria-hidden="true"></span>
      <button type="button" class="md-mermaid-control" data-mermaid-action="fit" title="Fit diagram" aria-label="Fit diagram">${MERMAID_FIT_ICON}</button>
      <button type="button" class="md-mermaid-control" data-mermaid-action="download" title="Download PNG" aria-label="Download diagram as PNG">${MERMAID_DOWNLOAD_ICON}</button>
    `;
    canvasEl.appendChild(controls);

    const viewport = {
      svg: svgRoot,
      stage,
      controls,
      zoomValue: controls.querySelector('.md-mermaid-zoom-value'),
      width: size.width,
      height: size.height,
      scale: 1,
      minScale: 0.2,
      maxScale: 5,
      x: 0,
      y: 0,
      hasInteracted: false,
      activePointerId: null,
      pointerX: 0,
      pointerY: 0
    };
    canvasEl._mermaidViewport = viewport;

    controls.addEventListener('click', function onMermaidControlClick(event) {
      const button = event.target.closest('[data-mermaid-action]');
      if (!button) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      const action = button.dataset.mermaidAction;
      if (action === 'zoom-in') {
        zoomMermaidViewport(canvasEl, 1.2);
      } else if (action === 'zoom-out') {
        zoomMermaidViewport(canvasEl, 1 / 1.2);
      } else if (action === 'fit') {
        fitMermaidViewport(canvasEl);
      } else if (action === 'download') {
        downloadMermaidPng(canvasEl, button);
      }
    });

    canvasEl.addEventListener('wheel', function onMermaidWheel(event) {
      event.preventDefault();
      zoomMermaidViewport(canvasEl, Math.exp(-event.deltaY * 0.0015), event.clientX, event.clientY);
    }, { passive: false });

    canvasEl.addEventListener('pointerdown', function onMermaidPointerDown(event) {
      if (event.button !== 0 || event.target.closest('.md-mermaid-controls')) {
        return;
      }
      event.preventDefault();
      viewport.activePointerId = event.pointerId;
      viewport.pointerX = event.clientX;
      viewport.pointerY = event.clientY;
      canvasEl.classList.add('is-panning');
      canvasEl.setPointerCapture(event.pointerId);
    });

    canvasEl.addEventListener('pointermove', function onMermaidPointerMove(event) {
      if (viewport.activePointerId !== event.pointerId) {
        return;
      }
      viewport.x += event.clientX - viewport.pointerX;
      viewport.y += event.clientY - viewport.pointerY;
      viewport.pointerX = event.clientX;
      viewport.pointerY = event.clientY;
      viewport.hasInteracted = true;
      applyMermaidViewportTransform(canvasEl);
    });

    function stopMermaidPan(event) {
      if (viewport.activePointerId !== event.pointerId) {
        return;
      }
      viewport.activePointerId = null;
      canvasEl.classList.remove('is-panning');
      if (canvasEl.hasPointerCapture && canvasEl.hasPointerCapture(event.pointerId)) {
        canvasEl.releasePointerCapture(event.pointerId);
      }
    }
    canvasEl.addEventListener('pointerup', stopMermaidPan);
    canvasEl.addEventListener('pointercancel', stopMermaidPan);
    canvasEl.addEventListener('dblclick', function onMermaidDoubleClick(event) {
      if (!event.target.closest('.md-mermaid-controls')) {
        fitMermaidViewport(canvasEl);
      }
    });
    canvasEl.addEventListener('keydown', function onMermaidKeyDown(event) {
      if (event.key === '+' || event.key === '=') {
        event.preventDefault();
        zoomMermaidViewport(canvasEl, 1.2);
      } else if (event.key === '-' || event.key === '_') {
        event.preventDefault();
        zoomMermaidViewport(canvasEl, 1 / 1.2);
      } else if (event.key === '0') {
        event.preventDefault();
        fitMermaidViewport(canvasEl);
      }
    });

    if (typeof ResizeObserver !== 'undefined') {
      viewport.resizeObserver = new ResizeObserver(function onMermaidResize() {
        if (!viewport.hasInteracted && canvasEl.isConnected) {
          fitMermaidViewport(canvasEl);
        }
      });
      viewport.resizeObserver.observe(canvasEl);
    }
    requestAnimationFrame(function fitRenderedMermaid() { fitMermaidViewport(canvasEl); });
  }

  // Read mermaid renderer from app state.
  function getMermaidRenderer() {
    if (typeof window !== 'undefined' && window.mermaid) {
      return window.mermaid;
    }
    if (typeof globalThis !== 'undefined' && globalThis.mermaid) {
      return globalThis.mermaid;
    }
    return null;
  }

  // Mermaid diagram rendering.
  function configureMermaid() {
    const mermaidApi = getMermaidRenderer();
    if (!mermaidApi || !mermaidApi.initialize) {
      return;
    }
    mermaidApi.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      theme: 'base',
      flowchart: {
        htmlLabels: false
      },
      themeVariables: buildMermaidThemeVariables()
    });
  }

  // Handle hydrate mermaid diagrams.
  function hydrateMermaidDiagrams($root) {
    const root = $root && $root.length ? $root : dom.$messagesInner;
    root.find('.md-mermaid-card').each(function hydrateMermaidCard() {
      const cardEl = this;
      if (cardEl.dataset.mermaidState === 'rendered' || cardEl.dataset.mermaidState === 'rendering') {
        return;
      }

      const canvasEl = cardEl.querySelector('.md-mermaid-canvas');
      const sourceEl = cardEl.querySelector('.md-mermaid-source code');
      const statusEl = cardEl.querySelector('.md-mermaid-status');
      const source = sourceEl ? String(sourceEl.textContent || '').trim() : '';
      if (!canvasEl || !source) {
        cardEl.dataset.mermaidState = 'error';
        if (statusEl) {
          statusEl.textContent = 'Mermaid diagram source is empty.';
        }
        return;
      }

      const mermaidApi = getMermaidRenderer();
      if (!mermaidApi || !mermaidApi.render) {
        cardEl.dataset.mermaidState = 'unavailable';
        if (statusEl) {
          statusEl.textContent = 'Mermaid renderer is unavailable.';
        }
        return;
      }

      const renderToken = `mermaid-${Date.now()}-${++mermaidRenderSeq}`;
      cardEl.dataset.mermaidState = 'rendering';
      cardEl.dataset.mermaidRenderToken = renderToken;
      if (statusEl) {
        statusEl.textContent = 'Rendering diagram...';
      }

      Promise.resolve(mermaidApi.render(`aslm-${renderToken}`, source))
        .then(function onMermaidRendered(result) {
          if (cardEl.dataset.mermaidRenderToken !== renderToken) {
            return;
          }
          const svg = result && result.svg ? result.svg : '';
          if (!svg) {
            throw new Error('Mermaid returned an empty SVG.');
          }
          canvasEl.innerHTML = sanitizeMermaidSvg(svg);
          enhanceMermaidSvg(canvasEl);
          if (result && typeof result.bindFunctions === 'function') {
            result.bindFunctions(canvasEl);
          }
          initializeMermaidViewport(canvasEl);
          cardEl.dataset.mermaidState = 'rendered';
          if (statusEl) {
            statusEl.textContent = '';
          }
        })
        .catch(function onMermaidError(error) {
          if (cardEl.dataset.mermaidRenderToken !== renderToken) {
            return;
          }
          cardEl.dataset.mermaidState = 'error';
          if (statusEl) {
            statusEl.textContent = `Could not render Mermaid diagram: ${error && error.message ? error.message : 'syntax error'}`;
          }
        });
    });
  }

  // Handle tool display name.
  function toolDisplayName(segment) {
    if (deepResearchUi && deepResearchUi.isDeepResearchSegment(segment)) {
      return 'Deep Research';
    }
    if (isSearchToolSegment(segment)) {
      return 'Search';
    }
    if (isReadPageToolSegment(segment)) {
      return 'Read page';
    }
    if (isSandboxPythonToolSegment(segment)) {
      return 'Python';
    }
    if (isSharedFileToolSegment(segment)) {
      return 'Shared file';
    }
    if (isImageViewToolSegment(segment)) {
      return 'Image';
    }
    if (isSandboxToolSegment(segment)) {
      return 'Sandbox';
    }

    return String(segment.toolName || segment.toolId || segment.alias || 'Tool').trim();
  }

  // Handle tool status text.
  function toolStatusText(segment) {
    const rawStatus = segment.toolUi && segment.toolUi.status ? String(segment.toolUi.status).trim().toLowerCase() : '';
    if (rawStatus === 'error' || rawStatus === 'timeout') {
      return rawStatus === 'timeout' ? 'Timeout' : 'Error';
    }
    return segment.result !== null && segment.result !== undefined ? 'Done' : 'Running';
  }

  // Handle tool status class.
  function toolStatusClass(segment) {
    const status = toolStatusText(segment).toLowerCase();
    if (status === 'error' || status === 'timeout') {
      return ' is-error';
    }
    if (status === 'running') {
      return ' is-pending';
    }
    return ' is-done';
  }

  // Format one MCP payload for the compact inline call view. Arguments are
  // always JSON, while tool output keeps plain stdout intact unless it is a
  // structured JSON object or array.
  function formatMcpPayload(value, forceJson) {
    if (forceJson) {
      try {
        return {
          text: JSON.stringify(value && typeof value === 'object' ? value : {}, null, 2),
          language: 'json'
        };
      } catch (_error) {
        return { text: String(value || ''), language: 'plaintext' };
      }
    }

    const text = value === null || value === undefined ? '' : String(value);
    const trimmed = text.trim();
    if (trimmed && (trimmed.charAt(0) === '{' || trimmed.charAt(0) === '[')) {
      try {
        const parsed = JSON.parse(trimmed);
        if (parsed && typeof parsed === 'object') {
          return { text: JSON.stringify(parsed, null, 2), language: 'json' };
        }
      } catch (_error) {
        // A stdout line can resemble JSON without being valid JSON. Preserve it.
      }
    }
    return { text, language: 'plaintext' };
  }

  // Render a generic MCP call without decorative chrome. The tool name is the
  // only header; input and stdout use the same quiet, left-ruled code treatment.
  function renderMcpToolCall(segment, toolSegmentIndex, options) {
    const renderOptions = options || {};
    const name = toolDisplayName(segment);
    const input = formatMcpPayload(
      segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {},
      true
    );
    const hasResult = segment.result !== null && segment.result !== undefined;
    const output = formatMcpPayload(segment.result, false);
    const dataIndex = Number.isInteger(toolSegmentIndex) ? ` data-tool-segment-index="${toolSegmentIndex}"` : '';

    const renderPayload = function renderPayload(payload, extraClass) {
      const safeLanguage = String(payload.language || 'plaintext').replace(/[^a-z0-9_-]/gi, '') || 'plaintext';
      return `
        <pre class="msg-mcp-payload msg-code-block language-${safeLanguage}${extraClass ? ` ${extraClass}` : ''}" data-language="${escapeAttributeValue(safeLanguage)}"><code>${highlightCode(payload.text, safeLanguage)}</code></pre>
      `;
    };

    return `
      <div class="msg-mcp-call${toolStatusClass(segment)}"${dataIndex} aria-label="${escapeAttributeValue(name)}">
        ${renderOptions.hideName ? '' : `<div class="msg-mcp-name">${escHtml(name)}</div>`}
        ${renderPayload(input, 'is-input')}
        ${hasResult && output.text ? renderPayload(output, 'is-output') : ''}
      </div>
    `;
  }

  // Handle reasoning tool detail.
  function reasoningToolDetail(segment) {
    if (isSearchToolSegment(segment)) {
      return searchQueryFromSegment(segment) || 'sources';
    }

    if (isReadPageToolSegment(segment)) {
      const sources = readPageSourcesFromSegment(segment);
      if (sources.length > 0) {
        return sources
          .map(function sourceLabel(source) { return source.display_domain || source.domain || source.url || ''; })
          .filter(Boolean)
          .slice(0, 2)
          .join(', ');
      }
      return 'source page';
    }

    if (isSharedFileToolSegment(segment)) {
      const file = sharedFileFromSegment(segment);
      const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
      return String((file && file.filename) || args.filename || args.path || args.file || 'file').trim();
    }

    if (isImageViewToolSegment(segment)) {
      const result = imageResultFromSegment(segment);
      const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
      return String((result && result.path) || args.path || args.file || args.filename || 'image').trim();
    }

    const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    const heavyKeys = new Set(heavyToolKeysForSegment(segment));
    const preferredKeys = ['query', 'q', 'url', 'urls', 'path', 'file', 'filename', 'command', 'cmd', 'code', 'prompt', 'input'];
    for (let index = 0; index < preferredKeys.length; index += 1) {
      const key = preferredKeys[index];
      if (heavyKeys.has(key)) {
        continue;
      }
      if (args[key] !== undefined && args[key] !== null && String(args[key]).trim() !== '') {
        return compactToolValue(args[key]);
      }
    }

    const keys = Object.keys(args).filter(function keepKey(key) {
      return !heavyKeys.has(key) && args[key] !== undefined && args[key] !== null && String(args[key]).trim() !== '';
    });
    return keys.slice(0, 2).map(function formatArg(key) {
      return `${key}: ${compactToolValue(args[key])}`;
    }).join(' · ');
  }

  // Handle tool icon html.
  function toolIconHtml(segment) {
    if (isSearchToolSegment(segment) || isReadPageToolSegment(segment)) {
      return icons.TOOL_SEARCH_ICON || icons.WEB_SEARCH_ICON || icons.GLOBE_ICON || '';
    }
    if (isSharedFileToolSegment(segment)) {
      return DOWNLOAD_FILE_ICON;
    }
    if (isWriteToolSegment(segment)) {
      return icons.TOOL_CODE_EXEC_ICON || icons.TOOL_BASH_ICON || '';
    }
    if (isEditToolSegment(segment)) {
      return icons.TOOL_CODE_EXEC_ICON || icons.TOOL_BASH_ICON || '';
    }
    if (isImageViewToolSegment(segment)) {
      return icons.TOOL_CODE_EXEC_ICON || icons.TOOL_BASH_ICON || '';
    }
    if (isSandboxToolSegment(segment)) {
      return icons.TOOL_CODE_EXEC_ICON || icons.TOOL_BASH_ICON || '';
    }
    return '<span class="msg-mcp-tool-icon" aria-hidden="true"></span>';
  }

  // Render reasoning tool row.
  function renderReasoningToolRow(segment, toolSegmentIndex) {
    const name = toolDisplayName(segment);
    const detail = reasoningToolDetail(segment);
    const status = toolStatusText(segment);
    const iconHtml = toolIconHtml(segment);
    const initial = name.charAt(0).toUpperCase() || 'T';
    const dataIndex = Number.isInteger(toolSegmentIndex) ? ` data-tool-segment-index="${toolSegmentIndex}"` : '';

    return `
      <div class="msg-reasoning-tool-row${toolStatusClass(segment)}"${dataIndex}>
        ${iconHtml || `<span class="msg-reasoning-tool-icon" aria-hidden="true">${escHtml(initial)}</span>`}
        <span class="msg-reasoning-tool-main">
          <span class="msg-reasoning-tool-name">${escHtml(name)}</span>
          ${detail ? `<span class="msg-reasoning-tool-detail">${escHtml(detail)}</span>` : ''}
        </span>
        <span class="msg-reasoning-tool-status">${escHtml(status)}</span>
      </div>
    `;
  }

  // Render reasoning tool item.
  function renderReasoningToolItem(item, options) {
    const segment = item && item.segment ? item.segment : item;
    const toolIndex = item && Number.isInteger(item.toolIndex) ? item.toolIndex : undefined;
    if (!segment) {
      return '';
    }

    if (deepResearchUi && deepResearchUi.isDeepResearchSegment(segment)) {
      return deepResearchUi.renderCard(segment, toolIndex);
    }
    if (isSearchToolSegment(segment)) {
      return renderSearchToolCard(segment, toolIndex, { compactLabel: true, hideIcon: false, expanded: !!item.expanded });
    }
    if (isReadPageToolSegment(segment)) {
      return renderReadPageToolCard([{ segment, index: toolIndex }]);
    }
    if (isImageViewToolSegment(segment)) {
      return renderSandboxImageToolBlock(segment, toolIndex) || renderReasoningToolRow(segment, toolIndex);
    }
    if (isSharedFileToolSegment(segment)) {
      return renderSharedFileCard(segment);
    }
    if (isCompressionContextSegment(segment)) {
      return renderCompressionContextCard(segment, toolIndex);
    }
    if (isWriteToolSegment(segment)) {
      return renderWriteToolCard(segment, toolIndex, { expanded: !!item.expanded });
    }
    if (isEditToolSegment(segment)) {
      return renderEditToolCard(segment, toolIndex, { expanded: !!item.expanded });
    }
    if (isSandboxToolSegment(segment)) {
      return renderSandboxToolBlock(segment, toolIndex);
    }
    return renderMcpToolCall(segment, toolIndex, options);
  }

  // Handle reasoning tool step title.
  function reasoningToolStepTitle(segment) {
    if (isSearchToolSegment(segment)) {
      const description = searchDescriptionFromSegment(segment);
      return description || (toolStatusText(segment) === 'Done' ? 'Searched sources' : 'Searching sources');
    }
    if (isReadPageToolSegment(segment)) {
      return 'Reading source page';
    }
    if (isSandboxPythonToolSegment(segment)) {
      return toolStatusText(segment) === 'Done' ? 'Ran Python' : 'Running Python';
    }
    if (isSharedFileToolSegment(segment)) {
      const file = sharedFileFromSegment(segment);
      return file && file.filename ? `Shared ${file.filename}` : 'Shared file';
    }
    if (isWriteToolSegment(segment)) {
      const path = writePathFromSegment(segment);
      return path ? `Writing ${path}` : 'Writing file';
    }
    if (isEditToolSegment(segment)) {
      const result = parseToolResultObject(segment);
      const path = editPathFromSegment(segment, result);
      return path ? `Editing ${path}` : 'Editing file';
    }
    if (isImageViewToolSegment(segment)) {
      const result = imageResultFromSegment(segment);
      const args = segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
      const path = String((result && result.path) || args.path || '').trim();
      return path ? `Viewing ${path}` : 'Viewing image';
    }
    if (isSandboxToolSegment(segment)) {
      return 'Running sandbox command';
    }
    return toolDisplayName(segment);
  }

  // Return a short action description explicitly supplied for a tool call.
  function reasoningToolActionDescription(segment) {
    if (isSearchToolSegment(segment)) {
      return searchDescriptionFromSegment(segment);
    }

    const toolUi = segment && segment.toolUi && typeof segment.toolUi === 'object' ? segment.toolUi : {};
    const compact = toolUi.compact && typeof toolUi.compact === 'object' ? toolUi.compact : {};
    const structured = segment && segment.structuredContent && typeof segment.structuredContent === 'object'
      ? segment.structuredContent
      : {};
    const structuredUi = structured.ui && typeof structured.ui === 'object' ? structured.ui : {};
    const args = segment && segment.arguments && typeof segment.arguments === 'object' ? segment.arguments : {};
    return String(
      toolUi.description
      || compact.description
      || structured.description
      || structuredUi.description
      || args.description
      || ''
    ).replace(/\s+/g, ' ').trim();
  }

  // Browser-agent activity has its own portal and must not drive the Thought label.
  function isIgnoredReasoningActivity(segment) {
    return !!(
      segment
      && (
        isSharedFileToolSegment(segment)
        || segment.type === 'browser_portal'
        || (browserPortalUi
          && typeof browserPortalUi.isBrowserToolSegment === 'function'
          && browserPortalUi.isBrowserToolSegment(segment))
      )
    );
  }

  // Remove parenthetical/service-only details from the compact live label.
  // Quoted prose remains readable, including bracket characters inside quotes.
  function stripReasoningPreviewBracketedText(value) {
    const output = [];
    const expectedClosers = [];
    const closingBrackets = new Set(Object.values(REASONING_PREVIEW_BRACKET_PAIRS));
    let activeQuoteCloser = '';

    Array.from(String(value || '')).forEach(function filterReasoningPreviewCharacter(character) {
      if (activeQuoteCloser) {
        output.push(character);
        if (character === activeQuoteCloser) {
          activeQuoteCloser = '';
        }
        return;
      }

      if (!expectedClosers.length && REASONING_PREVIEW_QUOTE_PAIRS[character]) {
        activeQuoteCloser = REASONING_PREVIEW_QUOTE_PAIRS[character];
        output.push(character);
        return;
      }

      const bracketCloser = REASONING_PREVIEW_BRACKET_PAIRS[character];
      if (bracketCloser) {
        if (!expectedClosers.length && output.length && !/\s/u.test(output[output.length - 1])) {
          output.push(' ');
        }
        expectedClosers.push(bracketCloser);
        return;
      }

      if (expectedClosers.length) {
        const matchingCloserIndex = expectedClosers.lastIndexOf(character);
        if (matchingCloserIndex >= 0) {
          expectedClosers.length = matchingCloserIndex;
        }
        return;
      }

      if (!closingBrackets.has(character)) {
        output.push(character);
      }
    });

    return output.join('');
  }

  // Markdown from reasoning models is frequently partial or malformed while
  // streaming. Remove syntax characters that can survive the normal plain-text
  // conversion instead of exposing unfinished emphasis, lists, or tables.
  function stripResidualReasoningMarkdownSyntax(value) {
    return String(value || '')
      .replace(/[\*_~`#|]+/g, ' ')
      .replace(/(^|\s)>{1,}(?=\s|$)/g, '$1')
      .replace(/(^|\s)[+-](?=\s)/g, '$1')
      .replace(/\\(?=[\\`*_{}\[\]()#+\-.!>|~])/g, '')
      .replace(/[\[\]{}]/g, ' ')
      .replace(/\s+([,.;:!?\u2026])/gu, '$1')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function sanitizeReasoningDisplayText(value) {
    return String(value || '')
      .replace(/\r\n?/g, '\n')
      .split('\n')
      .map(function sanitizeReasoningDisplayLine(line) {
        return stripResidualReasoningMarkdownSyntax(markdownToPlainText(line));
      })
      .join('\n')
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
  }

  function sanitizeReasoningPreviewText(value) {
    const plainText = markdownToPlainText(value)
      .replace(REASONING_PREVIEW_PRIVATE_MARKER_PATTERN, ' ')
      .replace(REASONING_PREVIEW_BRACKETED_CITATION_PATTERN, ' ')
      .replace(REASONING_PREVIEW_INTERNAL_MARKER_PATTERN, ' ')
      .replace(REASONING_PREVIEW_INVISIBLE_PATTERN, ' ');
    return stripResidualReasoningMarkdownSyntax(stripReasoningPreviewBracketedText(plainText))
      .replace(/\s+([,.;:!?\u2026])/gu, '$1')
      .replace(/\s+/g, ' ')
      .trim();
  }

  function reasoningPreviewWordCount(value) {
    const words = String(value || '').match(/\p{L}[\p{L}\p{M}]*(?:[-'\u2019][\p{L}\p{M}]+)*/gu);
    return words ? words.length : 0;
  }

  // Pick the latest complete clause so the label changes at punctuation, not per token.
  function reasoningThoughtExcerpt(content) {
    const text = sanitizeReasoningPreviewText(content);
    if (!text) {
      return '';
    }

    const clauses = text.match(/[^.!?,\u2026;:\n]+[.!?,\u2026;:]+(?:["'»”\)\]]+)?/g) || [];
    const readableClauses = clauses.map(function normalizeReasoningClause(clause) {
      return String(clause || '').trim();
    }).filter(function keepReadableReasoningClause(clause) {
      return clause.length >= REASONING_LABEL_MIN_CHARS
        && clause.length <= REASONING_LABEL_MAX_CHARS
        && reasoningPreviewWordCount(clause) >= REASONING_LABEL_MIN_WORDS;
    });
    return readableClauses.length
      ? normalizeReasoningPreviewPunctuation(readableClauses[readableClauses.length - 1])
      : '';
  }

  // Replace sentence-ending punctuation with one calm preview ellipsis while
  // preserving any closing quote or bracket after it.
  function normalizeReasoningPreviewPunctuation(label) {
    return sanitizeReasoningPreviewText(label).replace(/[.!?,\u2026;:]+(["'»”\)\]]*)$/u, '\u2026$1');
  }

  // Gather only the sources belonging to the latest displayed search or read
  // activity. Older results must not leak into the live preview when a new
  // request becomes active.
  function reasoningActivitySources(activeSegment) {
    const isSearchActivity = isSearchToolSegment(activeSegment);
    const isReadActivity = isReadPageToolSegment(activeSegment);
    if (!isSearchActivity && !isReadActivity) {
      return [];
    }

    const sources = isSearchActivity
      ? searchSourcesFromSegment(activeSegment)
      : readPageSourcesFromSegment(activeSegment);

    return dedupeSearchSources(sources.map(function normalizeActivitySource(source, index) {
      return normalizeSearchSourceItem(source, index + 1);
    }).filter(Boolean));
  }

  // Build the live Thought label and optional source stack.
  function reasoningTogglePresentation(items, options) {
    const renderOptions = options || {};
    const safeItems = Array.isArray(items) ? items : [];
    if (renderOptions.streaming !== true || renderOptions.reasoningComplete === true) {
      return { label: 'Thought', sources: [], dynamic: false };
    }

    for (let index = safeItems.length - 1; index >= 0; index -= 1) {
      const item = safeItems[index];
      if (!item) {
        continue;
      }
      if (item.type === 'thought') {
        const excerpt = reasoningThoughtExcerpt(item.content);
        if (excerpt) {
          return { label: excerpt, sources: [], dynamic: true };
        }
        continue;
      }
      if (item.type === 'tool_pending') {
        return { label: 'Preparing tool call', sources: [], dynamic: true };
      }
      if (item.type !== 'tool') {
        continue;
      }

      const segment = item.segment || item;
      if (isIgnoredReasoningActivity(segment)) {
        continue;
      }
      const sources = reasoningActivitySources(segment);
      const description = normalizeReasoningPreviewPunctuation(reasoningToolActionDescription(segment));
      let label = description;
      if (!label && isReadPageToolSegment(segment)) {
        label = sources.length > 1 ? `Reading ${sources.length} sources` : 'Reading source';
      }
      if (!label) {
        label = reasoningToolStepTitle(segment);
      }
      return { label: label || 'Using tool', sources, dynamic: true };
    }

    return { label: 'Thinking', sources: [], dynamic: true };
  }

  // Render up to three overlapping source favicons and a leading overflow count.
  function renderReasoningSourceStack(sources) {
    const safeSources = Array.isArray(sources) ? sources : [];
    if (!safeSources.length) {
      return '';
    }

    const visibleSources = safeSources.slice(0, 3);
    const overflowCount = Math.max(0, safeSources.length - visibleSources.length);
    const avatarsHtml = visibleSources.map(function renderReasoningSource(source) {
      const domain = String(source.display_domain || source.domain || source.url || '').trim();
      const faviconUrl = sourceFaviconUrl(source);
      const fallbackLetter = (domain.charAt(0) || '?').toUpperCase();
      const imgHtml = faviconUrl
        ? `<img class="msg-reasoning-source-favicon" src="${escapeAttributeValue(faviconUrl)}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='inline-flex';">`
        : '';
      const fallbackStyle = domainAccentStyle(domain, faviconUrl ? 'display:none;' : '');
      return `<span class="msg-reasoning-source" title="${escapeAttributeValue(domain)}">${imgHtml}<span class="msg-reasoning-source-fallback" style="${escapeAttributeValue(fallbackStyle)}">${escHtml(fallbackLetter)}</span></span>`;
    }).join('');

    return `
      <span class="msg-reasoning-sources" aria-label="${safeSources.length} sources">
        ${overflowCount > 0 ? `<span class="msg-reasoning-source-overflow">+${overflowCount}</span>` : ''}
        <span class="msg-reasoning-source-stack has-${visibleSources.length}">${avatarsHtml}</span>
      </span>
    `;
  }

  // Report whether  thought segments.
  function hasThoughtSegments(segments) {
    return (Array.isArray(segments) ? segments : []).some(function hasThought(segment) {
      return segment && segment.type === 'thought';
    });
  }

  // Report whether  reasoning marker.
  function hasReasoningMarker(rawText) {
    return !!findNextReasoningStart(rawText, 0);
  }

  // Handle should use reasoning shell.
  function shouldUseReasoningShell($msgRow, segments, rawText, renderOptions) {
    const options = renderOptions || {};
    const rawHasReasoningMarker = hasReasoningMarker(rawText);
    const segmentHasThoughts = hasThoughtSegments(segments);
    const trustedSegmentThoughts = segmentHasThoughts && options.trustThoughtSegments !== false;
    const sawReasoning = trustedSegmentThoughts || rawHasReasoningMarker;
    if (sawReasoning) {
      $msgRow.data('sawReasoning', true);
    }

    return !!options.reasoningMode
      || !!$msgRow.data('reasoningModeEnabled')
      || !!$msgRow.data('sawReasoning')
      || rawHasReasoningMarker
      || trustedSegmentThoughts;
  }

  // Report whether  closed reasoning.
  function hasClosedReasoning(rawText) {
    return /<\/(?:think|thinking|reasoning|analysis)>/i.test(String(rawText || ''));
  }

  // Handle split long reasoning sentence.
  function splitLongReasoningSentence(sentence, targetLength) {
    const words = String(sentence || '').trim().split(/\s+/).filter(Boolean);
    const chunks = [];
    let current = '';

    words.forEach(function addWord(word) {
      const next = current ? `${current} ${word}` : word;
      if (current && next.length > targetLength) {
        chunks.push(current);
        current = word;
        return;
      }
      current = next;
    });

    if (current) {
      chunks.push(current);
    }
    return chunks;
  }

  // Handle split reasoning text.
  function splitReasoningText(content) {
    const normalized = String(content || '').replace(/\s+/g, ' ').trim();
    if (!normalized) {
      return [];
    }

    const sentenceParts = normalized.match(/[^.!?…]+[.!?…]+["')\]]*|[^.!?…]+$/g) || [normalized];
    const chunks = [];
    let current = '';

    sentenceParts.forEach(function addSentence(part) {
      const sentence = String(part || '').trim();
      if (!sentence) {
        return;
      }

      if (sentence.length > REASONING_CHUNK_MAX_CHARS) {
        if (current) {
          chunks.push(current);
          current = '';
        }
        splitLongReasoningSentence(sentence, REASONING_CHUNK_TARGET_CHARS).forEach(function pushLongChunk(chunk) {
          if (chunk) {
            chunks.push(chunk);
          }
        });
        return;
      }

      const next = current ? `${current} ${sentence}` : sentence;
      if (current && next.length > REASONING_CHUNK_MAX_CHARS) {
        chunks.push(current);
        current = sentence;
        return;
      }

      current = next;
    });

    if (current) {
      chunks.push(current);
    }

    return chunks.length > 0 ? chunks : [normalized];
  }

  // Render reasoning thought text.
  function renderReasoningThoughtText(content) {
    const text = sanitizeReasoningDisplayText(content);
    if (!text) {
      return '';
    }
    return `<div class="msg-reasoning-text">${escHtml(text)}</div>`;
  }

  // Render pending tool call placeholder.
  function renderPendingToolCallPlaceholder() {
    return `
      <div class="msg-tool-pending-card" aria-live="polite">
        <span class="msg-tool-pending-icon" aria-hidden="true"></span>
        <span class="msg-tool-pending-main">
          <span class="msg-tool-pending-title">Preparing tool call</span>
          <span class="msg-tool-pending-detail">Tool input is being formed</span>
        </span>
        <span class="msg-tool-pending-dot" aria-hidden="true"></span>
      </div>
    `;
  }

  // Render reasoning group.
  function renderReasoningGroup(items, thoughtIndex, isExpanded, togglePresentation, options) {
    const safeItems = Array.isArray(items) ? items : [];
    const groupOptions = options || {};
    const presentation = togglePresentation && typeof togglePresentation === 'object'
      ? togglePresentation
      : { label: String(togglePresentation || 'Thought'), sources: [], dynamic: false };
    const thoughtCount = safeItems.filter(function countThoughts(item) { return item && item.type === 'thought'; }).length;
    const toolCount = safeItems.filter(function countTools(item) {
      return item && (item.type === 'tool' || item.type === 'tool_pending');
    }).length;
    const summaryParts = [];
    if (thoughtCount > 1) {
      summaryParts.push(`${thoughtCount} thoughts`);
    }
    if (toolCount > 0) {
      summaryParts.push(`${toolCount} tool ${toolCount === 1 ? 'call' : 'calls'}`);
    }

    // Tool-only group (no reasoning): normally render tool cards inline.
    // When reasoning is active, keep them in the same collapsed shell.
    if (thoughtCount === 0 && !groupOptions.forceWrapper) {
      const renderedTools = [];
      for (let itemIndex = 0; itemIndex < safeItems.length; itemIndex += 1) {
        const item = safeItems[itemIndex];
        if (!item) {
          continue;
        }
        if (item.type === 'tool') {
          const segment = item && item.segment ? item.segment : item;
          const toolIndex = item && Number.isInteger(item.toolIndex) ? item.toolIndex : undefined;
          if (isInstantSearchFlowSegment(segment) && isReadPageToolSegment(segment)) {
            continue;
          }
          if (isInstantSearchFlowSegment(segment) && isSearchToolSegment(segment)) {
            const searchItems = [];
            let cursor = itemIndex;
            while (cursor < safeItems.length) {
              const candidate = safeItems[cursor];
              const candidateSegment = candidate && candidate.type === 'tool'
                ? (candidate.segment || candidate)
                : null;
              if (!candidateSegment || !isInstantSearchFlowSegment(candidateSegment)) {
                break;
              }
              if (isSearchToolSegment(candidateSegment)) {
                searchItems.push({
                  segment: candidateSegment,
                  index: Number.isInteger(candidate.toolIndex) ? candidate.toolIndex : undefined
                });
              } else if (!isReadPageToolSegment(candidateSegment)) {
                break;
              }
              cursor += 1;
            }
            renderedTools.push(renderSearchToolGroup(searchItems, { expanded: false }));
            itemIndex = cursor - 1;
            continue;
          }
          if (isSandboxToolSegment(segment) && !isImageViewToolSegment(segment)) {
            renderedTools.push(renderCollapsedSandboxToolBlock(segment, toolIndex, !!item.expanded));
            continue;
          }
          renderedTools.push(renderReasoningToolItem(item));
          continue;
        }
        if (item.type === 'tool_pending') {
          renderedTools.push(renderPendingToolCallPlaceholder());
        }
      }
      const toolsHtml = renderedTools.join('');
      if (!toolsHtml.trim()) {
        return '';
      }
      return `<div class="msg-tool-only-group">${toolsHtml}</div>`;
    }

    const renderItems = safeItems.filter(function hideInstantReadPage(item) {
      const segment = item && item.type === 'tool' ? (item.segment || item) : null;
      return !(segment && isReadPageToolSegment(segment) && isInstantSearchFlowSegment(segment));
    });
    const contentHtml = renderItems.map(function renderReasoningItem(item) {
      if (!item) {
        return '';
      }
      if (item.type === 'tool_pending') {
        return `
          <div class="msg-reasoning-step msg-reasoning-step--tool is-pending">
            <div class="msg-reasoning-step-dot" aria-hidden="true"></div>
            <div class="msg-reasoning-step-body">
              <div class="msg-reasoning-step-title">
                <span>Preparing tool call</span>
                <span class="msg-reasoning-step-status">Pending</span>
              </div>
              ${renderPendingToolCallPlaceholder()}
            </div>
          </div>
        `;
      }
      if (item.type === 'tool') {
        const segment = item && item.segment ? item.segment : item;
        const title = reasoningToolStepTitle(segment || {});
        const status = toolStatusText(segment || {});
        const iconHtml = toolIconHtml(segment || {});
        const hideStepTitle = isWriteToolSegment(segment || {}) || isEditToolSegment(segment || {});
        const isDeepResearch = !!(deepResearchUi && deepResearchUi.isDeepResearchSegment(segment || {}));
        return `
          <div class="msg-reasoning-step msg-reasoning-step--tool${toolStatusClass(segment || {})}${isDeepResearch ? ' is-deep-research' : ''}">
            ${isDeepResearch ? '' : `<div class="msg-reasoning-step-dot" aria-hidden="true">${iconHtml}</div>`}
            <div class="msg-reasoning-step-body">
              ${isDeepResearch ? '' : `<div class="msg-reasoning-step-title">
                ${hideStepTitle ? '' : `<span>${escHtml(title)}</span>`}
                <span class="msg-reasoning-step-status">${escHtml(status)}</span>
              </div>`}
              ${renderReasoningToolItem(item, { hideName: true })}
            </div>
          </div>
        `;
      }
      return `
        <div class="msg-reasoning-step">
          <div class="msg-reasoning-step-dot" aria-hidden="true"></div>
          <div class="msg-reasoning-step-body">
            ${renderReasoningThoughtText(item.content || '')}
          </div>
        </div>
      `;
    }).join('');

    return `
      <div class="msg-thoughts-wrapper msg-reasoning-wrapper${isExpanded ? ' expanded' : ''}" data-thought-index="${thoughtIndex}">
        <button type="button" class="msg-thoughts-toggle msg-reasoning-toggle" aria-expanded="${isExpanded ? 'true' : 'false'}">
          ${renderReasoningSourceStack(presentation.sources)}
          <span class="msg-reasoning-title" data-reasoning-label="${escapeAttributeValue(presentation.label || 'Thought')}" title="${escapeAttributeValue(presentation.label || 'Thought')}">${escHtml(presentation.label || 'Thought')}</span>
          ${!presentation.dynamic && summaryParts.length ? `<span class="msg-reasoning-summary">${escHtml(summaryParts.join(' · '))}</span>` : ''}
        </button>
        <div class="msg-thoughts-content msg-reasoning-content" style="display:${isExpanded ? 'block' : 'none'};">${contentHtml}</div>
      </div>
    `;
  }

  // Render thought block.
  function renderThoughtBlock(content, thoughtIndex, isExpanded) {
    return `
      <div class="msg-thoughts-wrapper msg-reasoning-wrapper${isExpanded ? ' expanded' : ''}" data-thought-index="${thoughtIndex}">
        <button type="button" class="msg-thoughts-toggle msg-reasoning-toggle" aria-expanded="${isExpanded ? 'true' : 'false'}">
          <span class="msg-reasoning-title">Thought</span>
        </button>
        <div class="msg-thoughts-content msg-reasoning-content" style="display:${isExpanded ? 'block' : 'none'};">${renderReasoningThoughtText(content)}</div>
      </div>
    `;
  }

  // Create activity block.
  function createActivityBlock(key, innerHtml) {
    const safeKey = String(key || 'block');
    return {
      key: safeKey,
      html: `<div class="msg-activity-block" data-activity-key="${escapeAttributeValue(safeKey)}">${innerHtml}</div>`
    };
  }

  // Wrap only the freshly revealed text nodes after markdown has been parsed.
  // This keeps markdown structure intact while giving the advancing edge a
  // soft gradient instead of flashing a whole new network chunk at once.
  function decorateStreamingRevealTail(html, tailCharacterCount, revealFrame) {
    const count = Math.max(0, Math.floor(Number(tailCharacterCount) || 0));
    if (!count || typeof document === 'undefined') {
      return html;
    }

    const template = document.createElement('template');
    template.innerHTML = String(html || '');
    const walker = document.createTreeWalker(template.content, 4);
    const textNodes = [];
    let node = walker.nextNode();
    while (node) {
      if (node.nodeValue && node.nodeValue.trim()) {
        textNodes.push(node);
      }
      node = walker.nextNode();
    }

    let remaining = count;
    let processedFromEnd = 0;
    for (let index = textNodes.length - 1; index >= 0 && remaining > 0; index -= 1) {
      const textNode = textNodes[index];
      const value = String(textNode.nodeValue || '');
      const take = Math.min(remaining, Array.from(value).length);
      if (!take) {
        continue;
      }

      const characters = Array.from(value);
      const prefix = characters.slice(0, characters.length - take).join('');
      const tail = characters.slice(characters.length - take).join('');
      const fragment = document.createDocumentFragment();
      if (prefix) {
        fragment.appendChild(document.createTextNode(prefix));
      }
      const tailCharacters = Array.from(tail);
      tailCharacters.forEach(function wrapCleanRevealCharacter(character, characterIndex) {
        const distanceFromEnd = processedFromEnd + (tailCharacters.length - characterIndex - 1);
        const gradientProgress = count > 1 ? Math.min(1, distanceFromEnd / (count - 1)) : 1;
        const startOpacity = 0.28 + (gradientProgress * 0.68);
        const delayMs = Math.max(0, Math.round((count - distanceFromEnd - 1) * 2));
        const span = document.createElement('span');
        span.className = `msg-stream-reveal-tail msg-stream-reveal-tail--${Number(revealFrame) % 2 === 0 ? 'even' : 'odd'}`;
        span.style.setProperty('--stream-reveal-start-opacity', startOpacity.toFixed(2));
        span.style.setProperty('--stream-reveal-delay', `${delayMs}ms`);
        span.textContent = character;
        fragment.appendChild(span);
      });
      textNode.parentNode.replaceChild(fragment, textNode);
      processedFromEnd += take;
      remaining -= take;
    }

    return template.innerHTML;
  }

  // Return a stable identity key for an element so the morpher can match it
  // across re-renders without destroying the DOM node.
  function getElementMorphKey(el) {
    if (!el || el.nodeType !== 1) {
      return null;
    }
    const searchKey = el.classList.contains('msg-search-card')
      ? el.getAttribute('data-search-key')
      : null;
    if (searchKey) {
      return 'search:' + searchKey;
    }
    const researchKey = el.classList.contains('msg-deep-research-card')
      ? el.getAttribute('data-research-key')
      : null;
    if (researchKey) {
      return 'research:' + researchKey;
    }
    const writeIdx = el.getAttribute('data-write-segment-index');
    if (writeIdx !== null) {
      return 'write:' + writeIdx;
    }
    const editIdx = el.getAttribute('data-edit-segment-index');
    if (editIdx !== null) {
      return 'edit:' + editIdx;
    }
    const toolIdx = el.getAttribute('data-tool-segment-index');
    if (toolIdx !== null) {
      return 'tool:' + toolIdx;
    }
    const reasoningCardIdx = el.getAttribute('data-reasoning-card-index');
    if (reasoningCardIdx !== null && el.classList.contains('msg-reasoning-thought-card')) {
      return 'reasoning-card:' + reasoningCardIdx;
    }
    if (el.classList.contains('msg-tool-pending-card')) {
      return 'tool-pending';
    }
    const activityKey = el.getAttribute('data-activity-key');
    if (activityKey) {
      return 'activity:' + activityKey;
    }
    // Structural singletons keyed by class so their DOM identity is preserved
    // and CSS transitions (e.g. search extra-chips) are not interrupted.
    if (el.classList.contains('msg-search-extra-chips')) {
      return 'search-extra-chips';
    }
    if (el.classList.contains('msg-search-chips')) {
      return 'search-chips';
    }
    if (el.classList.contains('msg-write-preview')) {
      return 'write-preview';
    }
    if (el.classList.contains('msg-edit-preview')) {
      return 'edit-preview';
    }
    return null;
  }

  // Copy attributes from toEl onto fromEl without touching fromEl's identity.
  function syncElementAttrs(fromEl, toEl) {
    const preserveBrowserPortalClientAttrs =
      fromEl.nodeType === 1
      && toEl.nodeType === 1
      && fromEl.classList.contains('browser-portal')
      && toEl.classList.contains('browser-portal');
    const preservedBrowserPortal = {};
    if (preserveBrowserPortalClientAttrs) {
      [
        'data-browser-portal-timer-started',
        'data-browser-portal-hydrated',
        'data-browser-session-id',
        'data-browser-wait-started-at',
        'data-browser-wait-deadline-at',
        'data-browser-wait-elapsed'
      ].forEach(function copyAttr(name) {
        const value = fromEl.getAttribute(name);
        if (value !== null && value !== '') {
          preservedBrowserPortal[name] = value;
        }
      });
    }
    const toAttrs = toEl.attributes;
    for (let i = 0; i < toAttrs.length; i += 1) {
      const { name, value } = toAttrs[i];
      if (fromEl.getAttribute(name) !== value) {
        fromEl.setAttribute(name, value);
      }
    }
    for (let i = fromEl.attributes.length - 1; i >= 0; i -= 1) {
      const name = fromEl.attributes[i].name;
      if (!toEl.hasAttribute(name)) {
        fromEl.removeAttribute(name);
      }
    }
    if (preserveBrowserPortalClientAttrs) {
      Object.keys(preservedBrowserPortal).forEach(function restoreAttr(name) {
        fromEl.setAttribute(name, preservedBrowserPortal[name]);
      });
    }
  }

  // Recursively morph fromEl's attributes and children to match toEl while
  // keeping matched DOM nodes alive so CSS transitions are not interrupted.
  function morphDomNode(fromEl, toEl) {
    if (fromEl.nodeType === 3 && toEl.nodeType === 3) {
      if (fromEl.nodeValue !== toEl.nodeValue) {
        fromEl.nodeValue = toEl.nodeValue;
      }
      return fromEl;
    }
    if (fromEl.nodeType !== 1 || fromEl.nodeName !== toEl.nodeName) {
      const replacement = toEl.cloneNode(true);
      if (fromEl.parentNode) {
        fromEl.parentNode.replaceChild(replacement, fromEl);
      }
      return replacement;
    }
    syncElementAttrs(fromEl, toEl);
    morphDomChildren(fromEl, toEl);
    return fromEl;
  }

  // Reconcile fromParent's children against toParent's children, reusing
  // existing DOM nodes that share a morph key and matching unkeyed nodes by
  // tag name when possible.
  function morphDomChildren(fromParent, toParent) {
    const toChildren = Array.from(toParent.childNodes);
    const fromByKey = new Map();
    const fromUnkeyed = [];

    Array.from(fromParent.childNodes).forEach(function catalogFromChild(child) {
      const k = child.nodeType === 1 ? getElementMorphKey(child) : null;
      if (k) {
        fromByKey.set(k, child);
      } else {
        fromUnkeyed.push(child);
      }
    });

    let unkeyedIdx = 0;
    const targetNodes = [];

    toChildren.forEach(function processToChild(toChild) {
      const toKey = toChild.nodeType === 1 ? getElementMorphKey(toChild) : null;
      if (toKey && fromByKey.has(toKey)) {
        const fromChild = fromByKey.get(toKey);
        fromByKey.delete(toKey);
        targetNodes.push(morphDomNode(fromChild, toChild));
        return;
      }
      const candidate = fromUnkeyed[unkeyedIdx];
      if (candidate && candidate.nodeName === toChild.nodeName) {
        unkeyedIdx += 1;
        targetNodes.push(morphDomNode(candidate, toChild));
        return;
      }
      targetNodes.push(toChild.cloneNode(true));
    });

    let referenceNode = fromParent.firstChild;
    targetNodes.forEach(function placeTargetNode(node) {
      if (node === referenceNode) {
        referenceNode = referenceNode.nextSibling;
        return;
      }
      fromParent.insertBefore(node, referenceNode);
    });

    const targetSet = new Set(targetNodes);
    Array.from(fromParent.childNodes).forEach(function removeUnexpectedNode(node) {
      if (!targetSet.has(node)) {
        fromParent.removeChild(node);
      }
    });
  }

  // Handle apply activity blocks.
  function applyActivityBlocks($stream, blocks) {
    const safeBlocks = Array.isArray(blocks) ? blocks : [];
    const streamEl = $stream[0];
    if (!streamEl) {
      return;
    }

    // Build a keyed map of existing blocks for O(1) lookup.
    const existingByKey = new Map();
    $stream.children('.msg-activity-block[data-activity-key]').each(function () {
      existingByKey.set($(this).attr('data-activity-key'), this);
    });

    const targetNodes = [];

    safeBlocks.forEach(function processBlock(block) {
      const existing = existingByKey.get(block.key);
      if (existing) {
        existingByKey.delete(block.key);
        if ($(existing).data('activityHtml') !== block.html) {
          // Morph in-place so CSS transitions on child elements survive.
          const $newBlock = $(block.html);
          morphDomChildren(existing, $newBlock[0]);
          syncElementAttrs(existing, $newBlock[0]);
          $(existing).data('activityHtml', block.html);
        }
        targetNodes.push(existing);
      } else {
        const $block = $(block.html);
        $block.data('activityHtml', block.html);
        targetNodes.push($block[0]);
      }
    });

    // Remove blocks no longer in the list.
    existingByKey.forEach(function (el) {
      el.parentNode && el.parentNode.removeChild(el);
    });

    // Restore order without detaching nodes that are already in the right
    // position; active CSS transitions survive across stream render frames.
    let referenceNode = streamEl.firstChild;
    targetNodes.forEach(function placeTargetNode(node) {
      if (node === referenceNode) {
        referenceNode = referenceNode.nextSibling;
        return;
      }
      streamEl.insertBefore(node, referenceNode);
    });

    const targetSet = new Set(targetNodes);
    Array.from(streamEl.childNodes).forEach(function removeUnexpectedNode(node) {
      if (!targetSet.has(node)) {
        streamEl.removeChild(node);
      }
    });
    hydrateSandboxImages($stream);
    hydrateImageTokens($stream);
    hydrateSharedImageCards($stream);
    if (browserPortalUi && typeof browserPortalUi.hydrate === 'function') {
      browserPortalUi.hydrate($stream);
    }
  }

  // Keep live reasoning labels stable across stream morphs. Each status is
  // typed in, held briefly, then erased before the next status is typed.
  function stopReasoningLabelAnimator($msgRow) {
    const state = $msgRow.data('reasoningLabelAnimator');
    if (!state) {
      return;
    }
    if (state.typingTimer) {
      clearTimeout(state.typingTimer);
    }
    if (state.delayTimer) {
      clearTimeout(state.delayTimer);
    }
    $msgRow.removeData('reasoningLabelAnimator');
  }

  function updateReasoningLabelElement($msgRow, state) {
    const titleEl = $msgRow.find('.msg-reasoning-title[data-reasoning-label]').first()[0];
    if (!titleEl) {
      return false;
    }
    titleEl.textContent = state.currentText || '\u00a0';
    titleEl.classList.toggle('is-typing', !!state.typing);
    titleEl.classList.toggle('is-deleting', !!state.deleting);
    return true;
  }

  function prefersReducedReasoningLabelMotion() {
    return typeof window !== 'undefined'
      && typeof window.matchMedia === 'function'
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function reasoningLabelCharacterDelay(characterCount, reduceMotion) {
    if (reduceMotion) {
      return 0;
    }
    const baseDelay = Math.min(
      REASONING_LABEL_MAX_CHARACTER_DELAY_MS,
      Math.max(
        REASONING_LABEL_MIN_CHARACTER_DELAY_MS,
        Math.floor(REASONING_LABEL_MAX_TYPING_DURATION_MS / Math.max(1, characterCount))
      )
    );
    return baseDelay / REASONING_LABEL_TYPING_SPEED_MULTIPLIER;
  }

  function beginReasoningLabelTyping($msgRow, state, nextLabel, reduceMotion, options) {
    const typingOptions = options || {};
    if (typingOptions.preserveActiveLabel !== true) {
      state.activeLabel = nextLabel;
    }
    state.currentText = '';
    state.deleting = false;
    state.typing = true;
    state.holdCurrentLabel = typingOptions.holdAfterTyping === true;
    const characters = Array.from(nextLabel);
    let characterIndex = 0;
    const characterDelay = reasoningLabelCharacterDelay(characters.length, reduceMotion);
    state.characterDelay = characterDelay;

    function typeNextCharacter() {
      if (!$msgRow[0] || (typeof document !== 'undefined' && !document.documentElement.contains($msgRow[0]))) {
        stopReasoningLabelAnimator($msgRow);
        return;
      }
      if (reduceMotion) {
        characterIndex = characters.length;
      } else {
        characterIndex += 1;
      }
      state.currentText = characters.slice(0, characterIndex).join('');
      updateReasoningLabelElement($msgRow, state);

      if (characterIndex < characters.length) {
        state.typingTimer = setTimeout(typeNextCharacter, characterDelay);
        return;
      }

      state.typingTimer = null;
      state.typing = false;
      state.lastCompletedAt = Date.now();
      updateReasoningLabelElement($msgRow, state);
      scheduleReasoningLabel($msgRow, state, false);
    }

    updateReasoningLabelElement($msgRow, state);
    state.typingTimer = setTimeout(typeNextCharacter, reduceMotion ? 0 : characterDelay);
  }

  function scheduleReasoningLabel($msgRow, state, immediate) {
    const canDeleteCurrentLabel = state.streaming === true
      && !!state.currentText
      && state.holdCurrentLabel !== true;
    if (state.typing || state.deleting || (!state.queuedLabel && !canDeleteCurrentLabel)) {
      return;
    }
    if (state.delayTimer) {
      clearTimeout(state.delayTimer);
      state.delayTimer = null;
    }

    const reduceMotion = prefersReducedReasoningLabelMotion();
    const currentCharacters = Array.from(state.currentText || '');
    const typingDelay = state.characterDelay
      || reasoningLabelCharacterDelay(currentCharacters.length, reduceMotion);
    const deletionDelay = reduceMotion
      ? 0
      : Math.max(1, typingDelay / REASONING_LABEL_DELETE_SPEED_MULTIPLIER);
    const deletionDuration = currentCharacters.length * deletionDelay;
    const elapsed = state.lastCompletedAt ? Date.now() - state.lastCompletedAt : REASONING_LABEL_CHANGE_DELAY_MS;
    // Count deletion inside the hold so it does not make the transition longer.
    const waitMs = immediate
      ? 0
      : Math.max(0, REASONING_LABEL_CHANGE_DELAY_MS - elapsed - deletionDuration);
    state.delayTimer = setTimeout(function beginReasoningLabelTransition() {
      state.delayTimer = null;
      if (!state.queuedLabel && (!state.streaming || !state.currentText)) {
        return;
      }

      let remainingCharacters = Array.from(state.currentText || '');
      function deleteNextCharacter() {
        if (!$msgRow[0] || (typeof document !== 'undefined' && !document.documentElement.contains($msgRow[0]))) {
          stopReasoningLabelAnimator($msgRow);
          return;
        }
        if (reduceMotion) {
          remainingCharacters = [];
        } else {
          remainingCharacters.pop();
        }
        state.currentText = remainingCharacters.join('');
        updateReasoningLabelElement($msgRow, state);

        if (remainingCharacters.length > 0) {
          state.typingTimer = setTimeout(deleteNextCharacter, deletionDelay);
          return;
        }

        state.typingTimer = null;
        state.deleting = false;
        updateReasoningLabelElement($msgRow, state);
        const nextLabel = state.queuedLabel;
        state.queuedLabel = '';
        if (nextLabel) {
          beginReasoningLabelTyping($msgRow, state, nextLabel, reduceMotion);
        } else if (state.streaming) {
          // Never leave the compact reasoning header blank while waiting for
          // another sentence or tool call. Keep the source label identity so
          // repeated stream renders do not queue the erased sentence again.
          beginReasoningLabelTyping($msgRow, state, 'Thinking', reduceMotion, {
            preserveActiveLabel: true,
            holdAfterTyping: true
          });
        }
      }

      if (!remainingCharacters.length || reduceMotion) {
        deleteNextCharacter();
        return;
      }
      state.deleting = true;
      updateReasoningLabelElement($msgRow, state);
      state.typingTimer = setTimeout(deleteNextCharacter, deletionDelay);
    }, waitMs);
  }

  function syncReasoningLabelAnimator($msgRow, streaming) {
    const $title = $msgRow.find('.msg-reasoning-title[data-reasoning-label]').first();
    if (!$title.length) {
      stopReasoningLabelAnimator($msgRow);
      return;
    }

    const desiredLabel = String($title.attr('data-reasoning-label') || 'Thought');
    let animator = $msgRow.data('reasoningLabelAnimator');
    if (streaming !== true) {
      stopReasoningLabelAnimator($msgRow);
      $title.text(desiredLabel).removeClass('is-typing is-deleting');
      return;
    }

    if (!animator) {
      animator = {
        activeLabel: '',
        queuedLabel: desiredLabel,
        currentText: '',
        typing: false,
        deleting: false,
        typingTimer: null,
        delayTimer: null,
        lastCompletedAt: 0,
        characterDelay: 0,
        holdCurrentLabel: false,
        streaming: true
      };
      $msgRow.data('reasoningLabelAnimator', animator);
      updateReasoningLabelElement($msgRow, animator);
      scheduleReasoningLabel($msgRow, animator, true);
      return;
    }

    updateReasoningLabelElement($msgRow, animator);
    if (desiredLabel === animator.activeLabel) {
      animator.queuedLabel = '';
      return;
    }

    animator.queuedLabel = desiredLabel;
    scheduleReasoningLabel($msgRow, animator, false);
  }

  // Render activity timeline.
  function renderActivityTimeline($msgRow, segments, options) {
    const renderOptions = options || {};
    const useMarkdown = renderOptions.markdown !== false;
    const hideTextSegments = renderOptions.hideTextSegments === true;
    const suppressSharedFileToolRows = true;
    const $stream = $msgRow.find('.msg-activity-stream');
    const $bubble = $msgRow.find('.msg-bubble');
    rememberLoadedSandboxImages($msgRow);
    const activeDrawerRow = $activeReasoningMessageRow && $activeReasoningMessageRow[0];
    const activeWrapperRow = $activeReasoningWrapper ? $activeReasoningWrapper.closest('.msg')[0] : null;
    const shouldSyncOpenDrawer = $('#reasoningDrawer').hasClass('is-open')
      && (activeDrawerRow === $msgRow[0] || activeWrapperRow === $msgRow[0]);
    const activeReasoningIndex = shouldSyncOpenDrawer
      ? String($activeReasoningIndex || ($activeReasoningWrapper ? $activeReasoningWrapper.attr('data-thought-index') : '') || '')
      : '';

    if (!$stream.length) {
      return;
    }

    let renderSegments = Array.isArray(segments) ? segments.slice() : [];
    if (renderOptions.trustThoughtSegments === false) {
      renderSegments = renderSegments.filter(function removeUntrustedThought(segment) {
        return !(segment && segment.type === 'thought');
      });
    }
    if (deepResearchUi && typeof deepResearchUi.ingestTimeline === 'function') {
      deepResearchUi.ingestTimeline(renderSegments);
    }
    const forceReasoningShell = shouldUseReasoningShell(
      $msgRow,
      renderSegments,
      renderOptions.rawText || '',
      renderOptions
    );
    const pendingToolInReasoning = !!renderOptions.pendingTool && forceReasoningShell;

    if (pendingToolInReasoning) {
      renderSegments.push({ type: 'tool_pending' });
    }

    if (renderSegments.length === 0) {
      if (!renderOptions.pendingTool) {
        stopReasoningLabelAnimator($msgRow);
        setMessageSources($msgRow, []);
        $stream.hide().empty();
        $bubble.html('');
        $msgRow.removeAttr('data-expanded-thoughts');
        $msgRow.removeAttr('data-expanded-searches');
        $msgRow.removeAttr('data-expanded-search-keys');
        $msgRow.removeData('expandedSearchKeys');
        $msgRow.removeAttr('data-expanded-writes');
        $msgRow.removeAttr('data-expanded-edits');
        $msgRow.removeAttr('data-expanded-sandboxes');
        $msgRow.removeData('toolSegments');
        return;
      }
      renderSegments = [];
    }

    if (browserPortalUi && typeof browserPortalUi.enhanceSegments === 'function') {
      renderSegments = browserPortalUi.enhanceSegments(renderSegments, renderOptions);
    }

    segments = renderSegments;

    const expandedThoughts = getExpandedThoughtIndices($msgRow);
    const expandedSearches = getExpandedSearchIndices($msgRow);
    const expandedSearchKeys = getExpandedSearchKeys($msgRow);
    const expandedWrites = getExpandedWriteIndices($msgRow);
    const expandedEdits = getExpandedEditIndices($msgRow);
    const expandedSandboxes = getExpandedSandboxIndices($msgRow);
    let thoughtIndex = -1;
    let toolSegmentIndex = 0;
    const citationRegistry = createCitationRegistry();
    addAllSearchSourcesToCitationRegistry(citationRegistry, segments);
    setMessageSources($msgRow, messageSourcesFromSegments(segments, citationRegistry));
    const toolSegments = segments.filter(function onlyToolSegments(segment) {
      return segment.type === 'tool';
    });
    const standaloneDeepResearchCards = [];
    let standaloneToolIndex = 0;
    segments.forEach(function collectStandaloneResearchCard(segment) {
      if (!segment || segment.type !== 'tool') {
        return;
      }
      if (deepResearchUi && deepResearchUi.isDeepResearchSegment(segment)) {
        standaloneDeepResearchCards.push(deepResearchUi.renderCard(segment, standaloneToolIndex));
      }
      standaloneToolIndex += 1;
    });
    const lastToolSegmentIndex = segments.reduce(function findLastToolIndex(lastIndex, segment, index) {
      return segment && segment.type === 'tool' ? index : lastIndex;
    }, -1);
    let reasoningGroupIndex = -1;
    const blocks = [];
    function pushBlock(key, html) {
      if (!html || !String(html).trim()) {
        return;
      }
      blocks.push(createActivityBlock(key, html));
    }

    function pushTextSegmentBlock(segment, segmentIndex) {
      const liveClass = renderOptions.streaming === true ? ' is-live' : '';
      const revealTailChars = Number(segment && segment.streamRevealTailChars) || 0;
      const revealingClass = revealTailChars > 0 ? ' is-revealing' : '';
      const renderedText = useMarkdown
        ? renderMarkdownSegment(segment.content, citationRegistry)
        : renderPlainTextSegment(segment.content);
      const visibleText = revealTailChars > 0
        ? decorateStreamingRevealTail(renderedText, revealTailChars, segment.streamRevealFrame)
        : renderedText;
      pushBlock(
        `text-${segmentIndex}`,
        `
        <div class="msg-stream-text${liveClass}${revealingClass}">
          <div class="markdown-body${useMarkdown ? '' : ' is-streaming'}">${visibleText}</div>
        </div>
      `
      );
    }

    if (forceReasoningShell) {
      const reasoningItems = [];
      let activeToolSegmentIndex = 0;
      const lastActivitySegmentIndex = segments.reduce(function findLastActivityIndex(lastIndex, segment, index) {
        return segment
          && (segment.type === 'thought' || segment.type === 'tool' || segment.type === 'tool_pending')
          && !isIgnoredReasoningActivity(segment)
          ? index
          : lastIndex;
      }, -1);
      const finalReasoningAnchorTextIndex = segments.reduce(function findReasoningAnchor(lastIndex, segment, index) {
        if (hideTextSegments || index >= lastActivitySegmentIndex || !segment) {
          return lastIndex;
        }
        if (segment.type === 'tool_activity') {
          return lastIndex;
        }
        return segment.type === 'thought' || segment.type === 'tool' || segment.type === 'tool_pending'
          ? lastIndex
          : index;
      }, -1);
      const reasoningAnchorTextIndex = finalReasoningAnchorTextIndex;
      const isStreamingTextAfterReasoning = renderOptions.streaming === true
        && lastActivitySegmentIndex >= 0
        && segments.slice(lastActivitySegmentIndex + 1).some(function hasTrailingText(segment) {
          return segment
            && segment.type !== 'thought'
            && segment.type !== 'tool'
            && segment.type !== 'tool_pending'
            && String(segment.content || '').trim();
        });

      function renderActiveReasoningBlock() {
        return renderReasoningGroup(
          reasoningItems,
          0,
          expandedThoughts.has(0),
          reasoningTogglePresentation(reasoningItems, {
            streaming: renderOptions.streaming === true,
            reasoningComplete: isStreamingTextAfterReasoning
          }),
          { forceWrapper: true }
        );
      }

      segments.forEach(function collectReasoningItem(segment) {
        if (!segment) {
          return;
        }

        if (segment.type === 'browser_portal') {
          return;
        }

        if (segment.type === 'tool_activity') {
          return;
        }

        if (segment.type === 'thought') {
          reasoningItems.push(segment);
          return;
        }

        if (segment.type === 'tool_pending') {
          reasoningItems.push({ type: 'tool_pending' });
          return;
        }

        if (segment.type === 'tool') {
          const currentToolIndex = activeToolSegmentIndex;
          if (isSharedFileToolSegment(segment)) {
            activeToolSegmentIndex += 1;
            return;
          }
          if (hasCitationSources(segment)) {
            addSearchSourcesToCitationRegistry(citationRegistry, segment);
          }
          if (deepResearchUi && deepResearchUi.isDeepResearchSegment(segment)) {
            activeToolSegmentIndex += 1;
            return;
          }
          reasoningItems.push({
            type: 'tool',
            segment,
            toolIndex: currentToolIndex,
            expanded: isSearchToolSegment(segment)
              ? expandedSearches.has(currentToolIndex)
              : (isWriteToolSegment(segment)
                ? expandedWrites.has(currentToolIndex)
                : (isEditToolSegment(segment) ? expandedEdits.has(currentToolIndex) : false))
          });
          activeToolSegmentIndex += 1;
        }
      });

      // Deep Research is a first-class chat artifact. Keep it outside the
      // collapsible Thought/Reasoning shell regardless of model output order.
      if (standaloneDeepResearchCards.length) {
        pushBlock(
          'deep-research-pinned',
          `<div class="msg-tool-only-group msg-tool-only-group--deep-research">${standaloneDeepResearchCards.join('')}</div>`
        );
      }
      if (reasoningItems.length > 0 && reasoningAnchorTextIndex === -1) {
        pushBlock('reasoning-active', renderActiveReasoningBlock());
      }

      for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex += 1) {
        const segment = segments[segmentIndex];
        if (!segment) {
          continue;
        }

        if (segment.type === 'tool_activity') {
          continue;
        }

        if (segment.type === 'thought' || segment.type === 'tool' || segment.type === 'tool_pending') {
          continue;
        }

        if (segment.type === 'browser_portal') {
          pushBlock(segment.key || `browser-portal-${segmentIndex}`, segment.html);
          if (segmentIndex === reasoningAnchorTextIndex && reasoningItems.length > 0) {
            pushBlock('reasoning-active', renderActiveReasoningBlock());
          }
          continue;
        }

        if (!hideTextSegments) {
          pushTextSegmentBlock(segment, segmentIndex);
        }

        if (segmentIndex === reasoningAnchorTextIndex && reasoningItems.length > 0) {
          pushBlock('reasoning-active', renderActiveReasoningBlock());
        }
      }

      const pinnedSharedCards = collectPinnedSharedFileCards(segments);
      if (pinnedSharedCards.length) {
        pushBlock(
          'shared-files-pinned',
          `<div class="msg-tool-only-group msg-tool-only-group--shared-files">${pinnedSharedCards.join('')}</div>`
        );
      }

      $bubble.empty();
      applyActivityBlocks($stream, blocks);
      syncReasoningLabelAnimator($msgRow, renderOptions.streaming === true);
      $stream.css('display', 'flex');
      setExpandedThoughtIndices($msgRow, expandedThoughts);
      setExpandedSearchIndices($msgRow, expandedSearches);
      setExpandedSearchKeys($msgRow, expandedSearchKeys);
      setExpandedWriteIndices($msgRow, expandedWrites);
      setExpandedEditIndices($msgRow, expandedEdits);
      $msgRow.data('toolSegments', toolSegments);
      $msgRow.data('toolActivityEvents', segments
        .filter(function onlyToolActivity(segment) { return segment && segment.type === 'tool_activity'; })
        .map(function activityEvent(segment) { return segment.event; }));
      if (renderOptions.streaming !== true) {
        hydrateMermaidDiagrams($stream);
      hydrateImageTokens($stream);
        hydrateImageTokens($stream);
      }

      if (shouldSyncOpenDrawer) {
        const $nextActiveWrapper = $msgRow
          .find('.msg-thoughts-wrapper, .msg-reasoning-wrapper')
          .filter(function matchActiveReasoningWrapper() {
            return String($(this).attr('data-thought-index') || '') === '0';
          })
          .first();

        if ($nextActiveWrapper.length) {
          $activeReasoningWrapper = $nextActiveWrapper;
          $activeReasoningMessageRow = $msgRow;
          $activeReasoningIndex = '0';
          $nextActiveWrapper.addClass('is-active');
          $nextActiveWrapper.find('.msg-reasoning-toggle, .msg-thoughts-toggle').attr('aria-expanded', 'true');
          syncReasoningDrawerFromWrapper($nextActiveWrapper);
        } else {
          closeReasoningDrawer();
        }
      }
      return;
    }

    if (standaloneDeepResearchCards.length) {
      pushBlock(
        'deep-research-pinned',
        `<div class="msg-tool-only-group msg-tool-only-group--deep-research">${standaloneDeepResearchCards.join('')}</div>`
      );
    }

    for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex += 1) {
      const segment = segments[segmentIndex];

      if (segment && segment.type === 'tool_activity') {
        continue;
      }

      if (segment && segment.type === 'browser_portal') {
        pushBlock(segment.key || `browser-portal-${segmentIndex}`, segment.html);
        continue;
      }

      if (segment.type === 'thought' || segment.type === 'tool' || segment.type === 'tool_pending') {
        const groupStartIndex = segmentIndex;
        const groupItems = [];
        reasoningGroupIndex += 1;

        while (
          segmentIndex < segments.length
          && (
            segments[segmentIndex].type === 'thought'
            || segments[segmentIndex].type === 'tool'
            || segments[segmentIndex].type === 'tool_pending'
          )
        ) {
          const groupSegment = segments[segmentIndex];
          if (groupSegment.type === 'thought') {
            thoughtIndex += 1;
            groupItems.push(groupSegment);
          } else if (groupSegment.type === 'tool_pending') {
            groupItems.push({ type: 'tool_pending' });
          } else {
            if (suppressSharedFileToolRows && isSharedFileToolSegment(groupSegment)) {
              segmentIndex += 1;
              continue;
            }
            const currentToolIndex = toolSegmentIndex;
            if (deepResearchUi && deepResearchUi.isDeepResearchSegment(groupSegment)) {
              toolSegmentIndex += 1;
              segmentIndex += 1;
              continue;
            }
            if (hasCitationSources(groupSegment)) {
              addSearchSourcesToCitationRegistry(citationRegistry, groupSegment);
            }
            groupItems.push({
              type: 'tool',
              segment: groupSegment,
              toolIndex: currentToolIndex,
              expanded: isSearchToolSegment(groupSegment)
                ? expandedSearches.has(currentToolIndex)
                : (isWriteToolSegment(groupSegment)
                  ? expandedWrites.has(currentToolIndex)
                  : (isEditToolSegment(groupSegment)
                    ? expandedEdits.has(currentToolIndex)
                    : (isSandboxToolSegment(groupSegment) ? expandedSandboxes.has(currentToolIndex) : false)))
            });
            toolSegmentIndex += 1;
          }
          segmentIndex += 1;
        }

        segmentIndex -= 1;
        if (!groupItems.length) {
          continue;
        }
        if (
          groupItems.some(function hasToolItem(item) {
            return item && (item.type === 'tool' || item.type === 'tool_pending');
          })
          || groupStartIndex > lastToolSegmentIndex
          || forceReasoningShell
        ) {
          pushBlock(
            `reasoning-${reasoningGroupIndex}`,
            renderReasoningGroup(
              groupItems,
              reasoningGroupIndex,
              expandedThoughts.has(reasoningGroupIndex),
              reasoningTogglePresentation(groupItems, {
                streaming: renderOptions.streaming === true,
                reasoningComplete: false
              }),
              { forceWrapper: forceReasoningShell }
            )
          );
        }
        continue;
      }

      if (hideTextSegments) {
        continue;
      }

      pushTextSegmentBlock(segment, segmentIndex);
    }

    if (renderOptions.pendingTool && !pendingToolInReasoning) {
      pushBlock('tool-pending', renderPendingToolCallPlaceholder());
    }

    const pinnedSharedCards = collectPinnedSharedFileCards(segments);
    if (pinnedSharedCards.length) {
      pushBlock(
        'shared-files-pinned',
        `<div class="msg-tool-only-group msg-tool-only-group--shared-files">${pinnedSharedCards.join('')}</div>`
      );
    }

    $bubble.empty();
    applyActivityBlocks($stream, blocks);
    syncReasoningLabelAnimator($msgRow, renderOptions.streaming === true);
    $stream.css('display', 'flex');
    setExpandedThoughtIndices($msgRow, expandedThoughts);
    setExpandedSearchIndices($msgRow, expandedSearches);
    setExpandedSearchKeys($msgRow, expandedSearchKeys);
    setExpandedWriteIndices($msgRow, expandedWrites);
    setExpandedEditIndices($msgRow, expandedEdits);
    setExpandedSandboxIndices($msgRow, expandedSandboxes);
    $msgRow.data('toolSegments', toolSegments);
    $msgRow.data('toolActivityEvents', segments
      .filter(function onlyToolActivity(segment) { return segment && segment.type === 'tool_activity'; })
      .map(function activityEvent(segment) { return segment.event; }));
    if (renderOptions.streaming !== true) {
      hydrateMermaidDiagrams($stream);
      hydrateImageTokens($stream);
    }

    if (shouldSyncOpenDrawer) {
      const $nextActiveWrapper = $msgRow
        .find('.msg-thoughts-wrapper, .msg-reasoning-wrapper')
        .filter(function matchActiveReasoningWrapper() {
          return String($(this).attr('data-thought-index') || '') === activeReasoningIndex;
        })
        .first();

      if ($nextActiveWrapper.length) {
        $activeReasoningWrapper = $nextActiveWrapper;
        $activeReasoningMessageRow = $msgRow;
        $activeReasoningIndex = activeReasoningIndex;
        $nextActiveWrapper.addClass('is-active');
        $nextActiveWrapper.find('.msg-reasoning-toggle, .msg-thoughts-toggle').attr('aria-expanded', 'true');
        syncReasoningDrawerFromWrapper($nextActiveWrapper);
      } else {
        closeReasoningDrawer();
      }
    }
  }

  // Handle search segment count.
  function searchSegmentCount(segments) {
    return (Array.isArray(segments) ? segments : []).filter(function onlySearchSegment(segment) {
      return segment && segment.type === 'tool' && isSearchToolSegment(segment);
    }).length;
  }

  // Handle clear search batch hold.
  function clearSearchBatchHold($msgRow) {
    const holdState = $msgRow.data('searchBatchHoldState');
    if (holdState && holdState.timer) {
      window.clearTimeout(holdState.timer);
    }

    $msgRow.removeData('searchBatchHoldState');
    $msgRow.removeData('searchBatchHoldRaw');
  }

  // Handle clear pre tool text hold.
  function clearPreToolTextHold($msgRow) {
    const holdState = $msgRow.data('preToolTextHoldState');
    if (holdState && holdState.timer) {
      window.clearTimeout(holdState.timer);
    }

    $msgRow.removeData('preToolTextHoldState');
    $msgRow.removeData('preToolTextHoldRaw');
  }

  // Handle should hold pre tool text.
  function shouldHoldPreToolText($msgRow, parsed, rawText) {
    const segments = Array.isArray(parsed && parsed.segments) ? parsed.segments : [];
    if (!segments.length || segments.some(function hasTool(segment) { return segment && segment.type === 'tool'; })) {
      clearPreToolTextHold($msgRow);
      return false;
    }

    const hasRenderableText = segments.some(function hasTextLikeSegment(segment) {
      return segment
        && (segment.type === 'text' || segment.type === 'thought')
        && String(segment.content || '').trim();
    });
    if (!hasRenderableText) {
      clearPreToolTextHold($msgRow);
      return false;
    }

    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const holdState = $msgRow.data('preToolTextHoldState') || {};
    if (!holdState.startedAt) {
      holdState.startedAt = now;
    }
    if (holdState.released) {
      return false;
    }

    const remaining = PRE_TOOL_TEXT_HOLD_MS - (now - holdState.startedAt);
    $msgRow.data('preToolTextHoldRaw', rawText);

    if (remaining <= 0) {
      holdState.released = true;
      if (holdState.timer) {
        window.clearTimeout(holdState.timer);
        holdState.timer = null;
      }
      $msgRow.data('preToolTextHoldState', holdState);
      return false;
    }

    if (holdState.timer) {
      window.clearTimeout(holdState.timer);
    }
    holdState.timer = window.setTimeout(function renderHeldPreToolText() {
      const latestRaw = $msgRow.data('preToolTextHoldRaw');
      const latestState = $msgRow.data('preToolTextHoldState') || {};
      latestState.timer = null;
      latestState.released = true;
      $msgRow.data('preToolTextHoldState', latestState);
      renderMessageStream($msgRow, latestRaw || rawText);
    }, Math.max(0, remaining + 16));
    $msgRow.data('preToolTextHoldState', holdState);
    return true;
  }

  // Handle should hold streaming search batch.
  function shouldHoldStreamingSearchBatch($msgRow, parsed, rawText) {
    const count = searchSegmentCount(parsed.segments);
    if (count === 0) {
      clearSearchBatchHold($msgRow);
      return false;
    }

    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const holdState = $msgRow.data('searchBatchHoldState') || {};
    if (holdState.count !== count) {
      holdState.count = count;
      holdState.changedAt = now;
    }
    if (!holdState.changedAt) {
      holdState.changedAt = now;
    }

    const holdMs = count <= 1 ? SEARCH_BATCH_FIRST_CALL_HOLD_MS : SEARCH_BATCH_MULTI_CALL_SETTLE_MS;
    const remaining = holdMs - (now - holdState.changedAt);
    $msgRow.data('searchBatchHoldRaw', rawText);

    if (remaining <= 0) {
      if (holdState.timer) {
        window.clearTimeout(holdState.timer);
        holdState.timer = null;
      }
      $msgRow.data('searchBatchHoldState', holdState);
      return false;
    }

    if (holdState.timer) {
      window.clearTimeout(holdState.timer);
    }
    holdState.timer = window.setTimeout(function renderHeldSearchBatch() {
      const latestRaw = $msgRow.data('searchBatchHoldRaw');
      const latestState = $msgRow.data('searchBatchHoldState') || {};
      latestState.timer = null;
      $msgRow.data('searchBatchHoldState', latestState);
      renderMessageStream($msgRow, latestRaw || rawText);
    }, Math.max(0, remaining + 16));
    $msgRow.data('searchBatchHoldState', holdState);
    return true;
  }

  // Return the text segments that belong to the streamed reasoning/tool
  // sequence. Text before the first activity is ordinary answer text and must
  // not suddenly become buffered if a reasoning marker appears later.
  function streamingRevealTextKeys($msgRow, segments) {
    const keys = new Set();
    let sawReasoningActivity = !!$msgRow.data('sawReasoning');
    (Array.isArray(segments) ? segments : []).forEach(function collectRevealKey(segment, index) {
      if (!segment) {
        return;
      }
      if (
        segment.type === 'thought'
        || segment.type === 'tool_pending'
        || (segment.type === 'tool' && !isIgnoredReasoningActivity(segment))
      ) {
        sawReasoningActivity = true;
        return;
      }
      if (sawReasoningActivity && segment.type === 'text') {
        keys.add(`text-${index}`);
      }
    });
    return keys;
  }

  // Advance all buffered text cursors. The normal cadence is a little faster
  // than the Thought-label preview; final catch-up is bounded so actions do not
  // remain in a streaming state after the network response has closed.
  function advanceStreamingTextRevealState(revealState, now) {
    let pending = false;
    revealState.entries.forEach(function advanceEntry(entry) {
      const targetCharacters = Array.from(entry.target || '');
      const lag = Math.max(0, targetCharacters.length - entry.visibleLength);
      entry.revealedNow = 0;
      if (!lag) {
        entry.lastAdvancedAt = now;
        return;
      }

      if (now < entry.readyAt) {
        pending = true;
        return;
      }

      const interval = revealState.finalizing
        ? STREAM_TEXT_REVEAL_FINAL_CHARACTER_MS
        : STREAM_TEXT_REVEAL_CHARACTER_MS;
      const elapsedFrom = Math.max(entry.lastAdvancedAt || entry.readyAt, entry.readyAt);
      const elapsed = Math.max(0, now - elapsedFrom);
      entry.credit = (entry.credit || 0) + (elapsed / interval);
      let revealCount = Math.floor(entry.credit);
      if (revealState.finalizing) {
        revealCount = Math.max(revealCount, Math.ceil(lag / 10));
      }
      revealCount = Math.min(lag, revealCount);
      entry.lastAdvancedAt = now;
      if (!revealCount) {
        pending = true;
        return;
      }

      entry.credit = Math.max(0, entry.credit - revealCount);
      entry.visibleLength += revealCount;
      entry.revealedNow = revealCount;
      entry.revealFrame = (entry.revealFrame || 0) + 1;
      if (entry.visibleLength < targetCharacters.length) {
        pending = true;
      }
    });
    return pending || Array.from(revealState.entries.values()).some(function hasRemainingText(entry) {
      return entry.visibleLength < Array.from(entry.target || '').length;
    });
  }

  // Build one display snapshot while retaining the complete segment list for
  // layout. Empty buffered text blocks already occupy their correct ordering,
  // allowing the reasoning toggle to move before the first character appears.
  function streamingTextRevealSnapshot(revealState) {
    return revealState.segments.map(function revealSegment(segment, index) {
      if (!segment || segment.type !== 'text') {
        return segment;
      }
      const entry = revealState.entries.get(`text-${index}`);
      if (!entry) {
        return segment;
      }
      const characters = Array.from(entry.target || '');
      return {
        ...segment,
        content: characters.slice(0, entry.visibleLength).join(''),
        streamRevealTailChars: Math.min(STREAM_TEXT_REVEAL_TAIL_CHARS, entry.revealedNow || 0),
        streamRevealFrame: entry.revealFrame || 0
      };
    });
  }

  function clearStreamingTextReveal($msgRow) {
    const revealState = $msgRow.data('streamingTextRevealState');
    if (revealState && revealState.timer) {
      window.clearTimeout(revealState.timer);
    }
    $msgRow.removeData('streamingTextRevealState');
  }

  function renderMessageHtmlNow($msgRow, rawText, parsed) {
    clearStreamingTextReveal($msgRow);
    const safeParsed = parsed || parseMessageTimeline(rawText);
    renderActivityTimeline($msgRow, safeParsed.segments, { rawText });
    $msgRow.find('.msg-bubble').attr('data-raw', rawText).attr('data-copy', safeParsed.visibleText);
  }

  // Paint the next buffered frame without waiting for another network chunk.
  function scheduleStreamingTextRevealFrame($msgRow) {
    const revealState = $msgRow.data('streamingTextRevealState');
    if (!revealState || revealState.timer) {
      return;
    }
    const hasPendingText = Array.from(revealState.entries.values()).some(function hasPendingEntry(entry) {
      return entry.visibleLength < Array.from(entry.target || '').length;
    });
    if (!hasPendingText) {
      return;
    }

    revealState.timer = window.setTimeout(function renderStreamingTextRevealFrame() {
      const currentState = $msgRow.data('streamingTextRevealState');
      if (!currentState || currentState !== revealState) {
        return;
      }
      currentState.timer = null;
      if (!$msgRow[0] || (typeof document !== 'undefined' && !document.documentElement.contains($msgRow[0]))) {
        clearStreamingTextReveal($msgRow);
        return;
      }

      const messagesArea = dom.$messagesArea && dom.$messagesArea[0];
      const shouldFollowReveal = !!messagesArea
        && messagesArea.scrollHeight - messagesArea.clientHeight <= messagesArea.scrollTop + 50;
      const pending = advanceStreamingTextRevealState(currentState, Date.now());
      renderActivityTimeline($msgRow, streamingTextRevealSnapshot(currentState), {
        rawText: currentState.rawText,
        streaming: true
      });
      $msgRow.find('.msg-bubble')
        .attr('data-raw', currentState.rawText)
        .attr('data-copy', currentState.visibleText);
      if (shouldFollowReveal) {
        scrollBottom();
      }

      if (!pending && currentState.finalizing) {
        renderMessageHtmlNow(
          $msgRow,
          currentState.rawText,
          { segments: currentState.segments, visibleText: currentState.visibleText }
        );
        return;
      }
      scheduleStreamingTextRevealFrame($msgRow);
    }, STREAM_TEXT_REVEAL_FRAME_MS);
  }

  // Update targets from the latest parse and return the currently revealed
  // segment snapshot. Structural tool changes are applied immediately.
  function bufferStreamingTextSegments($msgRow, parsed, rawText, finalizing) {
    const reduceMotion = typeof window !== 'undefined'
      && typeof window.matchMedia === 'function'
      && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) {
      clearStreamingTextReveal($msgRow);
      return null;
    }
    const revealKeys = streamingRevealTextKeys($msgRow, parsed.segments);
    let revealState = $msgRow.data('streamingTextRevealState');
    if (!revealState && !revealKeys.size) {
      return null;
    }
    if (!revealState) {
      revealState = {
        entries: new Map(),
        segments: [],
        rawText: '',
        visibleText: '',
        finalizing: false,
        timer: null
      };
      $msgRow.data('streamingTextRevealState', revealState);
    }

    const now = Date.now();
    const activeKeys = new Set();
    parsed.segments.forEach(function updateRevealTarget(segment, index) {
      const key = `text-${index}`;
      if (!segment || segment.type !== 'text' || !revealKeys.has(key)) {
        return;
      }
      activeKeys.add(key);
      const target = String(segment.content || '');
      const targetLength = Array.from(target).length;
      let entry = revealState.entries.get(key);
      if (!entry) {
        entry = {
          target,
          visibleLength: 0,
          revealedNow: 0,
          readyAt: now + STREAM_TEXT_REVEAL_INITIAL_DELAY_MS,
          lastAdvancedAt: now + STREAM_TEXT_REVEAL_INITIAL_DELAY_MS,
          credit: 0,
          revealFrame: 0
        };
        revealState.entries.set(key, entry);
        return;
      }

      const oldTarget = String(entry.target || '');
      if (!target.startsWith(oldTarget)) {
        entry.visibleLength = Math.min(entry.visibleLength, targetLength);
      }
      entry.target = target;
      entry.visibleLength = Math.min(entry.visibleLength, targetLength);
      entry.revealedNow = 0;
    });
    Array.from(revealState.entries.keys()).forEach(function removeStaleEntry(key) {
      if (!activeKeys.has(key)) {
        revealState.entries.delete(key);
      }
    });

    revealState.segments = parsed.segments;
    revealState.rawText = rawText;
    revealState.visibleText = parsed.visibleText;
    revealState.finalizing = revealState.finalizing || finalizing === true;
    advanceStreamingTextRevealState(revealState, now);
    scheduleStreamingTextRevealFrame($msgRow);
    return streamingTextRevealSnapshot(revealState);
  }

  // Parse and render one assistant transcript string.
  function renderMessageHtml($msgRow, rawText) {
    clearSearchBatchHold($msgRow);
    clearPreToolTextHold($msgRow);
    const parsed = parseMessageTimeline(rawText);
    const revealState = $msgRow.data('streamingTextRevealState');
    if (revealState) {
      const displaySegments = bufferStreamingTextSegments($msgRow, parsed, rawText, true);
      const activeRevealState = $msgRow.data('streamingTextRevealState');
      if (!activeRevealState) {
        renderMessageHtmlNow($msgRow, rawText, parsed);
        return;
      }
      const hasPendingText = Array.from(activeRevealState.entries.values()).some(function hasPendingEntry(entry) {
        return entry.visibleLength < Array.from(entry.target || '').length;
      });
      if (!hasPendingText) {
        renderMessageHtmlNow($msgRow, rawText, parsed);
        return;
      }
      renderActivityTimeline($msgRow, displaySegments || parsed.segments, { rawText, streaming: true });
      $msgRow.find('.msg-bubble').attr('data-raw', rawText).attr('data-copy', parsed.visibleText);
      return;
    }
    renderMessageHtmlNow($msgRow, rawText, parsed);
  }

  // Parse and render one assistant transcript during active streaming.
  function renderMessageStream($msgRow, rawText) {
    const partialMarker = trailingPartialActivityMarkerInfo(rawText);
    const safeRawText = partialMarker ? stripTrailingPartialActivityMarker(rawText) : rawText;
    const hasOpenTool = hasOpenToolPayload(safeRawText);
    const hasPartialToolMarker = !!(partialMarker && partialMarker.isTool);
    const renderRawText = hasOpenTool ? stripOpenToolPayload(safeRawText) : safeRawText;
    const parsed = parseMessageTimeline(renderRawText);
    const hasPendingToolPayload = hasOpenTool || hasPartialToolMarker;

    if (hasPendingToolPayload) {
      clearSearchBatchHold($msgRow);
      clearPreToolTextHold($msgRow);
      const displaySegments = bufferStreamingTextSegments($msgRow, parsed, renderRawText, false);
      renderActivityTimeline($msgRow, displaySegments || parsed.segments, {
        markdown: false,
        pendingTool: true,
        rawText: renderRawText,
        streaming: true
      });
      $msgRow.find('.msg-bubble').attr('data-raw', rawText).attr('data-copy', parsed.visibleText);
      return;
    }

    if (hasThoughtSegments(parsed.segments) && !$msgRow.data('responseStartedAt')) {
      $msgRow.data('responseStartedAt', Date.now());
    }
    if (shouldHoldPreToolText($msgRow, parsed, rawText)) {
      $msgRow.find('.msg-bubble').attr('data-raw', rawText).attr('data-copy', parsed.visibleText);
      return;
    }
    if (shouldHoldStreamingSearchBatch($msgRow, parsed, rawText)) {
      $msgRow.find('.msg-bubble').attr('data-raw', rawText).attr('data-copy', parsed.visibleText);
      return;
    }

    const displaySegments = bufferStreamingTextSegments($msgRow, parsed, renderRawText, false);
    renderActivityTimeline($msgRow, displaySegments || parsed.segments, { rawText: renderRawText, streaming: true });
    $msgRow.find('.msg-bubble').attr('data-raw', rawText).attr('data-copy', parsed.visibleText);
  }

  // Ask the backend to reveal one validated shared file in the host file manager.
  async function revealSharedFile($button) {
    const path = String($button && $button.attr('data-shared-file-path') || '').trim();
    if (!path || !$button || $button.attr('aria-disabled') === 'true') {
      return;
    }

    $button.attr({ 'aria-busy': 'true', 'aria-disabled': 'true' }).addClass('is-busy');
    try {
      await postJson('/api/shared-file/reveal/', { path });
    } catch (error) {
      await messageDialog(
        t('errors.generic', null, 'Something went wrong'),
        error && error.message ? error.message : String(error)
      );
    } finally {
      $button.removeAttr('aria-busy aria-disabled').removeClass('is-busy');
    }
  }

  // Open the inspector for a delegated tool-card click.
  function sourceMessageRowForActivityCard($card) {
    const $row = $card.closest('.msg');
    if ($row.length) {
      return $row;
    }

    const $drawerBody = $card.closest('#reasoningDrawerBody');
    if ($drawerBody.length) {
      const $storedRow = $drawerBody.data('messageRow');
      if ($storedRow && $storedRow.length) {
        return $storedRow;
      }
      if ($activeReasoningMessageRow && $activeReasoningMessageRow.length) {
        return $activeReasoningMessageRow;
      }
    }

    return $();
  }

  // Open the tool inspector for one activity card.
  function openToolInspectorFromCard($card) {
    const index = parseInt($card.attr('data-tool-segment-index') || '-1', 10);
    if (!Number.isInteger(index) || index < 0) {
      return;
    }

    const $message = $card.closest('.msg');
    let toolSegments = $message.length ? ($message.data('toolSegments') || []) : [];
    if (!toolSegments.length) {
      const $drawerBody = $card.closest('#reasoningDrawerBody');
      toolSegments = ($drawerBody.length ? $drawerBody.data('toolSegments') : null) || $('#reasoningDrawerBody').data('toolSegments') || [];
    }

    const segment = toolSegments[index];
    if (segment) {
      if (deepResearchUi && deepResearchUi.isDeepResearchSegment(segment)) {
        deepResearchUi.openSegment(segment);
        return;
      }
      toolInspector.open(segment);
    }
  }

  // Expand or collapse extra search source chips.
  function toggleSearchSources($button) {
    const $card = $button.closest('.msg-search-card');
    const $row = sourceMessageRowForActivityCard($card);
    const expanded = !$card.hasClass('is-expanded');
    const moreCount = parseInt($button.attr('data-search-more-count') || '0', 10) || 0;
    const cardIndex = parseInt($card.attr('data-tool-segment-index') || '-1', 10);
    const expandedSearches = getExpandedSearchIndices($row);

    if (Number.isInteger(cardIndex) && cardIndex >= 0) {
      if (expanded) {
        expandedSearches.add(cardIndex);
      } else {
        expandedSearches.delete(cardIndex);
      }
      setExpandedSearchIndices($row, expandedSearches);
    }

    $card.toggleClass('is-expanded', expanded);
    $card.find('.msg-search-chip--more').attr('aria-expanded', expanded ? 'true' : 'false');
    $card.find('.msg-search-chip--more-collapsed .msg-search-chip-domain').text(`${MORE_LABEL} ${moreCount}`);
    $card.find('.msg-search-chip--more-expanded .msg-search-chip-domain').text(HIDE_LABEL);
  }

  // Expand or collapse one write preview card.
  function toggleWriteCard($card) {
    const $row = sourceMessageRowForActivityCard($card);
    const cardIndex = parseInt($card.attr('data-write-segment-index') || '-1', 10);
    const expandedWrites = getExpandedWriteIndices($row);
    const willExpand = !$card.hasClass('is-expanded');

    if (Number.isInteger(cardIndex) && cardIndex >= 0) {
      if (willExpand) {
        expandedWrites.add(cardIndex);
      } else {
        expandedWrites.delete(cardIndex);
      }
      setExpandedWriteIndices($row, expandedWrites);
    }

    // Update the existing card element in-place so the DOM node stays alive
    // and the next stream render (which also reads data-expanded-writes) agrees.
    $card.toggleClass('is-expanded', willExpand).attr('aria-expanded', willExpand ? 'true' : 'false');

    // Swap the preview content between collapsed and expanded line counts
    // by morphing the preview element directly.
    const toolSegments = $row.data('toolSegments') || $('#reasoningDrawerBody').data('toolSegments') || [];
    const segment = toolSegments[cardIndex];
    if (segment) {
      const $preview = $card.find('.msg-write-preview');
      const $newCard = $(renderWriteToolCard(segment, cardIndex, { expanded: willExpand }));
      const $newPreview = $newCard.find('.msg-write-preview');
      if ($preview.length && $newPreview.length) {
        morphDomChildren($preview[0], $newPreview[0]);
      }
    }
  }

  // Expand or collapse one edit preview card.
  function toggleEditCard($card) {
    const $row = sourceMessageRowForActivityCard($card);
    const cardIndex = parseInt($card.attr('data-edit-segment-index') || '-1', 10);
    const expandedEdits = getExpandedEditIndices($row);
    const willExpand = !$card.hasClass('is-expanded');

    if (Number.isInteger(cardIndex) && cardIndex >= 0) {
      if (willExpand) {
        expandedEdits.add(cardIndex);
      } else {
        expandedEdits.delete(cardIndex);
      }
      setExpandedEditIndices($row, expandedEdits);
    }

    $card.toggleClass('is-expanded', willExpand).attr('aria-expanded', willExpand ? 'true' : 'false');

    const toolSegments = $row.data('toolSegments') || $('#reasoningDrawerBody').data('toolSegments') || [];
    const segment = toolSegments[cardIndex];
    if (segment) {
      const $preview = $card.find('.msg-edit-preview');
      const $newCard = $(renderEditToolCard(segment, cardIndex, { expanded: willExpand }));
      const $newPreview = $newCard.find('.msg-edit-preview');
      if ($preview.length && $newPreview.length) {
        morphDomChildren($preview[0], $newPreview[0]);
      }
    }
  }

  // Expand or collapse one compact sandbox call.
  function toggleSandboxCard($toggle) {
    const $wrapper = $toggle.closest('.msg-sandbox-collapsible');
    const $row = sourceMessageRowForActivityCard($wrapper);
    const cardIndex = parseInt($wrapper.attr('data-sandbox-segment-index') || '-1', 10);
    const expandedSandboxes = getExpandedSandboxIndices($row);
    const willExpand = !$wrapper.hasClass('is-expanded');

    if (Number.isInteger(cardIndex) && cardIndex >= 0) {
      if (willExpand) {
        expandedSandboxes.add(cardIndex);
      } else {
        expandedSandboxes.delete(cardIndex);
      }
      setExpandedSandboxIndices($row, expandedSandboxes);
    }

    $wrapper.toggleClass('is-expanded', willExpand);
    $toggle.attr('aria-expanded', willExpand ? 'true' : 'false');
    $wrapper.find('.msg-sandbox-collapsible-content').first().prop('hidden', !willExpand);
  }

  // Expand or collapse one compression context card.
  function toggleCompressionContext($button) {
    const $card = $button.closest('.msg-compression-context');
    const $json = $card.find('.msg-compression-context-json').first();
    const isOpen = $json.is(':visible');
    $json.toggle(!isOpen);
    $button.text(isOpen ? 'Compressed context active' : 'Hide compressed context');
  }

  // Start panning inside an expanded write preview.
  function startWritePreviewPan(event, $preview) {
    if (event.button !== 1) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    const preview = $preview[0];
    if (!preview) {
      return;
    }

    const existingStop = $preview.data('writePreviewPanStop');
    if (typeof existingStop === 'function') {
      existingStop();
      return;
    }

    const anchorX = event.clientX;
    let currentX = event.clientX;
    let animationFrame = null;
    let isActive = true;
    let hasMoved = false;
    const startTime = Date.now();
    $preview.addClass('is-middle-panning');

    function scrollFrame() {
      if (!isActive) {
        return;
      }

      const distance = currentX - anchorX;
      if (Math.abs(distance) > 4) {
        preview.scrollLeft += distance * 0.18;
      }

      animationFrame = window.requestAnimationFrame(scrollFrame);
    }

    function onMouseMove(moveEvent) {
      moveEvent.preventDefault();
      currentX = moveEvent.clientX;
      if (Math.abs(currentX - anchorX) > 4) {
        hasMoved = true;
      }
    }

    function stopPan() {
      isActive = false;
      if (animationFrame !== null) {
        window.cancelAnimationFrame(animationFrame);
        animationFrame = null;
      }
      $preview.removeClass('is-middle-panning');
      $preview.removeData('writePreviewPanStop');
      $(document).off('.writePreviewPan');
    }

    function onMouseUp(upEvent) {
      if (upEvent.button !== 1) {
        return;
      }

      const isQuickClick = !hasMoved && Date.now() - startTime < 260;
      if (!isQuickClick) {
        stopPan();
      }
    }

    $(document).on('mousemove.writePreviewPan', onMouseMove);
    $(document).on('mouseup.writePreviewPan', onMouseUp);
    $(document).on('blur.writePreviewPan', stopPan);
    $(document).on('mousedown.writePreviewPan', function onNextMiddleDown(nextEvent) {
      if (nextEvent.target !== preview && !$(nextEvent.target).closest(preview).length) {
        nextEvent.preventDefault();
        stopPan();
      }
    });
    $preview.data('writePreviewPanStop', stopPan);
    animationFrame = window.requestAnimationFrame(scrollFrame);
  }


  // Message row rendering.
  // Render user text with slash commands shown as styled inline chips.
  // Mirrors the backend command grammar: /search, /tool <id>, /skill <name>.
  function buildUserTextSpan(text) {
    const $span = $('<span>');
    const value = String(text ?? '');
    const commandPattern = /(^|\s)\/(search(?![\w-])|(?:tool|skill)[ \t]+[\w][\w.-]*)/gi;
    let cursor = 0;
    let match;
    while ((match = commandPattern.exec(value)) !== null) {
      const commandStart = match.index + match[1].length;
      if (commandStart > cursor) {
        $span.append(document.createTextNode(value.slice(cursor, commandStart)));
      }
      const command = match[2];
      const label = /^search$/i.test(command)
        ? t('composer.searchCommandLabel', null, 'Web Search')
        : command.replace(/^(?:tool|skill)[ \t]+/i, '');
      $span.append($('<span class="msg-inline-command">').text(label));
      cursor = commandStart + 1 + command.length;
    }
    if (cursor < value.length) {
      $span.append(document.createTextNode(value.slice(cursor)));
    }
    return $span;
  }

  // Build one user or assistant message row.
  function buildMessageRow(role, text, attachments, timestamp, options) {
    const viewOptions = options || {};
    const isUser = role === 'user';
    const label = isUser ? 'You' : 'ASLM';
    const timeStr = timeNow(timestamp);
    const queuedBadge = isUser && viewOptions.queued
      ? '<span class="msg-status-pill">Queued</span>'
      : '';
    const messageKey = viewOptions.messageKey || '';
    const messageId = viewOptions.messageId || '';

    const branchLinks = Array.isArray(viewOptions.branchLinks) ? viewOptions.branchLinks : [];
    const branchLinksHtml = branchLinks.map(function renderBranchLink(link) {
      const direction = link && link.direction === 'to_origin' ? 'to_origin' : 'to_branch';
      const title = String(link && link.title || 'Chat');
      const label = direction === 'to_origin'
        ? t('messages.branchedFrom', { title }, `Branched from ${title}`)
        : t('messages.openBranch', { title }, `Branch: ${title}`);
      return `<a class="msg-branch-link" href="/chat/${escapeAttributeValue(link.chat_id || '')}/" data-chat-link="${escapeAttributeValue(link.chat_id || '')}">${escHtml(label)}</a>`;
    }).join('');

    let attachmentsHtml = '';

    const normalizedActivitySegments = Array.isArray(viewOptions.activitySegments)
      ? normalizeTrailingSharedFileShorthandSegments(viewOptions.activitySegments)
      : [];

    if (!isUser && isCompressionOnlyActivitySegments(normalizedActivitySegments)) {
      const markerHtml = normalizedActivitySegments
        .map(function renderMarker(segment, index) {
          return renderCompressionContextCard(segment, index);
        })
        .join('');
      return $(`
        <div class="msg msg-compression-marker" data-message-key="${escapeAttributeValue(messageKey)}"${messageId ? ` data-message-id="${messageId}"` : ''}>
          <div class="msg-compression-marker-body">${markerHtml}</div>
        </div>
      `);
    }

    if (isUser && attachments && attachments.length > 0) {
      attachmentsHtml = renderMessageAttachments(attachments, {
        side: 'user',
        source: 'upload',
        mode: 'message'
      });
    }

    const userBubbleAttachmentClass = isUser && attachmentsHtml ? ' msg-bubble--attachments' : '';

    const $row = $(`
      <div class="msg ${role}${viewOptions.queued ? ' is-queued' : ''}" data-message-key="${escapeAttributeValue(messageKey)}"${messageId ? ` data-message-id="${messageId}"` : ''}>
        <div class="msg-avatar">${isUser ? 'U' : 'A'}</div>
        <div class="msg-body">
          <div class="msg-meta">
            <span>${label}</span>
            <span>${timeStr}</span>
            ${queuedBadge}
          </div>
          ${!isUser ? '<div class="msg-activity-stream" style="display:none;"></div>' : ''}
          <div class="msg-bubble${userBubbleAttachmentClass}">${attachmentsHtml}</div>
          ${icons.buildMessageActionsHtml(isUser)}
        </div>
        ${branchLinksHtml ? `<div class="msg-branch-divider"><span></span><div>${branchLinksHtml}</div><span></span></div>` : ''}
      </div>
    `);

    if (isUser) {
      const $bubble = $row.find('.msg-bubble').attr('data-raw', text);
      const hasAttachmentCards = !!attachmentsHtml;
      if (hasAttachmentCards) {
        const body = String(text ?? '');
        if (body.length > 0) {
          $bubble.append(
            $('<div class="msg-bubble-caption"></div>').append(buildUserTextSpan(text))
          );
        }
      } else {
        $bubble.append(buildUserTextSpan(text));
      }
      $bubble.data('attachments', attachments || []);
    } else if (normalizedActivitySegments.length > 0) {
      $row.find('.msg-bubble').attr('data-raw', text);
      renderActivityTimeline($row, normalizedActivitySegments, {
        rawText: text,
        reasoningMode: viewOptions.reasoningMode === true,
        trustThoughtSegments: viewOptions.reasoningMode === true || hasReasoningMarker(text)
      });
    } else {
      renderMessageHtml($row, text);
    }

    return $row;
  }

  // Append one user or assistant message to the stream.

  // Public message API.
  function appendMessage(role, text, attachments, timestamp, options) {
    const viewOptions = options || {};
    const $row = buildMessageRow(role, text, attachments, timestamp, viewOptions);
    dom.$messagesInner.append($row);
    if (!viewOptions.deferSideEffects) {
      updateRegenButtons();
      scrollBottom();
    }
    return $row;
  }

  // Append a batch of stored messages with one DOM insertion.
  function appendMessages(messages, options) {
    const batchOptions = options || {};
    const rows = [];
    const fragment = document.createDocumentFragment();

    (messages || []).forEach(function buildStoredMessage(message) {
      const $row = buildMessageRow(
        message.role,
        message.text,
        message.attachments,
        message.timestamp,
        message.options || {}
      );
      rows.push($row);
      fragment.appendChild($row[0]);
    });

    dom.$messagesInner[0].appendChild(fragment);
    updateRegenButtons();
    if (batchOptions.scroll !== false) {
      scrollBottom();
    }
    return rows;
  }

  // Toggle the queued badge for one user row.
  function setQueuedMessageState($row, queued) {
    if (!$row || !$row.length) {
      return;
    }

    $row.toggleClass('is-queued', !!queued);

    const $meta = $row.find('.msg-meta');
    let $badge = $meta.find('.msg-status-pill');

    if (queued) {
      if (!$badge.length) {
        $badge = $('<span class="msg-status-pill">Queued</span>');
        $meta.append($badge);
      }
    } else {
      $badge.remove();
    }
  }

  // Append a temporary assistant typing row.
  function appendTyping(timestamp) {
    const timeStr = timeNow(timestamp);
    const startedAt = timestamp ? new Date(timestamp).getTime() : Date.now();
    const $row = $(`
      <div class="msg assistant">
        <div class="msg-avatar">A</div>
        <div class="msg-body">
          <div class="msg-meta">
            <span>ASLM</span>
            <span>${timeStr}</span>
          </div>
          <div class="msg-activity-stream" style="display:none;"></div>
          <div class="msg-bubble">
            <div class="typing-indicator">
              <div class="typing-dot"></div>
              <div class="typing-dot"></div>
              <div class="typing-dot"></div>
            </div>
          </div>
        </div>
      </div>
    `);

    $row.data('responseStartedAt', Number.isFinite(startedAt) ? startedAt : Date.now());
    dom.$messagesInner.append($row);
    return $row;
  }


  // Clipboard helpers.
  // Copy text using the legacy textarea fallback.
  function fallbackCopy(text, onSuccess) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.cssText = 'position:fixed;top:-9999px;left:-9999px;opacity:0';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    try {
      if (document.execCommand('copy')) {
        onSuccess && onSuccess();
      }
    } catch (_error) {
      // Ignore legacy clipboard errors.
    }

    document.body.removeChild(textarea);
  }

  // Copy the visible message content to the clipboard.
  function copyMessage($button) {
    const $btn = $button || $();
    const $messageBody = $btn.closest('.msg-body');
    const $bubble = $messageBody.find('.msg-bubble');
    const rawText = $bubble.attr('data-copy') || $bubble.attr('data-raw') || $bubble.text();
    const citationRegistry = createCitationRegistry();

    // The rendered chips are the source of truth for citations visible in this message.
    $messageBody.find('.msg-citation-chip[data-citation-id][href]').each(function registerRenderedCitation() {
      addCitationSource(citationRegistry, {
        id: this.getAttribute('data-citation-id'),
        url: this.getAttribute('href')
      });
    });

    const text = replaceCitationHandlesWithSourceUrls(rawText, citationRegistry);

    // Swap the icon briefly to confirm the copy action.
    function onCopied() {
      const originalHtml = $btn.html();
      $btn.html(icons.COPIED_ICON);
      setTimeout(function restoreIcon() {
        $btn.html(originalHtml);
      }, 1200);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(onCopied).catch(function fallback() {
        fallbackCopy(text, onCopied);
      });
      return;
    }

    fallbackCopy(text, onCopied);
  }

  // Clipboard and reasoning drawer.
  function copyCodeBlock($button) {
    const $btn = $button || $();
    const $card = $btn.closest('.md-code-card, .md-mermaid-card');
    const $code = $card.find('pre code').first();
    const text = $code.text();
    if (!$code.length) {
      return;
    }

    function onCopied() {
      const originalHtml = $btn.html();
      $btn.addClass('is-copied').html(icons.COPIED_ICON);
      setTimeout(function restoreIcon() {
        $btn.removeClass('is-copied').html(originalHtml);
      }, 1200);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(onCopied).catch(function fallback() {
        fallbackCopy(text, onCopied);
      });
      return;
    }

    fallbackCopy(text, onCopied);
  }
  // Keep host theme colors in an otherwise origin-isolated preview document.
  function codePreviewTheme() {
    const styles = window.getComputedStyle(document.documentElement);
    const colorScheme = String(styles.colorScheme || '').includes('light') ? 'light' : 'dark';
    const probe = document.createElement('span');
    probe.hidden = true;
    const probeParent = document.body || document.documentElement;
    if (probeParent) {
      probeParent.appendChild(probe);
    }

    function color(variableName, fallback) {
      probe.style.color = `var(${variableName})`;
      const value = String(window.getComputedStyle(probe).color || '').trim();
      return value && (!window.CSS || !CSS.supports || CSS.supports('color', value)) ? value : fallback;
    }

    const theme = {
      scheme: colorScheme,
      background: color('--surface-primary', colorScheme === 'light' ? '#ffffff' : '#161616'),
      surface: color('--surface-secondary', colorScheme === 'light' ? '#f5f5f5' : '#222222'),
      text: color('--c-text', colorScheme === 'light' ? '#171717' : '#f1f1f1'),
      muted: color('--c-text-muted', colorScheme === 'light' ? '#666666' : '#a7a7a7'),
      border: color('--c-border', colorScheme === 'light' ? '#dddddd' : '#3a3a3a'),
      accent: color('--c-primary', '#6f7cff')
    };

    probe.remove();
    return theme;
  }

  // Escape source that is embedded inside one generated script element.
  function escapePreviewScriptSource(source) {
    return String(source || '').replace(/<\/script/gi, '<\\/script');
  }

  // Keep a CSS fence from escaping its generated style element.
  function escapePreviewStyleSource(source) {
    return String(source || '').replace(/<\/style/gi, '<\\/style');
  }

  // Build a strict CSP and adaptive base style for an isolated web preview.
  function codePreviewDocumentHead() {
    const theme = codePreviewTheme();
    return `
      <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; media-src data: blob:; font-src data:; connect-src 'none'; object-src 'none'; frame-src 'none'; worker-src 'none'; base-uri 'none'; form-action 'none'; navigate-to 'none'">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        :root { color-scheme: ${theme.scheme}; --preview-bg: ${theme.background}; --preview-surface: ${theme.surface}; --preview-text: ${theme.text}; --preview-muted: ${theme.muted}; --preview-border: ${theme.border}; --preview-accent: ${theme.accent}; }
        html { min-height: 100%; background: var(--preview-bg); color: var(--preview-text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        body { min-height: 100%; margin: 0; padding: 24px; box-sizing: border-box; }
        *, *::before, *::after { box-sizing: border-box; }
      </style>
    `;
  }

  // Insert the sandbox policy before user HTML while preserving full documents.
  function buildHtmlCodePreview(source) {
    const head = codePreviewDocumentHead();
    const html = String(source || '');
    if (/<head(?:\s[^>]*)?>/i.test(html)) {
      return html.replace(/<head(?:\s[^>]*)?>/i, function injectHead(openingHead) {
        return `${openingHead}${head}`;
      });
    }
    if (/<html(?:\s[^>]*)?>/i.test(html)) {
      return html.replace(/<html(?:\s[^>]*)?>/i, function injectHtml(openingHtml) {
        return `${openingHtml}<head>${head}</head>`;
      });
    }
    return `<!doctype html><html><head>${head}</head><body>${html}</body></html>`;
  }

  // Build the complete srcdoc for one supported fenced web language.
  function buildCodePreviewDocument(kind, source) {
    const safeSource = String(source || '');
    if (kind === 'html') {
      return buildHtmlCodePreview(safeSource);
    }
    if (kind === 'svg') {
      return `<!doctype html><html><head>${codePreviewDocumentHead()}<style>body{display:grid;place-items:center;overflow:auto}svg{max-width:100%;height:auto}</style></head><body>${safeSource}</body></html>`;
    }
    if (kind === 'css') {
      return `<!doctype html><html><head>${codePreviewDocumentHead()}<style>${escapePreviewStyleSource(safeSource)}</style></head><body><main class="preview"><h1>CSS Preview</h1><p>This sample content uses the stylesheet from the code block.</p><button type="button">Button</button></main></body></html>`;
    }
    if (kind === 'javascript') {
      return `<!doctype html><html><head>${codePreviewDocumentHead()}<style>#__aslm_console{margin:16px 0 0;padding:14px;border:1px solid var(--preview-border);border-radius:10px;background:var(--preview-surface);color:var(--preview-text);font:12px/1.55 ui-monospace,monospace;white-space:pre-wrap}</style></head><body><main id="app"></main><pre id="__aslm_console" hidden></pre><script>
        (() => {
          const output = document.getElementById('__aslm_console');
          const write = (level, values) => {
            output.hidden = false;
            const line = values.map(value => {
              if (typeof value === 'string') return value;
              try { return JSON.stringify(value, null, 2); } catch (_error) { return String(value); }
            }).join(' ');
            output.textContent += (output.textContent ? '\\n' : '') + (level === 'log' ? '' : '[' + level + '] ') + line;
          };
          ['log', 'info', 'warn', 'error'].forEach(level => { console[level] = (...values) => write(level, values); });
          window.addEventListener('error', event => write('error', [event.message]));
          window.addEventListener('unhandledrejection', event => write('error', [event.reason]));
        })();
        ${escapePreviewScriptSource(safeSource)}
      </script></body></html>`;
    }
    return '';
  }

  // Switch a previewable code card between highlighted source and live output.
  function setCodeBlockView($button) {
    const $btn = $button || $();
    const $card = $btn.closest('.md-code-card--previewable');
    const action = String($btn.attr('data-code-action') || '');
    const card = $card[0];
    if (!card || (action !== 'source' && action !== 'preview')) {
      return;
    }

    const showPreview = action === 'preview';
    const preview = card.querySelector('.md-code-preview');
    const frame = preview && preview.querySelector('.md-code-preview-frame');
    const code = card.querySelector('pre code');
    const pre = card.querySelector('pre');
    if (!preview || !frame || !code || !pre) {
      return;
    }

    card.dataset.codeView = showPreview ? 'preview' : 'source';
    pre.hidden = showPreview;
    preview.hidden = !showPreview;
    card.querySelectorAll('.md-code-view-btn').forEach(function syncCodeViewButton(button) {
      const active = button.dataset.codeAction === action;
      button.classList.toggle('is-active', active);
      button.setAttribute('aria-pressed', active ? 'true' : 'false');
    });


    if (showPreview) {
      frame.srcdoc = buildCodePreviewDocument(card.dataset.codePreviewKind, code.textContent || '');
    } else {
      frame.removeAttribute('srcdoc');
    }
  }




  // Reasoning drawer state.
  let $activeReasoningWrapper = null;
  let $activeReasoningMessageRow = null;
  let $activeReasoningIndex = '';
  let $activeSourcesMessageRow = null;
  const REASONING_DRAWER_DEFAULT_WIDTH = 340;
  const REASONING_DRAWER_MIN_WIDTH = 300;
  const REASONING_DRAWER_MAX_WIDTH = 720;
  const REASONING_DRAWER_CLOSE_WIDTH = 220;

  // Read the maximum width allowed for the reasoning drawer.
  function reasoningDrawerMaxWidth() {
    return Math.max(
      REASONING_DRAWER_MIN_WIDTH,
      Math.min(REASONING_DRAWER_MAX_WIDTH, Math.floor(window.innerWidth * 0.82))
    );
  }

  // Apply one pixel width to the reasoning drawer.
  function setReasoningDrawerWidth(width) {
    const clamped = Math.max(
      REASONING_DRAWER_MIN_WIDTH,
      Math.min(reasoningDrawerMaxWidth(), Math.round(Number(width) || REASONING_DRAWER_DEFAULT_WIDTH))
    );
    $('#reasoningDrawer').css('--reasoning-drawer-width', `${clamped}px`);
  }

  // Reset the reasoning drawer to its default width.
  function resetReasoningDrawerWidth() {
    setReasoningDrawerWidth(REASONING_DRAWER_DEFAULT_WIDTH);
  }

  // Sync drawer content from one reasoning wrapper.
  function syncReasoningDrawerFromWrapper($wrapper) {
    const $drawer = $('#reasoningDrawer');
    const $body = $('#reasoningDrawerBody');
    const $summary = $('#reasoningDrawerSummary');

    if (!$drawer.hasClass('is-open') || !$body.length) {
      return;
    }

    const $content = $wrapper.find('.msg-thoughts-content, .msg-reasoning-content').first();
    const summaryText = $wrapper.find('.msg-reasoning-summary').text().trim();
    const bodyEl = $body[0];
    const isNearBottom = bodyEl ? bodyEl.scrollHeight - bodyEl.clientHeight <= bodyEl.scrollTop + 50 : false;
    const nextBodyHtml = `<div class="msg-reasoning-content">${$content.html() || ''}</div>`;
    $summary.text(summaryText);

    if (bodyEl) {
      const template = document.createElement('template');
      template.innerHTML = nextBodyHtml;
      morphDomChildren(bodyEl, template.content);
    } else {
      $body.html(nextBodyHtml);
    }

    $body
      .data('messageRow', $wrapper.closest('.msg'))
      .data('toolSegments', $wrapper.closest('.msg').data('toolSegments') || []);
    hydrateMermaidDiagrams($body);
    hydrateImageTokens($body);
    if (isNearBottom && bodyEl) {
      bodyEl.scrollTop = bodyEl.scrollHeight;
    }
  }

  // Open the reasoning drawer for one message wrapper.
  function openReasoningDrawer($wrapper) {
    const $drawer = $('#reasoningDrawer');
    const $backdrop = $('#reasoningDrawerBackdrop');

    if (!$drawer.length) {
      return;
    }

    closeSourcesDrawer();
    resetReasoningDrawerWidth();

    // Deactivate previously active pill.
    if ($activeReasoningWrapper && $activeReasoningWrapper[0] !== $wrapper[0]) {
      $activeReasoningWrapper.removeClass('is-active');
      $activeReasoningWrapper.find('.msg-reasoning-toggle, .msg-thoughts-toggle').attr('aria-expanded', 'false');
    }

    // If clicking the same pill while open, close the drawer.
    if ($activeReasoningWrapper && $activeReasoningWrapper[0] === $wrapper[0] && $drawer.hasClass('is-open')) {
      closeReasoningDrawer();
      return;
    }

    $activeReasoningWrapper = $wrapper;
    $activeReasoningMessageRow = $wrapper.closest('.msg');
    $activeReasoningIndex = String($wrapper.attr('data-thought-index') || '');
    $wrapper.addClass('is-active');
    $wrapper.find('.msg-reasoning-toggle, .msg-thoughts-toggle').attr('aria-expanded', 'true');

    $drawer.addClass('is-open');
    $backdrop.addClass('is-visible');
    syncReasoningDrawerFromWrapper($wrapper);
  }

  // Close the reasoning drawer and clear its content.
  function closeReasoningDrawer() {
    const $drawer = $('#reasoningDrawer');
    const $backdrop = $('#reasoningDrawerBackdrop');

    $drawer.removeClass('is-open is-resizing');
    $backdrop.removeClass('is-visible');
    $('body').removeClass('is-resizing-reasoning-drawer');

    if ($activeReasoningWrapper) {
      $activeReasoningWrapper.removeClass('is-active');
      $activeReasoningWrapper.find('.msg-reasoning-toggle, .msg-thoughts-toggle').attr('aria-expanded', 'false');
      $activeReasoningWrapper = null;
    }
    $activeReasoningMessageRow = null;
    $activeReasoningIndex = '';

    // Clear body after transition.
    setTimeout(function clearDrawerBody() {
      if (!$drawer.hasClass('is-open')) {
        $('#reasoningDrawerBody').empty().removeData('toolSegments').removeData('messageRow');
      }
    }, 250);
  }

  // Return only safe HTTP(S) links for clickable source cards.
  function safeMessageSourceUrl(value) {
    try {
      const parsed = new URL(String(value || '').trim(), window.location.href);
      return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.href : '';
    } catch (_error) {
      return '';
    }
  }

  // Render one source as a compact domain/title/preview card.
  function renderSourcesDrawerItem(source) {
    const safeSource = source && typeof source === 'object' ? source : {};
    const domain = String(safeSource.display_domain || safeSource.domain || safeSource.url || '').trim();
    if (!domain) {
      return '';
    }

    const url = safeMessageSourceUrl(safeSource.url || safeSource.link || safeSource.href || '');
    const title = String(safeSource.title || domain).replace(/\s+/g, ' ').trim();
    const preview = String(
      safeSource.preview
      || safeSource.snippet
      || safeSource.summary
      || safeSource.content
      || safeSource.text
      || ''
    ).replace(/\s+/g, ' ').trim();
    const date = String(safeSource.date || safeSource.published_date || safeSource.published_at || '').trim();
    const faviconUrl = sourceFaviconUrl(safeSource);
    const fallbackLetter = (domain.charAt(0) || '?').toUpperCase();
    const imgHtml = faviconUrl
      ? `<img class="sources-drawer-favicon" src="${escapeAttributeValue(faviconUrl)}" alt="" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='inline-flex';">`
      : '';
    const fallbackStyle = domainAccentStyle(domain, faviconUrl ? 'display:none;' : '');
    const tagName = url ? 'a' : 'div';
    const linkAttrs = url ? ` href="${escapeAttributeValue(url)}" target="_blank" rel="noopener noreferrer"` : '';
    const previewText = [date, preview].filter(Boolean).join(date && preview ? ' — ' : '');

    return `
      <${tagName} class="sources-drawer-item"${linkAttrs}>
        <div class="sources-drawer-domain-row">
          ${imgHtml}<span class="sources-drawer-fallback" style="${escapeAttributeValue(fallbackStyle)}">${escHtml(fallbackLetter)}</span>
          <span class="sources-drawer-domain">${escHtml(domain)}</span>
        </div>
        <div class="sources-drawer-item-title">${escHtml(title)}</div>
        ${previewText ? `<div class="sources-drawer-item-preview">${escHtml(previewText)}</div>` : ''}
      </${tagName}>
    `;
  }

  // Keep an open Sources drawer synchronized with its response row.
  function syncSourcesDrawer($msgRow) {
    const $drawer = $('#sourcesDrawer');
    const $body = $('#sourcesDrawerBody');
    if (!$drawer.hasClass('is-open') || !$body.length || !$msgRow || !$msgRow.length) {
      return;
    }
    const sources = $msgRow.data('messageSources') || [];
    const safeSources = Array.isArray(sources) ? sources : [];
    $('#sourcesDrawerCount').text(`· ${safeSources.length}`);
    $body.html(safeSources.map(renderSourcesDrawerItem).join(''));
  }

  // Open or toggle the Sources drawer for one assistant response.
  function openSourcesDrawer($button) {
    const $msgRow = $button && $button.length ? $button.closest('.msg.assistant') : $();
    const sources = $msgRow.data('messageSources') || [];
    const $drawer = $('#sourcesDrawer');
    if (!$msgRow.length || !Array.isArray(sources) || !sources.length || !$drawer.length) {
      return;
    }

    if ($activeSourcesMessageRow && $activeSourcesMessageRow[0] === $msgRow[0] && $drawer.hasClass('is-open')) {
      closeSourcesDrawer();
      return;
    }

    closeReasoningDrawer();
    if ($activeSourcesMessageRow) {
      $activeSourcesMessageRow.find('.msg-sources-btn').removeClass('is-active').attr('aria-expanded', 'false');
    }
    $activeSourcesMessageRow = $msgRow;
    $msgRow.find('.msg-sources-btn').addClass('is-active').attr('aria-expanded', 'true');
    $drawer.addClass('is-open');
    $('#sourcesDrawerBackdrop').addClass('is-visible');
    syncSourcesDrawer($msgRow);
  }

  // Close the Sources drawer and clear its active action button.
  function closeSourcesDrawer() {
    const $drawer = $('#sourcesDrawer');
    $drawer.removeClass('is-open');
    $('#sourcesDrawerBackdrop').removeClass('is-visible');
    if ($activeSourcesMessageRow) {
      $activeSourcesMessageRow.find('.msg-sources-btn').removeClass('is-active').attr('aria-expanded', 'false');
    }
    $activeSourcesMessageRow = null;
    setTimeout(function clearSourcesDrawerBody() {
      if (!$drawer.hasClass('is-open')) {
        $('#sourcesDrawerBody').empty();
        $('#sourcesDrawerCount').empty();
      }
    }, 250);
  }

  // Bind drag-to-resize behavior on the reasoning drawer.
  function bindReasoningDrawerResize() {
    const $handle = $('#reasoningDrawerResizeHandle');
    if (!$handle.length) {
      return;
    }

    $handle.on('pointerdown', function onReasoningResizeStart(event) {
      const $drawer = $('#reasoningDrawer');
      if (!$drawer.hasClass('is-open')) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      let shouldClose = false;
      $drawer.addClass('is-resizing');
      $('body').addClass('is-resizing-reasoning-drawer');

      function widthFromPointer(pointerEvent) {
        return window.innerWidth - pointerEvent.clientX;
      }

      function onMove(moveEvent) {
        const nextWidth = widthFromPointer(moveEvent);
        shouldClose = nextWidth <= REASONING_DRAWER_CLOSE_WIDTH;
        if (!shouldClose) {
          setReasoningDrawerWidth(nextWidth);
        }
      }

      function onEnd() {
        $(document).off('.reasoningDrawerResize');
        $drawer.removeClass('is-resizing');
        $('body').removeClass('is-resizing-reasoning-drawer');
        if (shouldClose) {
          closeReasoningDrawer();
        }
      }

      $(document)
        .on('pointermove.reasoningDrawerResize', onMove)
        .on('pointerup.reasoningDrawerResize pointercancel.reasoningDrawerResize', onEnd);
    });
  }

  // Thought UI.
  function toggleThoughtSection($toggle) {
    const $wrapper = $toggle.closest('.msg-thoughts-wrapper, .msg-reasoning-wrapper');
    openReasoningDrawer($wrapper);
  }


  // Markdown configuration.
  // Code highlighting is applied after sanitization when code cards are built.
  function configureMarkdown() {
    if (typeof marked === 'undefined') {
      configureMermaid();
      return;
    }

    marked.setOptions({
      breaks: true
    });

    if (!markedStrikethroughDoubleTildeOnlyInstalled) {
      markedStrikethroughDoubleTildeOnlyInstalled = true;
      const delDoubleTilde = /^(~~)(?=[^\s~])((?:\\.|[^\\])*?(?:\\.|[^\s~\\]))\1(?=[^~]|$)/;
      marked.use({
        tokenizer: {
          del(src) {
            const cap = delDoubleTilde.exec(src);
            if (cap) {
              return {
                type: 'del',
                raw: cap[0],
                text: cap[2],
                tokens: this.lexer.inlineTokens(cap[2])
              };
            }
          }
        }
      });
    }

    configureMermaid();
  }

  // Let Deep Research reuse the exact web-search and Markdown/citation renderers
  // instead of maintaining visually divergent copies in its inspector.
  if (deepResearchUi && typeof deepResearchUi.setCanonicalRenderers === 'function') {
    deepResearchUi.setCanonicalRenderers({
      search(segment, index, options) {
        return renderSearchToolCard(segment, index, options);
      },
      report(report, sources) {
        const citationRegistry = createCitationRegistry();
        (Array.isArray(sources) ? sources : []).forEach(function registerResearchSource(source, index) {
          addCitationSource(citationRegistry, source, index + 1);
          const aliases = source && (
            source.citation_aliases || source.citationAliases || source.aliases
          );
          (Array.isArray(aliases) ? aliases : []).forEach(function registerResearchAlias(alias) {
            addCitationSource(citationRegistry, { ...source, id: alias }, index + 1);
          });
        });
        return renderMarkdownSegment(report, citationRegistry);
      },
      hydrate(root) {
        const $root = root ? $(root) : dom.$messagesInner;
        hydrateMermaidDiagrams($root);
        hydrateImageTokens($root);
      }
    });
  }

  bindReasoningDrawerResize();
  bindAttachmentMediaEvents();
  bindCitationPreviewCards(document);

  return {
    appendMessage,
    appendMessages,
    appendCompressionPending,
    appendTyping,
    configureMarkdown,
    copyCodeBlock,
    setCodeBlockView,
    copyMessage,
    openToolInspectorFromCard,
    renderMessageHtml,
    renderMessageStream,
    removeCompressionPending,
    revealSharedFile,
    scrollBottom,
    setQueuedMessageState,
    toggleSearchSources,
    toggleEditCard,
    toggleSandboxCard,
    toggleCompressionContext,
    toggleWriteCard,
    startWritePreviewPan,
    toggleThoughtSection,
    closeReasoningDrawer,
    openSourcesDrawer,
    closeSourcesDrawer,
    syncMessageSourcesButton,
    updateRegenButtons,
    updateSendButtons
  };
}
