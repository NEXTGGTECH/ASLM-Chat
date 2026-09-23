// Copyright NEXTGGTECH. Elastic License 2.0.

import { getCsrfToken } from '../main/api.js';
import { escHtml } from '../main/utils.js';
import { t } from '../main/i18n.js';
import { largeTextDialog } from './dialogs.js';

// Attachment UI.
// Create helpers for file picking, previews, and attachment state.
export function createAttachmentsUi(context) {
  const { dom, icons, state } = context;
  let updateSendButtons = function noop() {};

  // Integration hooks.
  // Register the send-button refresh callback owned by the message UI.
  function setUpdateSendButtons(fn) {
    updateSendButtons = typeof fn === 'function' ? fn : function noop() {};
  }

  // Return whether a URL contains inline base64 data.
  function isInlineDataUrl(value) {
    return String(value || '').startsWith('data:');
  }

  // Extract a base64 payload only from inline data URLs.
  function dataUrlToBase64(value) {
    const dataUrl = String(value || '');
    if (!isInlineDataUrl(dataUrl)) {
      return '';
    }
    return dataUrl.replace(/^data:[^;]+;base64,/, '');
  }

  // Convert a fetched Blob into a data URL.
  function readBlobAsDataUrl(blob) {
    return new Promise(function resolveBlob(resolve, reject) {
      const reader = new FileReader();
      reader.onload = function onLoad() {
        resolve(String(reader.result || ''));
      };
      reader.onerror = function onError() {
        reject(reader.error || new Error('Failed to read attachment'));
      };
      reader.readAsDataURL(blob);
    });
  }


  // Attachment controls.
  // Toggle attachment buttons and badges from model capabilities.
  function updateAttachmentControls() {
    dom.$attachBtn.show();
    dom.$attachBtnConv.show();
    $(document).trigger('aslm:modelCapabilitiesChanged');
  }

  // Clear all pending attachments from both composers.
  function clearPendingAttachments() {
    state.attachmentState.pending = [];
    dom.$imagePreviewStrip.empty().hide();
    dom.$imagePreviewStripConv.empty().hide();
    dom.$imageInput.val('');
    dom.$imageInputConv.val('');
    updateSendButtons();
  }


  // Attachment normalization.
  // Normalize one stored or runtime attachment into the shared UI shape.
  function normalizeAttachment(attachment) {
    if (!attachment) {
      return null;
    }

    if (typeof attachment === 'string') {
      const source = String(attachment || '');
      const isRemoteUrl = source.startsWith('/') || /^https?:\/\//i.test(source);
      const base64 = isRemoteUrl ? '' : dataUrlToBase64(source) || source;

      return {
        kind: 'image',
        name: '',
        mimeType: 'image/jpeg',
        size: 0,
        base64,
        dataUrl: source,
        contentUrl: isRemoteUrl ? source : ''
      };
    }

    const fileId = String(attachment.fileId || attachment.file_id || '').trim();
    let contentUrl = String(attachment.contentUrl || attachment.content_url || '').trim();

    // If the server didn't return a content_url but we have a file_id, derive it
    // from the known upload serving endpoint so media players get a valid src.
    if (!contentUrl && fileId) {
      contentUrl = `/api/uploads/${encodeURIComponent(fileId)}/content/`;
    }

    const dataUrl = attachment.dataUrl || attachment.data_url || contentUrl || '';
    const previewDataUrl = attachment.previewDataUrl || attachment.preview_data_url || '';
    const mimeType = attachment.mimeType || attachment.mime_type || 'application/octet-stream';
    const base64 = attachment.base64 || attachment.data || dataUrlToBase64(dataUrl);

    return {
      id: attachment.id || null,
      kind: attachment.kind || 'file',
      fileId,
      name: attachment.name || '',
      mimeType,
      size: attachment.size || attachment.size_bytes || 0,
      base64,
      dataUrl: dataUrl || (base64 ? `data:${mimeType};base64,${base64}` : ''),
      previewDataUrl,
      contentUrl,
      recordType: attachment.recordType || attachment.record_type || '',
      status: attachment.status || 'ready',
      displayKind: attachment.displayKind || attachment.display_kind || '',
      typeLabel: attachment.typeLabel || attachment.type_label || '',
      // Preserve pasted-artifact specific fields (full text for preview/edit before send).
      pasted: !!attachment.pasted,
      textContent: attachment.textContent || attachment.text_content || '',
      // URL attachment (link pasted or dropped; content fetched server-side via read_page).
      url: attachment.url || attachment.href || '',
    };
  }

  // Ensure a stored attachment has inline data before it is sent again.
  async function resolveAttachmentData(attachment) {
    const normalized = normalizeAttachment(attachment);
    if (!normalized) {
      return null;
    }

    if (normalized.base64) {
      return normalized;
    }

    const fetchUrl = normalized.contentUrl || (!isInlineDataUrl(normalized.dataUrl) ? normalized.dataUrl : '');
    if (!fetchUrl || typeof fetch === 'undefined') {
      return normalized;
    }

    const response = await fetch(fetchUrl);
    if (!response.ok) {
      throw new Error(`Failed to load attachment: ${response.status}`);
    }

    const blob = await response.blob();
    const dataUrl = await readBlobAsDataUrl(blob);
    const mimeType = blob.type || normalized.mimeType;

    return {
      ...normalized,
      mimeType,
      size: normalized.size || blob.size || 0,
      base64: dataUrlToBase64(dataUrl),
      dataUrl,
      // Carry forward pasted artifact info (textContent is the source of truth for edits).
      pasted: normalized.pasted,
      textContent: normalized.textContent,
      url: normalized.url
    };
  }


  // Preview rendering.
  // Build the subtitle shown under one pending attachment chip.
  function previewLabel(attachment) {
    if (attachment.pasted || attachment.displayKind === 'pasted') {
      return attachment.typeLabel || t('dialog.pastedTypeLabel', null, 'Pasted');
    }
    return attachment.typeLabel || attachment.mimeType || 'File';
  }

  // Build the badge text shown on non-image attachment chips.
  function uploadIconLabel(attachment) {
    const kind = String(attachment.displayKind || attachment.kind || '').toLowerCase();
    if (kind === 'image') {
      return 'IMG';
    }
    if (kind === 'audio') {
      return 'AUDIO';
    }
    if (kind === 'video') {
      return 'VIDEO';
    }
    if (kind === 'archive') {
      return 'ZIP';
    }
    if (kind === 'code') {
      return '</>';
    }
    if (kind === 'table') {
      return 'CSV';
    }
    if (kind === 'document') {
      return 'DOC';
    }
    if (kind === 'pasted') {
      return 'FILE';
    }
    return 'FILE';
  }

  // Report whether one File object should be treated as an image.
  function isImageFile(file) {
    const name = String(file && file.name ? file.name : '').toLowerCase();
    const mimeType = String(file && file.type ? file.type : '').toLowerCase();
    return mimeType.startsWith('image/')
      || /\.(png|jpe?g|webp|gif|bmp|avif)$/i.test(name);
  }

  // Infer display kind and label for one selected file.
  function displayKindForFile(file) {
    const name = String(file && file.name ? file.name : '').toLowerCase();
    const mimeType = String(file && file.type ? file.type : '').toLowerCase();
    if (isImageFile(file)) {
      return ['image', 'Image'];
    }
    if (mimeType.startsWith('audio/') || /\.(mp3|wav|ogg|oga|m4a|aac|flac|opus)$/i.test(name)) {
      return ['audio', 'Audio'];
    }
    if (mimeType.startsWith('video/') || /\.(mp4|webm|mov|m4v|ogv|avi|mkv)$/i.test(name)) {
      return ['video', 'Video'];
    }
    if (name.endsWith('.zip') || mimeType === 'application/zip' || mimeType === 'application/x-zip-compressed') {
      return ['archive', 'ZIP archive'];
    }
    if (name.endsWith('.rar') || name.endsWith('.7z')) {
      return ['archive', 'Archive'];
    }
    if (name.endsWith('.pdf') || mimeType === 'application/pdf') {
      return ['document', 'PDF document'];
    }
    if (name.endsWith('.docx')) {
      return ['document', 'Word document'];
    }
    if (name.endsWith('.xlsx')) {
      return ['table', 'Excel spreadsheet'];
    }
    if (name.endsWith('.pptx')) {
      return ['presentation', 'PowerPoint presentation'];
    }
    if (/\.(py|js|ts|css|html|sql|sh|ps1)$/i.test(name)) {
      return ['code', 'Code file'];
    }
    if (/\.(txt|md|log|json|yaml|yml|xml|csv)$/i.test(name) || mimeType.startsWith('text/')) {
      return name.endsWith('.csv') ? ['table', 'CSV table'] : ['text', 'Text file'];
    }
    return ['file', 'File'];
  }

  // Pasted text artifact configuration and helpers.
  // Thresholds for treating clipboard text as a compact PASTED artifact instead of raw input text.
  const PASTED_TEXT_MIN_CHARS = 512;
  const PASTED_TEXT_MIN_LINES = 20;

  // Decide whether pasted plain text should become a file-like editable artifact.
  function shouldTreatPasteAsArtifact(rawText) {
    const text = String(rawText || '');
    if (!text.trim()) {
      return false;
    }
    if (text.length >= PASTED_TEXT_MIN_CHARS) {
      return true;
    }
    const lineCount = text.split('\n').length;
    return lineCount >= PASTED_TEXT_MIN_LINES;
  }

  // Small single-line preview for the file chip (no scroller, just a hint of the content).
  function getPastedPreviewText(text, maxLen = 55) {
    let t = String(text || '').trim().replace(/\s+/g, ' ');
    if (!t) return '';
    if (t.length <= maxLen) return t;
    return t.substring(0, maxLen - 3) + '...';
  }

  // URL detection and helpers for link attachments (keep original text in input).
  function isValidHttpUrl(str) {
    try {
      const u = new URL(String(str || '').trim());
      return u.protocol === 'http:' || u.protocol === 'https:';
    } catch (_e) {
      return false;
    }
  }

  function truncateUrlForWidget(url, maxLen = 50) {
    const t = String(url || '').trim();
    if (t.length <= maxLen) return t;
    // simple head...tail for urls
    const head = t.slice(0, Math.floor(maxLen * 0.6));
    const tail = t.slice(-Math.floor(maxLen * 0.3));
    return `${head}...${tail}`;
  }

  // Rebuild both preview strips from the current pending attachments.
  function rebuildPreviewStrips() {
    const $strips = dom.$imagePreviewStrip.add(dom.$imagePreviewStripConv);
    $strips.empty();

    if (state.attachmentState.pending.length === 0) {
      $strips.hide();
      updateSendButtons();
      return;
    }

    state.attachmentState.pending.forEach(function renderAttachment(attachment, index) {
      let html = '';

      const imagePreviewSrc = attachment.dataUrl || attachment.previewDataUrl || '';
      if ((attachment.kind === 'image' || attachment.displayKind === 'image') && imagePreviewSrc) {
        html = `
          <div class="img-preview-thumb" data-idx="${index}">
            <img src="${imagePreviewSrc}" alt="Attached image">
            <button class="img-preview-remove" aria-label="Remove attachment">
              ${icons.REMOVE_ATTACHMENT_ICON}
            </button>
          </div>
        `;
      } else if (attachment.pasted || attachment.displayKind === 'pasted') {
        // Render pasted text as a regular file attachment chip (matching the design of other file uploads).
        // Whole chip (except X) is clickable to open the editor.
        // Shows "Pasted text" + a small content preview (no scroller).
        const isUploading = attachment.status === 'uploading';
        const isError = attachment.status === 'error';
        const iconLabel = 'FILE';
        const displayName = 'Pasted text';
        let meta = isUploading ? t('dialog.pastedUploading', null, 'Uploading...') :
                   (isError ? t('dialog.pastedUploadFailed', null, 'Upload failed') : '');
        if (!meta) {
          const shortPrev = getPastedPreviewText(attachment.textContent);
          meta = shortPrev || t('dialog.pastedTypeLabel', null, 'Pasted');
        }
        html = `
          <div class="file-preview-chip is-pasted${isUploading ? ' is-uploading' : ''}${isError ? ' is-error' : ''}" data-idx="${index}">
            <div class="file-preview-icon" aria-hidden="true">${escHtml(iconLabel)}</div>
            <div class="file-preview-name">${escHtml(displayName)}</div>
            <div class="file-preview-meta">${escHtml(meta)}</div>
            <button class="img-preview-remove" aria-label="${escHtml(t('dialog.pastedRemove', null, 'Remove pasted text'))}">
              ${icons.REMOVE_ATTACHMENT_ICON}
            </button>
          </div>
        `;
      } else if (attachment.displayKind === 'url' || attachment.kind === 'url') {
        // URL attachment: looks like regular file chip, badge "URL", description = the address (truncated to fit).
        // Text of the URL is kept in the input (we never preventDefault for links).
        const isUploading = attachment.status === 'uploading';
        const isError = attachment.status === 'error';
        const url = attachment.url || attachment.name || '';
        const displayUrl = truncateUrlForWidget(url);
        const meta = isUploading ? t('dialog.pastedUploading', null, 'Uploading...') :
                     (isError ? t('dialog.pastedUploadFailed', null, 'Upload failed') : 'URL');
        html = `
          <div class="file-preview-chip is-url${isUploading ? ' is-uploading' : ''}${isError ? ' is-error' : ''}" data-idx="${index}">
            <div class="file-preview-icon" aria-hidden="true">URL</div>
            <div class="file-preview-name">${escHtml(displayUrl)}</div>
            <div class="file-preview-meta">${escHtml(meta)}</div>
            <button class="img-preview-remove" aria-label="Remove URL attachment">
              ${icons.REMOVE_ATTACHMENT_ICON}
            </button>
          </div>
        `;
      } else {
        const isUploading = attachment.status === 'uploading';
        const isError = attachment.status === 'error';
        html = `
          <div class="file-preview-chip${isUploading ? ' is-uploading' : ''}${isError ? ' is-error' : ''}" data-idx="${index}">
            <div class="file-preview-icon" aria-hidden="true">${escHtml(uploadIconLabel(attachment))}</div>
            <div class="file-preview-name">${escHtml(attachment.name || 'File')}</div>
            <div class="file-preview-meta">${escHtml(isUploading ? t('dialog.pastedUploading', null, 'Uploading...') : (isError ? t('dialog.pastedUploadFailed', null, 'Upload failed') : previewLabel(attachment)))}</div>
            <button class="img-preview-remove" aria-label="Remove attachment">
              ${icons.REMOVE_ATTACHMENT_ICON}
            </button>
          </div>
        `;
      }

      $strips.append(html);
    });

    $strips.show();
    updateSendButtons();
  }

  // Remove one pending attachment by index.
  function removePendingAttachment(index) {
    if (!Number.isInteger(index) || index < 0) {
      return;
    }

    state.attachmentState.pending.splice(index, 1);
    rebuildPreviewStrips();
  }

  // Open an editor for a pasted text artifact so the user can view / modify the full content
  // before the message is sent. Updates textContent in place and re-uploads the revised version.
  async function editPastedAttachment(index) {
    if (!Number.isInteger(index) || index < 0) {
      return;
    }
    const pendingList = state.attachmentState.pending;
    const attachment = pendingList[index];
    if (!attachment || (!attachment.pasted && attachment.displayKind !== 'pasted')) {
      return;
    }

    const originalText = String(attachment.textContent || '');
    const edited = await largeTextDialog({
      title: t('dialog.editPastedTitle', null, 'Edit pasted content'),
      value: originalText,
      placeholder: t('dialog.editPastedPlaceholder', null, 'Edit the pasted text. Use Ctrl/Cmd+Enter to save.'),
      confirmText: t('dialog.save', null, 'Save changes'),
      required: false   // allow clearing if user really wants
    });

    if (edited === null) {
      // User cancelled
      return;
    }

    const newText = String(edited);

    // Always store the (possibly edited) full content for future renders/sends.
    attachment.textContent = newText;

    // If the user cleared everything, treat as removal of the artifact.
    if (!newText.trim()) {
      pendingList.splice(index, 1);
      rebuildPreviewStrips();
      updateSendButtons();
      return;
    }

    // Recreate a fresh File from the edited text and (re)upload so the server copy is current.
    const filename = 'pasted.txt';
    let newFile;
    try {
      newFile = new File([newText], filename, { type: 'text/plain', lastModified: Date.now() });
    } catch (_e) {
      newFile = new Blob([newText], { type: 'text/plain' });
      newFile.name = filename;
    }

    attachment.status = 'uploading';
    attachment.fileId = '';
    rebuildPreviewStrips();

    // Re-run the upload for this attachment (overwrites previous fileId / status).
    uploadOneFile(newFile, attachment)
      .then(function onReupload() {
        if (attachment.status !== 'error') {
          attachment.status = 'ready';
        }
        rebuildPreviewStrips();
      })
      .catch(function onReuploadError(err) {
        console.error('Re-upload of edited pasted text failed', err);
        attachment.status = 'error';
        attachment.typeLabel = t('dialog.pastedUploadFailed', null, 'Upload failed');
        rebuildPreviewStrips();
      });
  }


  // File input handling.
  // Read one File object as a data URL for local previews.
  function readFileAsDataUrl(file) {
    return new Promise(function resolveFile(resolve, reject) {
      const reader = new FileReader();
      reader.onload = function onLoad(loadEvent) {
        resolve(String(loadEvent.target.result || ''));
      };
      reader.onerror = function onError() {
        reject(reader.error || new Error('Failed to read file'));
      };
      reader.readAsDataURL(file);
    });
  }

  // Upload one file to the server and merge the response into pending state.
  async function uploadOneFile(file, pendingAttachment) {
    const formData = new FormData();
    formData.append('files', file, file.name || 'file');
    formData.append('scope', state.currentChatId || 'pending');
    formData.append('supports_vision', state.visionState.supported ? '1' : '0');
    Array.from(state.selectedToolServerIds || []).forEach(function appendToolServerId(serverId) {
      const normalized = String(serverId || '').trim();
      if (normalized) {
        formData.append('tool_server_ids', normalized);
      }
    });

    const response = await fetch('/api/uploads/', {
      method: 'POST',
      headers: {
        'X-CSRFToken': getCsrfToken()
      },
      body: formData
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.status}`);
    }

    const payload = await response.json();
    const uploadedFile = Array.isArray(payload.files) ? payload.files[0] : null;
    if (!uploadedFile || uploadedFile.status === 'error') {
      throw new Error((uploadedFile && uploadedFile.error) || 'Upload failed');
    }

    Object.assign(pendingAttachment, {
      fileId: uploadedFile.file_id || '',
      name: uploadedFile.name || pendingAttachment.name,
      mimeType: uploadedFile.mime_type || pendingAttachment.mimeType,
      size: uploadedFile.size_bytes || pendingAttachment.size,
      status: uploadedFile.status || 'ready',
      displayKind: uploadedFile.display_kind || pendingAttachment.displayKind,
      typeLabel: uploadedFile.type_label || pendingAttachment.typeLabel,
      contentUrl: uploadedFile.content_url || pendingAttachment.contentUrl || ''
    });

    // Re-assert pasted artifact identity after server response (server may return generic 'text').
    if (pendingAttachment.pasted) {
      pendingAttachment.displayKind = 'pasted';
      pendingAttachment.typeLabel = pendingAttachment.typeLabel || t('dialog.pastedTypeLabel', null, 'Pasted');
      pendingAttachment.name = 'Pasted text';
    }
  }

  // Upload selected files and queue them for the next request.
  async function queueFiles(files) {
    const maxAttachments = 20;
    const selectedFiles = Array.from(files || []);

    if (!selectedFiles.length) {
      return;
    }

    selectedFiles.forEach(function queueFile(file) {
      const isImage = isImageFile(file);
      const [displayKind, typeLabel] = displayKindForFile(file);

      if (state.attachmentState.pending.length >= maxAttachments) {
        console.warn(`Max ${maxAttachments} attachments allowed`);
        return;
      }

      const canPreviewAsMedia = displayKind === 'audio' || displayKind === 'video';
      const objectPreviewUrl = canPreviewAsMedia && typeof URL !== 'undefined' && URL.createObjectURL
        ? URL.createObjectURL(file)
        : '';
      const pendingAttachment = {
        kind: isImage && state.visionState.supported ? 'image' : 'file',
        fileId: '',
        name: file.name || '',
        mimeType: file.type || 'application/octet-stream',
        size: file.size || 0,
        base64: '',
        dataUrl: '',
        previewDataUrl: objectPreviewUrl,
        status: 'uploading',
        displayKind,
        typeLabel
      };

      state.attachmentState.pending.push(pendingAttachment);
      rebuildPreviewStrips();

      const imagePreviewPromise = isImage
        ? readFileAsDataUrl(file).then(function applyDataUrl(dataUrl) {
          if (state.visionState.supported) {
            pendingAttachment.dataUrl = dataUrl;
            pendingAttachment.base64 = String(dataUrl || '').split(',')[1] || '';
          } else {
            pendingAttachment.previewDataUrl = dataUrl;
          }
          rebuildPreviewStrips();
        }).catch(function ignorePreviewError() {})
        : Promise.resolve();

      Promise.all([uploadOneFile(file, pendingAttachment), imagePreviewPromise])
        .then(function onUploaded() {
          if (pendingAttachment.status !== 'error') {
            pendingAttachment.status = 'ready';
          }
          rebuildPreviewStrips();
        })
        .catch(function onUploadError(error) {
          console.error(error);
          pendingAttachment.status = 'error';
          pendingAttachment.typeLabel = t('dialog.pastedUploadFailed', null, 'Upload failed');
          rebuildPreviewStrips();
        });
    });
  }


  // Clipboard and drag-drop helpers.
  // Build a filename for one pasted image blob.
  function clipboardImageName(mimeType, index) {
    const normalizedMime = String(mimeType || '').toLowerCase();
    const extensionByMime = {
      'image/avif': 'avif',
      'image/bmp': 'bmp',
      'image/gif': 'gif',
      'image/jpeg': 'jpg',
      'image/png': 'png',
      'image/webp': 'webp'
    };
    const extension = extensionByMime[normalizedMime] || 'png';
    return `pasted-image-${Date.now()}-${index + 1}.${extension}`;
  }

  // Normalize one clipboard image File with a generated filename when needed.
  function normalizeClipboardImageFile(file, index) {
    if (!file || !isImageFile(file)) {
      return null;
    }

    if (file.name) {
      return file;
    }

    try {
      return new File([file], clipboardImageName(file.type, index), {
        type: file.type || 'image/png',
        lastModified: Date.now()
      });
    } catch (_error) {
      return file;
    }
  }

  // Collect unique image files from a clipboard DataTransfer.
  function collectClipboardImageFiles(clipboardData) {
    const files = [];
    const seen = new Set();

    function addFile(file) {
      const normalizedFile = normalizeClipboardImageFile(file, files.length);
      if (!normalizedFile) {
        return;
      }
      const key = [
        normalizedFile.name || '',
        normalizedFile.type || '',
        normalizedFile.size || 0,
        normalizedFile.lastModified || 0
      ].join('|');
      if (seen.has(key)) {
        return;
      }
      seen.add(key);
      files.push(normalizedFile);
    }

    Array.from((clipboardData && clipboardData.items) || []).forEach(function collectItem(item) {
      if (!item || String(item.kind || '').toLowerCase() !== 'file') {
        return;
      }
      if (!String(item.type || '').toLowerCase().startsWith('image/')) {
        return;
      }
      addFile(item.getAsFile && item.getAsFile());
    });

    if (!files.length) {
      Array.from((clipboardData && clipboardData.files) || []).forEach(addFile);
    }

    return files;
  }

  // Queue pasted clipboard images when the clipboard carries files.
  function handleClipboardPaste(clipboardData) {
    const files = collectClipboardImageFiles(clipboardData);
    if (files.length) {
      queueFiles(files);
      return true;
    }

    const text = clipboardData && typeof clipboardData.getData === 'function'
      ? clipboardData.getData('text/plain')
      : '';

    // Only a paste consisting entirely of one URL becomes a URL attachment.
    // Return false so the browser also keeps that URL in the input.
    const pastedUrl = String(text || '').trim();
    if (pastedUrl && !/\s/.test(pastedUrl) && isValidHttpUrl(pastedUrl)) {
      queueUrlAttachment(pastedUrl);
      return false;
    }

    // Large plain text paste becomes one editable artifact without URL attachments.
    if (text && shouldTreatPasteAsArtifact(text)) {
      queuePastedTextArtifact(text);
      return true;
    }

    return false;
  }

  // Queue a URL as a special attachment (no file upload). The URL text stays in the composer input.
  // Displayed like a file chip with "URL" badge + truncated address. Content will be fetched
  // server-side via WebSearch read_page after send and injected into LLM context.
  function queueUrlAttachment(url) {
    const clean = String(url || '').trim();
    if (!clean || !isValidHttpUrl(clean)) {
      return;
    }
    const pending = {
      kind: 'url',
      url: clean,
      name: clean,
      mimeType: 'text/x-url',
      size: 0,
      base64: '',
      dataUrl: '',
      status: 'ready',
      displayKind: 'url',
      typeLabel: 'URL'
    };
    state.attachmentState.pending.push(pending);
    rebuildPreviewStrips();
  }

  // Create a synthetic text file from a large clipboard paste and queue it as a special editable artifact.
  // The artifact keeps the full original text for preview + editing and uploads it so the backend
  // treats it consistently as an attached file (text extraction, prompt blocks, history, etc.).
  function queuePastedTextArtifact(text) {
    // Use a clean, user-friendly filename for the uploaded artifact.
    // The timestamp is not needed for display; backend handles uniqueness via content/scope.
    const filename = 'pasted.txt';
    let syntheticFile;
    try {
      syntheticFile = new File([text], filename, {
        type: 'text/plain',
        lastModified: Date.now()
      });
    } catch (_err) {
      // Very old browsers fallback (rare)
      syntheticFile = new Blob([text], { type: 'text/plain' });
      syntheticFile.name = filename;
    }

    queueFiles([syntheticFile]);

    // Enrich the just-added pending item with artifact metadata (used by special renderer + editor).
    // queueFiles pushes synchronously before starting the async upload.
    const pendingList = state.attachmentState.pending;
    const last = pendingList.length ? pendingList[pendingList.length - 1] : null;
    if (last) {
      last.pasted = true;
      last.textContent = String(text || '');
      last.displayKind = 'pasted';
      last.typeLabel = t('dialog.pastedTypeLabel', null, 'Pasted');
      last.name = 'Pasted text';
      // Force a rebuild so the special pasted chip renders even before upload finishes.
      rebuildPreviewStrips();
    }
  }

  // Read selected files and queue them for the next request.
  function handleFileInput(event) {
    queueFiles(event.target.files || []);

    $(event.target).val('');
  }

  // Queue files dropped onto the chat shell overlay.
  function handleDroppedFiles(files) {
    queueFiles(files || []);
  }

  // Collect URLs from drag data (text/uri-list or plain text link) for drag & drop of links.
  function collectDroppedUrls(dataTransfer) {
    const out = [];
    if (!dataTransfer) return out;

    const uriList = dataTransfer.getData('text/uri-list') || '';
    if (uriList) {
      uriList.split(/\r?\n/).forEach(function (line) {
        const u = line.trim();
        if (u && isValidHttpUrl(u)) out.push(u);
      });
    }
    const plain = (dataTransfer.getData('text/plain') || '').trim();
    if (plain && isValidHttpUrl(plain) && !out.includes(plain)) {
      out.push(plain);
    }
    return out;
  }

  return {
    clearPendingAttachments,
    collectDroppedUrls,
    editPastedAttachment,
    handleDroppedFiles,
    handleFileInput,
    handleClipboardPaste,
    normalizeAttachment,
    queueUrlAttachment,
    rebuildPreviewStrips,
    removePendingAttachment,
    resolveAttachmentData,
    setUpdateSendButtons,
    updateAttachmentControls
  };
}
