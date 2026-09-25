// Copyright NEXTGGTECH. Elastic License 2.0.

import { normalizeEngineValue } from '../engines/engine-registry.js';
import { initI18n, t } from './i18n.js';
import { escHtml, escapeAttributeValue, parseJsonScript } from './utils.js';

initI18n();

// Icon helpers.
// Build one SVG sprite reference string.
function svgIcon(iconPath, attrs) {
  return `<svg ${attrs}><use href="${iconPath}#icon"></use></svg>`;
}

// Build one image icon string for bitmap UI assets.
function imageIcon(iconPath, className, altText) {
  return `<img class="${className}" src="${iconPath}" alt="${altText || ''}" aria-hidden="true">`;
}


// Application context.
// Create the shared DOM map, icon set, and runtime state.
export function createAppContext() {
  const dom = {
    $body: $('body'),
    $chatShell: $('.chat-shell'),
    $newChatBtn: $('#newChatBtn'),
    $historyList: $('#historyList'),
    $chatTitle: $('#chatTitle'),
    $messagesArea: $('#messagesArea'),
    $messagesInner: $('#messagesInner'),
    $welcomeScreen: $('#welcomeScreen'),
    $chatInput: $('#chatInput'),
    $sendBtn: $('#sendBtn'),
    $contextUsageBtn: $('#contextUsageBtn'),
    $chatInputConv: $('#chatInputConv'),
    $sendBtnConv: $('#sendBtnConv'),
    $contextUsageBtnConv: $('#contextUsageBtnConv'),
    $conversationInput: $('#conversationInput'),
    $engineSelector: $('#engineSelector'),
    $engineAddressGroup: $('#engineAddressGroup'),
    $engineAddressInput: $('#engineAddressInput'),
    $engineAddressStatus: $('#engineAddressStatus'),
    $engineAddressHint: $('#engineAddressHint'),
    $engineApiKeyGroup: $('#engineApiKeyGroup'),
    $engineApiKeyToggleRow: $('#engineApiKeyToggleRow'),
    $engineApiKeyEnabled: $('#engineApiKeyEnabled'),
    $engineApiKeyInput: $('#engineApiKeyInput'),
    $engineApiKeyStatus: $('#engineApiKeyStatus'),
    $modelSelector: $('#modelSelector'),
    $presetGroup: $('#ollamaPresetGroup'),
    $presetSelector: $('#ollamaPresetSelector'),
    $presetCreateBtn: $('#ollamaPresetCreateBtn'),
    $presetRenameBtn: $('#ollamaPresetRenameBtn'),
    $presetDeleteBtn: $('#ollamaPresetDeleteBtn'),
    $groupTools: $('#group-tools'),
    $dividerTools: $('#divider-tools'),
    $skillsSettingsContent: $('#skillsSettingsContent'),
    $toolInspectorModal: $('#toolInspectorModal'),
    $chatItemDropdown: $('#chatItemDropdown'),
    $systemPrompt: $('#systemPrompt'),
    $imageInput: $('#imageInput'),
    $imageInputConv: $('#imageInputConv'),
    $imagePreviewStrip: $('#imagePreviewStrip'),
    $imagePreviewStripConv: $('#imagePreviewStripConv'),
    $composerMenuBtn: $('#composerMenuBtn'),
    $composerMenuBtnConv: $('#composerMenuBtnConv'),
    $composerMenuPopover: $('#composerMenuPopover'),
    $composerMenuPopoverConv: $('#composerMenuPopoverConv'),
    $composerToolLists: $('.composer-tool-list'),
    $composerToolMenus: $('[data-tool-menu]'),
    $composerModePills: $('.composer-mode-pills'),
    $composerMcpHosts: $('.composer-mcp-host'),
    $composerSkillsMenus: $('[data-skills-menu]'),
    $composerSkillsHosts: $('.composer-skills-host'),
    $attachBtn: $('#attachBtn'),
    $attachBtnConv: $('#attachBtnConv'),
    $thinkToggleBtn: $('#thinkToggleBtn'),
    $thinkToggleBtnConv: $('#thinkToggleBtnConv'),
    $thinkLevelSelector: $('#thinkLevelSelector'),
    $thinkLevelSelectorConv: $('#thinkLevelSelectorConv'),
    $sidebarRight: $('#sidebarRight'),
    $sidebarRightToggle: $('#sidebarRightToggle')
  };

  const runtimeSettings = parseJsonScript('runtimeSettingsData') || {};
  const defaultAllToolServers = parseJsonScript('availableToolServersData') || [];
  const defaultAvailableToolServers = defaultAllToolServers.filter(function filterBundled(server) {
    return !server || !server.user_mcp;
  });
  const defaultUserMcpToolServers = defaultAllToolServers.filter(function filterUserMcp(server) {
    return server && server.user_mcp;
  });
  const uiIconPaths = parseJsonScript('uiIconPathsData') || {};

  const icons = {
    STOP_ICON: svgIcon(uiIconPaths.stopSquare, 'width="18" height="18" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    SEND_ICON: svgIcon(uiIconPaths.send, 'width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" aria-hidden="true"'),
    REMOVE_ATTACHMENT_ICON: svgIcon(uiIconPaths.removeAttachment, 'width="10" height="10" fill="none" stroke="currentColor" stroke-width="2.5" viewBox="0 0 24 24" aria-hidden="true"'),
    COPY_MESSAGE_ICON: svgIcon(uiIconPaths.copy, 'width="14" height="14" viewBox="0 0 24 24" aria-hidden="true"'),
    REGENERATE_ICON: svgIcon(uiIconPaths.refresh, 'width="18" height="18" viewBox="0 -960 960 960" aria-hidden="true"'),
    BRANCH_ICON: svgIcon(uiIconPaths.branch, 'width="17" height="17" viewBox="0 0 21 21" aria-hidden="true"'),
    EDIT_MESSAGE_ICON: svgIcon(uiIconPaths.pen, 'width="15" height="15" viewBox="0 0 24 24" aria-hidden="true"'),
    DELETE_MESSAGE_ICON: svgIcon(uiIconPaths.trash, 'width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" aria-hidden="true"'),
    COPIED_ICON: svgIcon(uiIconPaths.check, 'width="14" height="14" viewBox="0 -960 960 960" aria-hidden="true"'),
    ARROW_LEFT_ICON: svgIcon(uiIconPaths.arrowLeft, 'width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" aria-hidden="true"'),
    CHAT_ITEM_ICON: svgIcon(uiIconPaths.chatBubble, 'width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" aria-hidden="true"'),
    GLOBE_ICON: svgIcon(uiIconPaths.globe, 'width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" aria-hidden="true"'),
    EYE_ICON: svgIcon(uiIconPaths.eye, 'width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" aria-hidden="true"'),
    CODE_ICON: svgIcon(uiIconPaths.toolCode, 'width="16" height="16" fill="currentColor" viewBox="0 -960 960 960" aria-hidden="true"'),
    ADD_ICON: svgIcon(uiIconPaths.addMenu, 'width="16" height="16" fill="currentColor" viewBox="0 -960 960 960" aria-hidden="true"'),
    WEB_SEARCH_ICON: imageIcon(uiIconPaths.webSearch, 'msg-search-main-icon', ''),
    WEB_SEARCH_ERROR_ICON: imageIcon(uiIconPaths.webSearchError, 'msg-search-main-icon', ''),
    TOOL_CODE_EXEC_ICON: imageIcon(uiIconPaths.toolBash, 'msg-tool-svg-icon', ''),
    TOOL_BASH_ICON: imageIcon(uiIconPaths.toolTerminal || uiIconPaths.toolBash, 'msg-tool-svg-icon', ''),
    TOOL_MAKE_FILE_ICON: imageIcon(uiIconPaths.toolMakeFile, 'msg-tool-svg-icon', ''),
    TOOL_EDIT_FILE_ICON: imageIcon(uiIconPaths.toolEditFile, 'msg-tool-svg-icon', ''),
    TOOL_SEARCH_ICON: imageIcon(uiIconPaths.toolSearch, 'msg-tool-svg-icon', ''),
    TOOL_IMAGE_VIEW_ICON: imageIcon(uiIconPaths.toolImageView, 'msg-tool-svg-icon', ''),
    BROWSER_CURSOR_ICON: svgIcon(uiIconPaths.browserCursor, 'width="16" height="16" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    BROWSER_KEYBOARD_ICON: svgIcon(uiIconPaths.browserKeyboard, 'width="16" height="16" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    BROWSER_NAVIGATE_ICON: svgIcon(uiIconPaths.browserNavigate, 'width="16" height="16" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    BROWSER_SCROLL_ICON: svgIcon(uiIconPaths.browserScroll, 'width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24" aria-hidden="true"'),
    BROWSER_TYPE_ICON: svgIcon(uiIconPaths.browserType, 'width="16" height="16" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    BROWSER_DISCONNECTED_ICON: svgIcon(uiIconPaths.browserDisconnected, 'width="16" height="16" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    CHAT_ITEM_MENU_ICON: svgIcon(uiIconPaths.ellipsisVertical, 'width="14" height="14" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    PLAY_ICON: svgIcon(uiIconPaths.play, 'width="20" height="20" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    PAUSE_ICON: svgIcon(uiIconPaths.pause, 'width="20" height="20" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    VOLUME_ICON: svgIcon(uiIconPaths.volume, 'width="18" height="18" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    VOLUME_MUTED_ICON: svgIcon(uiIconPaths.volumeMuted, 'width="18" height="18" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    FULLSCREEN_ICON: svgIcon(uiIconPaths.fullscreen, 'width="18" height="18" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    FULLSCREEN_EXIT_ICON: svgIcon(uiIconPaths.fullscreenExit, 'width="18" height="18" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    POPOUT_ICON: svgIcon(uiIconPaths.popout, 'width="18" height="18" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    DOCK_ICON: svgIcon(uiIconPaths.dock, 'width="18" height="18" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    DOWNLOAD_FILE_ICON: svgIcon(uiIconPaths.download, 'width="20" height="20" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    AUDIO_FILE_ICON: svgIcon(uiIconPaths.audio, 'width="20" height="20" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    VIDEO_FILE_ICON: svgIcon(uiIconPaths.video, 'width="20" height="20" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true"'),
    CLOSE_ICON: svgIcon(uiIconPaths.removeAttachment, 'width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.25" viewBox="0 0 24 24" aria-hidden="true"'),
    SKILLS_FOLDER_ICON: svgIcon(uiIconPaths.skillsFolder, 'width="16" height="16" fill="currentColor" viewBox="0 -960 960 960" aria-hidden="true"'),
    SKILLS_FOLDER_OPEN_ICON: svgIcon(uiIconPaths.skillsFolderOpen, 'width="16" height="16" fill="currentColor" viewBox="0 -960 960 960" aria-hidden="true"'),
    SKILLS_FILE_ICON: svgIcon(uiIconPaths.skillsFile, 'width="16" height="16" fill="currentColor" viewBox="0 -960 960 960" aria-hidden="true"'),
    // Composer flyout action icons (match Attach files sizing).
    SKILLS_MANAGE_ICON: svgIcon(uiIconPaths.skillsManage, 'width="18" height="18" fill="currentColor" viewBox="0 -960 960 960" aria-hidden="true"'),
    MCP_EDIT_ICON: svgIcon(uiIconPaths.mcpEdit, 'width="18" height="18" fill="currentColor" viewBox="0 -960 960 960" aria-hidden="true"'),
    MCP_RELOAD_ICON: svgIcon(uiIconPaths.mcpReload, 'width="18" height="18" fill="currentColor" viewBox="0 -960 960 960" aria-hidden="true"')
  };

  // Build the shared action buttons used by assistant and user rows.
  icons.buildMessageActionsHtml = function buildMessageActionsHtml(isUser) {
    const sourcesLabel = t('sources.title', null, 'Sources');
    const sourcesLabelAttr = escapeAttributeValue(sourcesLabel);
    const branchLabel = escapeAttributeValue(t('messages.branch', null, 'Branch from here'));
    const editLabel = escapeAttributeValue(t('messages.editMessage', null, 'Edit message'));
    return `<div class="msg-actions">
      <button class="msg-action-btn msg-copy-btn" title="Copy" aria-label="Copy message">${icons.COPY_MESSAGE_ICON}</button>
      ${isUser ? '' : `<button class="msg-action-btn msg-branch-btn" title="${branchLabel}" aria-label="${branchLabel}">${icons.BRANCH_ICON}</button>`}
      ${isUser ? `<button class="msg-action-btn msg-edit-btn" title="${editLabel}" aria-label="${editLabel}">${icons.EDIT_MESSAGE_ICON}</button>` : ''}
      <button class="msg-action-btn msg-regen-btn" title="Regenerate" aria-label="Regenerate response">${icons.REGENERATE_ICON}</button>
      <button class="msg-action-btn msg-delete-btn" title="Delete" aria-label="Delete message">${icons.DELETE_MESSAGE_ICON}</button>
      <button class="msg-action-btn msg-sources-btn" type="button" title="${sourcesLabelAttr}" aria-label="${sourcesLabelAttr}" aria-expanded="false" hidden>
        <span class="msg-sources-btn-icons" aria-hidden="true">${icons.GLOBE_ICON}</span>
        <span class="msg-sources-btn-label">${escHtml(sourcesLabel)}</span>
      </button>
    </div>`;
  };

  return {
    dom,
    icons,
    state: {
      runtimeSettings,
      defaultAvailableToolServers,
      availableToolServers: Array.isArray(defaultAvailableToolServers) ? defaultAvailableToolServers.slice() : [],
      userMcpToolServers: Array.isArray(defaultUserMcpToolServers) ? defaultUserMcpToolServers.slice() : [],
      selectedToolServerIds: new Set(),
      deepResearchEnabled: false,
      currentChatId: null,
      engineSelectionVersion: 0,
      modelInfoRequestVersion: 0,
      activeEngine: normalizeEngineValue(runtimeSettings['llm-engine'] || dom.$body.data('llm-engine') || 'ollama-service'),
      modelsCache: {},
      modelsRefreshTimer: null,
      modelsRefreshInFlight: false,
      presetState: {
        engine: '',
        model: '',
        activePresetId: '',
        presets: []
      },
      presetSyncTimer: null,
      isChatGenerating: false,
      currentAbortController: null,
      queuedMessageCounter: 0,
      chatRequestQueue: [],
      contextUsage: {},
      currentModelInfo: null,
      currentModelInfoEndpoint: null,
      activeMenuTarget: null,
      visionState: {
        supported: false
      },
      fileState: {
        supported: false
      },
      attachmentState: {
        pending: []
      },
      thinkState: {
        supported: false,
        paramName: 'think',
        toggleSupported: false,
        enabled: true,
        levelSupported: false,
        levelParamName: 'think_level',
        levelOptions: [],
        level: ''
      },
      toolState: {
        supported: false
      }
    }
  };
}
