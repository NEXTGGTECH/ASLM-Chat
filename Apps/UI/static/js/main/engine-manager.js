// Copyright NEXTGGTECH. Elastic License 2.0.

import { getJson, postJson } from './api.js';
import { t } from './i18n.js';
import { confirmDialog, textDialog } from '../ui/dialogs.js';
import { LLM_PARAMETER_OPTION_SETS } from './constants.js';
import { getEngineAdapter, getModelEndpoint, normalizeEngineValue } from '../engines/engine-registry.js';
import { isLocalHostname, normalizeAddressForParsing } from './utils.js';

// Engine manager.
// Create the runtime controller for engine selection, model loading, and presets.
export function createEngineManager(context, dependencies) {
  const { attachmentsUi, parametersUi } = dependencies;
  const { dom, state } = context;
  const LAST_MODEL_STORAGE_KEY = 'aslm.lastModelByEngine';

  // Last-model persistence helpers.
  // Read the serialized map of last selected models by engine.
  function readLastModelMap() {
    try {
      const raw = window.localStorage.getItem(LAST_MODEL_STORAGE_KEY);
      if (!raw) {
        return {};
      }
      const parsed = JSON.parse(raw);
      return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (_error) {
      return {};
    }
  }

  // Persist one model as the latest selection for one engine.
  function rememberLastModel(engine, modelName) {
    const normalizedEngine = normalizeEngineValue(engine);
    const normalizedModel = String(modelName || '').trim();
    if (!normalizedEngine || !normalizedModel) {
      return;
    }

    const nextMap = readLastModelMap();
    nextMap[normalizedEngine] = normalizedModel;
    try {
      window.localStorage.setItem(LAST_MODEL_STORAGE_KEY, JSON.stringify(nextMap));
    } catch (_error) {
      // Ignore storage failures in privacy-restricted contexts.
    }
  }

  // Read the latest remembered model for one engine.
  function getRememberedLastModel(engine) {
    const normalizedEngine = normalizeEngineValue(engine);
    const modelMap = readLastModelMap();
    return String(modelMap[normalizedEngine] || '').trim();
  }

  // Active selection helpers.
  // Read the active engine in canonical form.
  function getActiveEngine() {
    return normalizeEngineValue(state.activeEngine);
  }

  // Read the currently selected model name.
  function getSelectedModelName() {
    return String(dom.$modelSelector.val() || '').trim();
  }

  // Resolve the active preset record from the preset state.
  function getActivePreset() {
    return (state.presetState.presets || []).find(function findPreset(preset) {
      return preset.id === state.presetState.activePresetId;
    }) || null;
  }


  // Capability state helpers.
  // Reset the thinking state to the default unsupported values.
  function resetThinkState() {
    state.thinkState.supported = false;
    state.thinkState.paramName = 'think';
    state.thinkState.toggleSupported = false;
    state.thinkState.enabled = true;
    state.thinkState.levelSupported = false;
    state.thinkState.levelParamName = 'think_level';
    state.thinkState.levelOptions = [];
    state.thinkState.level = '';
  }

  // Reset the preset UI and in-memory preset metadata.
  function resetPresetUi() {
    state.presetState = {
      engine: '',
      model: '',
      activePresetId: '',
      presets: []
    };
    dom.$presetSelector.empty().append('<option value="">Default</option>');
    dom.$presetGroup.hide();
    dom.$presetRenameBtn.prop('disabled', true);
    dom.$presetDeleteBtn.prop('disabled', true);
  }

  // Apply preset payload data to the preset selector controls.
  function applyPresetState(payload) {
    const activeEngine = getActiveEngine();
    if (!payload || !getEngineAdapter(activeEngine).supportsPresets || !getSelectedModelName()) {
      resetPresetUi();
      return;
    }

    state.presetState = {
      engine: activeEngine,
      model: payload.model || getSelectedModelName(),
      activePresetId: payload.active_preset_id || '',
      presets: Array.isArray(payload.presets) ? payload.presets : []
    };

    dom.$presetSelector.empty();
    state.presetState.presets.forEach(function appendPreset(preset) {
      const label = preset.is_default ? `${preset.name} (Default)` : preset.name;
      const $option = $('<option>').val(preset.id).text(label);
      if (preset.id === state.presetState.activePresetId) {
        $option.prop('selected', true);
      }
      dom.$presetSelector.append($option);
    });

    const activePreset = getActivePreset();
    const isDefaultPreset = !activePreset || !!activePreset.is_default;
    dom.$presetRenameBtn.prop('disabled', isDefaultPreset);
    dom.$presetDeleteBtn.prop('disabled', isDefaultPreset);
    dom.$presetGroup.show();
  }


  // Endpoint helpers.
  // Resolve the runtime setting key used for an engine address.
  function getEngineAddressKey(engine) {
    return getEngineAdapter(engine).addressKey || null;
  }

  // Read the saved engine address from runtime settings.
  function getEngineAddress(engine) {
    const key = getEngineAddressKey(engine);
    return key ? (state.runtimeSettings[key] || '') : '';
  }

  // Resolve the runtime setting key used for an engine API key.
  function getEngineApiKeyKey(engine) {
    const canonicalEngine = normalizeEngineValue(engine);
    const runtimeKeys = state.runtimeSettings.engine_api_key_keys || {};
    return runtimeKeys[canonicalEngine] || getEngineAdapter(canonicalEngine).apiKeyKey || null;
  }

  // Report whether the selected engine already has a stored API key.
  function hasStoredEngineApiKey(engine) {
    const canonicalEngine = normalizeEngineValue(engine);
    const engineApiKeys = state.runtimeSettings.engine_api_keys || {};
    if (typeof engineApiKeys[canonicalEngine] === 'boolean') {
      return engineApiKeys[canonicalEngine];
    }

    const legacyFlag = `has_${canonicalEngine.replace(/-/g, '_')}_api_key`;
    return !!state.runtimeSettings[legacyFlag];
  }

  // Report whether the configured LM Studio address points to a local host.
  function isLocalLmsAddress() {
    const address = normalizeAddressForParsing(getEngineAddress('lms'));
    if (!address) {
      return true;
    }

    try {
      return isLocalHostname(new URL(address).hostname);
    } catch (_error) {
      return true;
    }
  }

  // Update the address status badge text and state.
  function setEngineAddressStatus(text, status) {
    dom.$engineAddressStatus.text(text || '');
    dom.$engineAddressStatus.removeClass('is-pending is-error');

    if (status) {
      dom.$engineAddressStatus.addClass(`is-${status}`);
    }
  }

  // Update the API key status badge text and state.
  function setEngineApiKeyStatus(text, status) {
    dom.$engineApiKeyStatus.text(text || '');
    dom.$engineApiKeyStatus.removeClass('is-pending is-error');

    if (status) {
      dom.$engineApiKeyStatus.addClass(`is-${status}`);
    }
  }

  // Rebuild the engine address and API key controls for the active engine.
  function updateEngineAddressUi() {
    const adapter = getEngineAdapter(getActiveEngine());
    const addressKey = adapter.addressKey || null;
    const hasEditableAddress = Boolean(addressKey);
    const apiKeyKey = getEngineApiKeyKey(adapter.id);
    const hasApiKeySupport = Boolean(apiKeyKey);
    const apiKeyRequired = Boolean(adapter.apiKeyRequired);
    const hasStoredApiKey = hasApiKeySupport && hasStoredEngineApiKey(adapter.id);

    dom.$engineAddressGroup.toggle(hasEditableAddress);
    dom.$engineAddressHint.text(adapter.addressHint || 'Configure the selected engine endpoint.');
    dom.$engineApiKeyGroup.toggle(hasApiKeySupport);

    if (!hasEditableAddress) {
      setEngineAddressStatus('Managed', null);
    } else {
      dom.$engineAddressInput.val(getEngineAddress(adapter.id));
      setEngineAddressStatus('Saved', null);
    }

    if (!hasApiKeySupport) {
      dom.$engineApiKeyToggleRow.show();
      dom.$engineApiKeyEnabled.prop('checked', false);
      dom.$engineApiKeyInput.val('').hide();
      setEngineApiKeyStatus('Off', null);
      return;
    }

    // Engines that always need a key (e.g. Gemini) skip the on/off toggle and
    // keep the key field visible so it can always be entered or replaced.
    dom.$engineApiKeyToggleRow.toggle(!apiKeyRequired);
    const keyFieldActive = apiKeyRequired || hasStoredApiKey;
    dom.$engineApiKeyEnabled.prop('checked', keyFieldActive);
    dom.$engineApiKeyInput.val('');
    dom.$engineApiKeyInput.toggle(keyFieldActive);
    dom.$engineApiKeyInput.attr(
      'placeholder',
      hasStoredApiKey ? 'Stored API key. Enter a new one to replace it' : 'Enter a new API key'
    );
    setEngineApiKeyStatus(hasStoredApiKey ? 'On' : 'Off', null);
  }

  // Reset all model-dependent UI state to a neutral placeholder.
  function resetModelUiState(message) {
    parametersUi.showModelPlaceholder(message || 'Models load on demand');
    state.currentModelInfo = null;
    resetPresetUi();
    parametersUi.resetDynamicPanels();
    parametersUi.updateVisibleDividers();
    state.visionState.supported = false;
    state.fileState.supported = false;
    resetThinkState();
    state.toolState.supported = false;
    parametersUi.updateAvailableToolServers(state.defaultAvailableToolServers);
    attachmentsUi.updateAttachmentControls();
    parametersUi.updateThinkControls();
    parametersUi.renderToolControls();
  }


  // Model cache helpers.
  // Drop the cached model list for one engine.
  function clearModelCache(engine) {
    delete state.modelsCache[normalizeEngineValue(engine)];
  }

  // Normalize and de-duplicate a raw model list.
  function normalizeModelNames(models) {
    return Array.from(new Set(
      (Array.isArray(models) ? models : []).map(function normalizeName(modelName) {
        return String(modelName || '').trim();
      }).filter(Boolean)
    ));
  }

  // Compare two model lists after normalization.
  function areModelListsEqual(left, right) {
    const first = normalizeModelNames(left);
    const second = normalizeModelNames(right);
    if (first.length !== second.length) {
      return false;
    }

    return first.every(function compare(modelName, index) {
      return modelName === second[index];
    });
  }

  // Read the best available model list for one engine.
  function getAvailableModelsForEngine(engine) {
    const canonicalEngine = normalizeEngineValue(engine);
    if (Array.isArray(state.modelsCache[canonicalEngine]) && state.modelsCache[canonicalEngine].length > 0) {
      return state.modelsCache[canonicalEngine].slice();
    }

    return dom.$modelSelector.find('option').map(function readOption() {
      return $(this).val();
    }).get().filter(Boolean);
  }


  // Model list refresh helpers (all engines).
  // Clear the scheduled model refresh timer.
  function clearModelsRefreshTimer() {
    if (state.modelsRefreshTimer !== null) {
      window.clearTimeout(state.modelsRefreshTimer);
      state.modelsRefreshTimer = null;
    }
  }

  // Compute the steady LM Studio model refresh interval.
  function getLmsSteadyRefreshInterval() {
    const adapter = getEngineAdapter('lms');
    if (typeof adapter.getModelRefreshInterval === 'function') {
      return adapter.getModelRefreshInterval(isLocalLmsAddress());
    }
    return isLocalLmsAddress() ? 3000 : 15000;
  }

  // Resolve the next refresh interval for one engine, or 0 to stop polling.
  function getModelsRefreshInterval(engine) {
    const canonicalEngine = normalizeEngineValue(engine);

    // Recover fast for every engine while the model list is empty / not found.
    if (getAvailableModelsForEngine(canonicalEngine).length === 0) {
      return 2000;
    }

    // Keep a steady background refresh for the locally managed engines.
    if (canonicalEngine === 'ollama-service') {
      return 5000;
    }
    if (canonicalEngine === 'lms') {
      return getLmsSteadyRefreshInterval();
    }

    // Cloud engines (OpenAI, Gemini) stop polling once a list is available.
    return 0;
  }

  // Schedule the next model refresh for the active engine.
  function scheduleModelsRefresh(delayMs) {
    clearModelsRefreshTimer();

    const intervalMs = typeof delayMs === 'number' ? delayMs : getModelsRefreshInterval(getActiveEngine());
    if (!intervalMs || intervalMs <= 0) {
      return;
    }

    state.modelsRefreshTimer = window.setTimeout(function triggerRefresh() {
      refreshActiveEngineModelList().catch(function onRefreshError(error) {
        console.error('Failed to refresh models:', error);
      });
    }, intervalMs);
  }

  // Restart polling for the active engine.
  function syncModelsRefresh() {
    scheduleModelsRefresh();
  }


  // Model list rendering.
  // Rebuild the model selector and return the chosen model value.
  function renderModelOptions(models, preferredModel) {
    const uniqueModels = normalizeModelNames(models);
    const fallbackModel = uniqueModels[0] || '';
    const rememberedModel = getRememberedLastModel(getActiveEngine());
    const selectedModel = uniqueModels.includes(preferredModel)
      ? preferredModel
      : (uniqueModels.includes(rememberedModel) ? rememberedModel : fallbackModel);

    dom.$modelSelector.empty();

    if (!uniqueModels.length) {
      dom.$modelSelector.append('<option value="">No models available</option>');
      return '';
    }

    uniqueModels.forEach(function appendModel(modelName) {
      const $option = $('<option>').val(modelName).text(modelName);
      if (modelName === selectedModel) {
        $option.prop('selected', true);
      }
      dom.$modelSelector.append($option);
    });

    return selectedModel;
  }

  // Fetch the model list for one engine, including Ollama warm-up retry.
  async function fetchModelsForEngine(engine, fetchOptions) {
    const forceRefresh = Boolean(fetchOptions && fetchOptions.refresh);
    const refreshSuffix = forceRefresh ? '&refresh=1' : '';

    // Perform one raw /api/models request.
    async function runFetch() {
      const data = await getJson(`/api/models/?engine=${encodeURIComponent(engine)}${refreshSuffix}`);
      return data.models || [];
    }

    const models = await runFetch();
    if (models.length > 0 || normalizeEngineValue(engine) !== 'ollama-service') {
      return models;
    }

    await new Promise(function waitForWarmup(resolve) {
      window.setTimeout(resolve, 1200);
    });
    return runFetch();
  }

  // Ensure the active engine has a rendered model list and loaded model info.
  async function ensureModelsLoadedForActiveEngine(options) {
    const loadOptions = options || {};
    const engine = getActiveEngine();
    const preferredModel = loadOptions.preferredModel || dom.$modelSelector.val() || getRememberedLastModel(engine) || '';

    if (Array.isArray(state.modelsCache[engine]) && state.modelsCache[engine].length > 0) {
      const selectedModel = renderModelOptions(state.modelsCache[engine], preferredModel);
      await loadModelInfo(selectedModel);
      return selectedModel;
    }

    resetModelUiState('Loading models...');
    const models = await fetchModelsForEngine(engine);
    state.modelsCache[engine] = models;
    const selectedModel = renderModelOptions(models, preferredModel);
    await loadModelInfo(selectedModel);
    return selectedModel;
  }

  // Reload models for the active engine after an address or key change.
  async function refreshActiveEngineModels(options) {
    const refreshOptions = options || {};
    const engine = normalizeEngineValue(refreshOptions.engine || getActiveEngine());
    const preferredModel = Object.prototype.hasOwnProperty.call(refreshOptions, 'preferredModel')
      ? (refreshOptions.preferredModel || '')
      : (dom.$modelSelector.val() || '');
    const selectionVersion = typeof refreshOptions.selectionVersion === 'number'
      ? refreshOptions.selectionVersion
      : state.engineSelectionVersion;

    clearModelCache(engine);

    if (selectionVersion !== state.engineSelectionVersion || engine !== getActiveEngine()) {
      return '';
    }

    updateEngineAddressUi();
    resetModelUiState(refreshOptions.loadingMessage || 'Loading models...');
    return ensureModelsLoadedForActiveEngine({
      preferredModel
    });
  }


  // Runtime settings.
  // Persist one runtime settings patch and refresh local state.
  async function saveRuntimeSettings(patch) {
    state.runtimeSettings = await postJson('/api/runtime_settings/', patch);
    return state.runtimeSettings;
  }

  // Switch the active engine and rebuild the model-related UI.
  async function applyEngineSelection(engine, options) {
    const settingsOptions = options || {};
    const normalizedEngine = normalizeEngineValue(engine);
    const selectionVersion = ++state.engineSelectionVersion;
    state.modelInfoRequestVersion += 1;
    const previousEngine = state.activeEngine;
    const autoLoadModels = settingsOptions.autoLoadModels !== false;

    state.activeEngine = normalizedEngine;
    clearModelsRefreshTimer();
    dom.$body.data('llm-engine', normalizedEngine);
    dom.$engineSelector.val(normalizedEngine);
    updateEngineAddressUi();
    resetModelUiState('Models load on demand');

    if (settingsOptions.persist === false) {
      state.runtimeSettings['llm-engine'] = normalizedEngine;
      clearModelCache(normalizedEngine);

      if (autoLoadModels) {
        try {
          await ensureModelsLoadedForActiveEngine({
            preferredModel: settingsOptions.preferredModel || ''
          });
        } catch (error) {
          console.error('Failed to load models after engine initialization:', error);
          resetModelUiState('No models available');
        }
      }
      syncModelsRefresh();
      return;
    }

    try {
      setEngineAddressStatus('Switching...', 'pending');
      state.runtimeSettings = await saveRuntimeSettings({ 'llm-engine': normalizedEngine });
      state.runtimeSettings['llm-engine'] = normalizedEngine;
      clearModelCache(normalizedEngine);

      if (selectionVersion !== state.engineSelectionVersion) {
        return;
      }

      updateEngineAddressUi();
      setEngineAddressStatus(getEngineAddressKey(normalizedEngine) ? 'Saved' : 'Managed', null);

      if (autoLoadModels) {
        try {
          await ensureModelsLoadedForActiveEngine({
            preferredModel: settingsOptions.preferredModel || ''
          });
        } catch (error) {
          console.error('Failed to load models after engine switch:', error);
          resetModelUiState('No models available');
        }
      }

      syncModelsRefresh();
    } catch (error) {
      state.activeEngine = previousEngine;
      state.runtimeSettings['llm-engine'] = previousEngine;
      dom.$body.data('llm-engine', previousEngine);
      dom.$engineSelector.val(previousEngine);
      updateEngineAddressUi();
      resetModelUiState('Models load on demand');
      syncModelsRefresh();
      throw error;
    }
  }

  // Refresh the active engine model list while polling is active.
  async function refreshActiveEngineModelList() {
    clearModelsRefreshTimer();

    const engine = getActiveEngine();

    if (state.modelsRefreshInFlight) {
      scheduleModelsRefresh();
      return;
    }

    const refreshVersion = state.engineSelectionVersion;
    const previousModels = normalizeModelNames(state.modelsCache[engine] || getAvailableModelsForEngine(engine));
    const previousSelectedModel = getSelectedModelName();
    const hadRenderedOptions = dom.$modelSelector.children().length > 0;

    state.modelsRefreshInFlight = true;

    try {
      // Bypass the backend list cache so the poll reflects the live engine state.
      const refreshedModels = normalizeModelNames(await fetchModelsForEngine(engine, { refresh: true }));

      if (refreshVersion !== state.engineSelectionVersion || getActiveEngine() !== engine) {
        return;
      }

      state.modelsCache[engine] = refreshedModels;

      const shouldRerender = !hadRenderedOptions
        || !areModelListsEqual(previousModels, refreshedModels)
        || (previousSelectedModel && !refreshedModels.includes(previousSelectedModel));

      if (!shouldRerender) {
        return;
      }

      const selectedModel = renderModelOptions(refreshedModels, previousSelectedModel);
      if (selectedModel !== previousSelectedModel) {
        await loadModelInfo(selectedModel);
      }
    } finally {
      state.modelsRefreshInFlight = false;

      if (refreshVersion === state.engineSelectionVersion && getActiveEngine() === engine) {
        scheduleModelsRefresh();
      }
    }
  }


  // Preset helpers.
  // Build the preset configuration payload for the active engine.
  function buildActivePresetConfigPayload() {
    const adapter = getEngineAdapter(getActiveEngine());
    const optionsPayload = parametersUi.collectOptionsPayload();
    if (typeof adapter.buildPresetConfig === 'function') {
      return adapter.buildPresetConfig(optionsPayload);
    }
    return optionsPayload;
  }

  // Sync the current option values into the active preset.
  async function syncActivePreset() {
    const adapter = getEngineAdapter(getActiveEngine());
    if (!adapter.presetApiBase) {
      return;
    }

    const modelName = getSelectedModelName();
    if (!modelName) {
      return;
    }

    const payload = await postJson(`${adapter.presetApiBase}/sync/`, {
      model: modelName,
      config: buildActivePresetConfigPayload()
    });
    applyPresetState(payload);
  }

  // Debounce preset sync calls while the user edits controls.
  function schedulePresetSync() {
    const adapter = getEngineAdapter(getActiveEngine());
    if (!adapter.supportsPresets) {
      return;
    }

    window.clearTimeout(state.presetSyncTimer);
    state.presetSyncTimer = window.setTimeout(function syncLater() {
      syncActivePreset().catch(function onSyncError(error) {
        console.error('Failed to sync preset:', error);
      });
    }, 220);
  }


  // Model info loading.
  // Load capabilities and defaults for the selected model.
  async function loadModelInfo(model) {
    const requestedEngine = getActiveEngine();
    const requestedEndpoint = getModelEndpoint(requestedEngine, state.runtimeSettings);
    const selectionVersion = state.engineSelectionVersion;
    const requestVersion = ++state.modelInfoRequestVersion;

    if (!model) {
      state.currentModelInfo = null;
      resetPresetUi();
      parametersUi.resetDynamicPanels();
      parametersUi.updateVisibleDividers();
      state.visionState.supported = false;
      state.fileState.supported = false;
      resetThinkState();
      state.toolState.supported = false;
      state.selectedToolServerIds = new Set();
      parametersUi.updateAvailableToolServers(state.defaultAvailableToolServers);
      attachmentsUi.updateAttachmentControls();
      parametersUi.updateThinkControls();
      parametersUi.renderToolControls();
      return;
    }

    try {
      const data = await getJson(`/api/model_info/?engine=${encodeURIComponent(requestedEngine)}&model=${encodeURIComponent(model)}`);

      if (requestVersion !== state.modelInfoRequestVersion || selectionVersion !== state.engineSelectionVersion
        || requestedEngine !== getActiveEngine() || model !== getSelectedModelName()) {
        return;
      }

      // Rebuild all capability-dependent panels from the latest payload.
      state.currentModelInfo = data;
      state.currentModelInfoEndpoint = requestedEndpoint;
      rememberLastModel(requestedEngine, model);
      parametersUi.resetDynamicPanels();
      applyPresetState(
        data.ollama_presets
        || data.lms_presets
        || data.openai_presets
        || data.google_genai_presets
        || null
      );

      state.toolState.supported = !!data.supports_tool_calling;
      parametersUi.updateAvailableToolServers(data.available_tool_servers || state.defaultAvailableToolServers);
      if (!state.toolState.supported) {
        state.selectedToolServerIds = new Set();
        parametersUi.renderToolControls();
      }

      state.visionState.supported = !!data.supports_vision;
      state.fileState.supported = !!data.supports_files;
      attachmentsUi.updateAttachmentControls();
      // Do not drop queued attachments during routine model-info refreshes.
      // Pending files should only be cleared explicitly by user action,
      // after send, or when starting a new chat.

      state.thinkState.supported = !!data.supports_thinking;
      state.thinkState.paramName = data.think_param_name || 'think';
      state.thinkState.levelSupported = !!data.supports_think_level;
      state.thinkState.toggleSupported = data.supports_think_toggle === undefined
        ? false
        : !!data.supports_think_toggle;
      state.thinkState.levelParamName = data.think_level_param_name || 'think_level';
      const metadataThinkLevelOptions = Array.isArray(data.think_level_options) && data.think_level_options.length > 0
        ? data.think_level_options.map(function mapOption(value) { return String(value); })
        : [];
      const fallbackThinkLevelOptions = state.thinkState.levelSupported
        ? (LLM_PARAMETER_OPTION_SETS[state.thinkState.levelParamName] || [])
        : [];
      state.thinkState.levelOptions = metadataThinkLevelOptions.length > 0
        ? metadataThinkLevelOptions
        : fallbackThinkLevelOptions.slice();
      state.thinkState.enabled = data.defaults && data.defaults[state.thinkState.paramName] !== undefined
        ? String(data.defaults[state.thinkState.paramName]).toLowerCase() === 'true' || data.defaults[state.thinkState.paramName] === true
        : true;
      state.thinkState.level = data.defaults && data.defaults[state.thinkState.levelParamName] !== undefined
        ? String(data.defaults[state.thinkState.levelParamName])
        : (state.thinkState.levelOptions[0] || '');
      if (
        state.thinkState.levelSupported
        && state.thinkState.levelOptions.length > 0
        && String(state.thinkState.level || '').toLowerCase() === 'off'
        && state.thinkState.enabled
      ) {
        state.thinkState.level = state.thinkState.levelOptions[0];
      }

      parametersUi.renderThinkLevelControls();
      parametersUi.updateThinkControls();

      if (!data.defaults) {
        parametersUi.updateVisibleDividers();
        return;
      }

      // Remove thinking keys because they are rendered by dedicated controls.
      const adapter = getEngineAdapter(getActiveEngine());
      let defaults = { ...data.defaults };
      delete defaults[state.thinkState.paramName];
      delete defaults[state.thinkState.levelParamName];
      if (typeof adapter.sanitizeModelDefaults === 'function') {
        defaults = adapter.sanitizeModelDefaults(defaults);
      }

      parametersUi.renderModelParameters(data, defaults);
    } catch (error) {
      if (requestVersion !== state.modelInfoRequestVersion || selectionVersion !== state.engineSelectionVersion) {
        return;
      }

      state.currentModelInfo = null;
      resetPresetUi();
      parametersUi.resetDynamicPanels();
      parametersUi.updateVisibleDividers();
      parametersUi.updateAvailableToolServers(state.defaultAvailableToolServers);
      state.visionState.supported = false;
      state.fileState.supported = false;
      resetThinkState();
      state.toolState.supported = false;
      parametersUi.renderThinkLevelControls();
      attachmentsUi.updateAttachmentControls();
      parametersUi.updateThinkControls();
      parametersUi.renderToolControls();
      console.error('Failed to load model parameters', error);
    }
  }


  // Preset actions.
  // Activate one preset for the selected model.
  async function selectPreset(presetId) {
    const adapter = getEngineAdapter(getActiveEngine());
    const modelName = getSelectedModelName();
    if (!adapter.presetApiBase || !presetId || !modelName) {
      return;
    }

    const payload = await postJson(`${adapter.presetApiBase}/select/`, {
      model: modelName,
      preset_id: presetId
    });
    applyPresetState(payload);
    await loadModelInfo(modelName);
  }

  // Create a new preset from the current control values.
  async function createPreset() {
    const adapter = getEngineAdapter(getActiveEngine());
    const modelName = getSelectedModelName();
    if (!adapter.presetApiBase || !modelName) {
      return;
    }

    const requestedName = await textDialog({
      title: t('settings.presetNewTitle', null, 'New preset'),
      label: t('settings.presetNameLabel', null, 'Preset name'),
      confirmText: t('settings.presetNew', null, 'New'),
      required: false
    });
    if (requestedName === null) {
      return;
    }

    const payload = await postJson(`${adapter.presetApiBase}/create/`, {
      model: modelName,
      name: requestedName.trim(),
      config: buildActivePresetConfigPayload()
    });
    applyPresetState(payload);
    await loadModelInfo(modelName);
  }

  // Rename the active custom preset.
  async function renamePreset() {
    const adapter = getEngineAdapter(getActiveEngine());
    const activePreset = getActivePreset();
    const modelName = getSelectedModelName();
    if (!adapter.presetApiBase || !modelName || !activePreset || activePreset.is_default) {
      return;
    }

    const requestedName = await textDialog({
      title: t('settings.presetRenameTitle', null, 'Rename preset'),
      label: t('settings.presetNameLabel', null, 'Preset name'),
      value: activePreset.name || '',
      confirmText: t('sidebar.rename', null, 'Rename')
    });
    if (requestedName === null) {
      return;
    }

    const payload = await postJson(`${adapter.presetApiBase}/rename/`, {
      model: modelName,
      preset_id: activePreset.id,
      name: requestedName.trim()
    });
    applyPresetState(payload);
  }

  // Delete the active custom preset and refresh model info.
  async function deletePreset() {
    const adapter = getEngineAdapter(getActiveEngine());
    const activePreset = getActivePreset();
    const modelName = getSelectedModelName();
    if (!adapter.presetApiBase || !modelName || !activePreset || activePreset.is_default) {
      return;
    }

    const confirmed = await confirmDialog({
      title: t('confirm.deletePresetTitle', null, 'Delete preset'),
      message: t('confirm.deletePreset', { name: activePreset.name }, `Delete preset "${activePreset.name}"?`),
      confirmText: t('settings.presetDelete', null, 'Delete'),
      danger: true
    });
    if (!confirmed) {
      return;
    }

    const payload = await postJson(`${adapter.presetApiBase}/delete/`, {
      model: modelName,
      preset_id: activePreset.id
    });
    applyPresetState(payload);
    await loadModelInfo(modelName);
  }


  // Persistence actions.
  // Persist the active engine address and reload models when needed.
  async function persistEngineAddress() {
    const engine = getActiveEngine();
    const addressKey = getEngineAddressKey(engine);
    const addressValue = dom.$engineAddressInput.val().trim();
    const selectionVersion = ++state.engineSelectionVersion;

    if (!addressKey) {
      return;
    }

    clearModelsRefreshTimer();

    if ((state.runtimeSettings[addressKey] || '') === addressValue) {
      setEngineAddressStatus('Saved', null);
      syncModelsRefresh();
      return;
    }

    try {
      setEngineAddressStatus('Saving...', 'pending');
      state.runtimeSettings = await saveRuntimeSettings({ [addressKey]: addressValue });
      await refreshActiveEngineModels({
        engine,
        selectionVersion,
        preferredModel: ''
      });
      syncModelsRefresh();
    } catch (error) {
      console.error('Failed to save engine address:', error);
      resetModelUiState('No models available');
      setEngineAddressStatus('Error', 'error');
      syncModelsRefresh();
    }
  }

  // Toggle the active engine API key state.
  async function handleApiKeyToggle() {
    const engine = getActiveEngine();
    const apiKeyKey = getEngineApiKeyKey(engine);
    if (!apiKeyKey) {
      return;
    }

    const isEnabled = dom.$engineApiKeyEnabled.is(':checked');
    dom.$engineApiKeyInput.toggle(isEnabled);
    if (isEnabled) {
      setEngineApiKeyStatus('On', null);
      dom.$engineApiKeyInput.trigger('focus');
      return;
    }

    try {
      const selectionVersion = ++state.engineSelectionVersion;
      setEngineApiKeyStatus('Saving...', 'pending');
      state.runtimeSettings = await saveRuntimeSettings({ [apiKeyKey]: '' });
      await refreshActiveEngineModels({
        engine,
        selectionVersion
      });
    } catch (error) {
      console.error('Failed to update API key state:', error);
      resetModelUiState('No models available');
      setEngineApiKeyStatus('Error', 'error');
    }
  }

  // Persist a new API key for the active engine.
  async function persistApiKey() {
    const engine = getActiveEngine();
    const apiKeyKey = getEngineApiKeyKey(engine);
    if (!apiKeyKey || !dom.$engineApiKeyEnabled.is(':checked')) {
      return;
    }

    const apiKeyValue = dom.$engineApiKeyInput.val().trim();
    if (!apiKeyValue) {
      setEngineApiKeyStatus(hasStoredEngineApiKey(engine) ? 'On' : 'Off', null);
      return;
    }

    try {
      const selectionVersion = ++state.engineSelectionVersion;
      setEngineApiKeyStatus('Saving...', 'pending');
      state.runtimeSettings = await saveRuntimeSettings({ [apiKeyKey]: apiKeyValue });
      dom.$engineApiKeyInput.val('');
      await refreshActiveEngineModels({
        engine,
        selectionVersion
      });
    } catch (error) {
      console.error('Failed to save API key:', error);
      resetModelUiState('No models available');
      setEngineApiKeyStatus('Error', 'error');
    }
  }

  return {
    applyEngineSelection,
    createPreset,
    deletePreset,
    fetchModelsForEngine,
    getActiveEngine,
    getActivePreset,
    getEngineAddressKey,
    getEngineApiKeyKey,
    getSelectedModelName,
    hasStoredEngineApiKey,
    handleApiKeyToggle,
    loadModelInfo,
    persistApiKey,
    persistEngineAddress,
    refreshActiveEngineModels,
    rememberLastModel,
    renamePreset,
    resetModelUiState,
    schedulePresetSync,
    selectPreset,
    setEngineAddressStatus,
    syncModelsRefresh,
    updateEngineAddressUi
  };
}
