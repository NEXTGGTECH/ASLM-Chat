// Copyright NEXTGGTECH. Elastic License 2.0.

import { googleGenAiAdapter } from './google-genai.js';
import { lmsAdapter } from './lms.js';
import { ollamaServiceAdapter } from './ollama-service.js';
import { openAiAdapter } from './openai.js';

const ENGINE_ADAPTERS = [
  ollamaServiceAdapter,
  lmsAdapter,
  openAiAdapter,
  googleGenAiAdapter
];

const adaptersById = new Map(
  ENGINE_ADAPTERS.map(function mapAdapter(adapter) {
    return [adapter.id, adapter];
  })
);

const aliasMap = new Map();

ENGINE_ADAPTERS.forEach(function registerAdapter(adapter) {
  aliasMap.set(adapter.id, adapter.id);

  (adapter.aliases || []).forEach(function registerAlias(alias) {
    aliasMap.set(String(alias || '').trim().toLowerCase(), adapter.id);
  });
});

// Engine lookup helpers.
// Normalize any engine alias into a canonical engine id.
export function normalizeEngineValue(engine) {
  const normalized = String(engine || '').trim().toLowerCase();
  return aliasMap.get(normalized) || normalized || ollamaServiceAdapter.id;
}

// Resolve one engine adapter from the registry.
export function getEngineAdapter(engine) {
  return adaptersById.get(normalizeEngineValue(engine)) || ollamaServiceAdapter;
}

// Metadata is scoped to the server URL even when its address control is hidden.
export function getModelEndpoint(engine, runtimeSettings) {
  const canonical = normalizeEngineValue(engine);
  return canonical === 'ollama-service'
    ? ''
    : String(runtimeSettings.engine_urls?.[canonical] || '').trim().replace(/\/+$/, '');
}

// Report whether the engine supports presets.
export function isPresetCapableEngine(engine) {
  return !!getEngineAdapter(engine).supportsPresets;
}
