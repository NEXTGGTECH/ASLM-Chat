# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from core.extract.scoring import query_terms
from core.search.quality import _SUSPICIOUS_PACK_CAP, emd_penalty, suspicious_url_penalty
from core.search.triage import TriageAction, TriageSession


# Suspicious URL penalty

# Verify that ordinary URLs (including second-level country registries such as .com.br and .co.uk) get no
# penalty. None of the bad-site URL tells (identity-blind, zero I/O) may fire on the ordinary URLs that
# make up the bulk of real SERPs.
def test_clean_urls_are_untouched():
    for url in (
        "https://www.anthropic.com/news/claude-sonnet-5",
        "https://docs.anthropic.com/en/docs/claude-code/overview",
        "https://github.com/anthropics/claude-code",
        "https://example.com.br/artigo",
        "https://example.co.uk/story",
        "https://en.wikipedia.org/wiki/Claude",
    ):
        assert suspicious_url_penalty(url) == 0.0, url


# Verify that an IP literal host and a non-default port are penalized, while the default HTTPS port is
# not.
def test_ip_literal_and_port_tells():
    assert suspicious_url_penalty("https://203.0.113.7/download") < 0.0
    assert suspicious_url_penalty("https://site.example:8080/page") < 0.0
    assert suspicious_url_penalty("https://site.example:443/page") == 0.0


# Verify that a well-known domain name embedded in a foreign domain (github.com.evil-mirror.xyz) is
# penalized, while the real github.com does not trip its own name.
def test_embedded_gtld_costume():
    assert suspicious_url_penalty("https://github.com.evil-mirror.xyz/login") < 0.0
    assert suspicious_url_penalty("https://github.com/anthropics") == 0.0


# Verify that many-hyphen and year-stamped domain names are penalized, while two hyphens are ordinary
# branding and not a tell.
def test_hyphen_and_year_stamped_names():
    assert suspicious_url_penalty("https://best-ai-coding-tools-2026.com/review") < 0.0
    assert suspicious_url_penalty("https://top10vpn2024.net/list") < 0.0
    assert suspicious_url_penalty("https://my-cool-site.com/post") == 0.0


# Verify that a URL with several tells is capped at the pack cap.
def test_pack_is_capped():
    ugly = "https://github.com.best-free-vpn-review-2026.evil:8080/x"
    assert suspicious_url_penalty(ugly) == -_SUSPICIOUS_PACK_CAP


# Exact-match domain penalty

# Verify that exact-match domains built from the query terms (claudecode.pro, claude-code.dev) are
# penalized.
def test_emd_hits_query_squatter():
    terms = tuple(query_terms("claude code official documentation"))
    assert emd_penalty("https://claudecode.pro/docs", terms) < 0.0
    assert emd_penalty("https://claude-code.dev/docs", terms) < 0.0


# Verify that partial matches and official domains get no exact-match penalty, and that a single-term
# query never fires since every brand is its own exact match.
def test_emd_leaves_partial_and_official_domains_alone():
    terms = tuple(query_terms("claude code official documentation"))
    assert emd_penalty("https://claude.com/product/claude-code", terms) == 0.0
    assert emd_penalty("https://code.claude.com/docs", terms) == 0.0
    assert emd_penalty("https://github.com/anthropics", terms) == 0.0
    assert emd_penalty("https://claudecode.pro/docs", tuple(query_terms("claudecode"))) == 0.0


# Triage integration

# Verify that URL tells demote a squatter source below the clean official one without ever skipping it:
# the clean source is parsed and the squatter scores lower but is not skipped.
def test_tells_demote_but_never_skip():
    session = TriageSession("claude code official documentation")
    clean = session.ingest_source(
        engine="google", provider_family="google", rank=1,
        url="https://docs.anthropic.com/claude-code",
        title="Claude Code official documentation",
        snippet="The official documentation for Claude Code: setup, usage and features.",
    )
    squatter = session.ingest_source(
        engine="google", provider_family="google", rank=2,
        url="https://claude-code-docs-2026.xyz/claude-code",
        title="Claude Code official documentation",
        snippet="The official documentation for Claude Code: setup, usage and features.",
    )
    assert clean.action == TriageAction.PARSE
    assert squatter.score < clean.score
    assert squatter.action != TriageAction.SKIP
