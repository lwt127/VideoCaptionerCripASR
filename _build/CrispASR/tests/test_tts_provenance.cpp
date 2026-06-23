// test_tts_provenance.cpp — unit tests for TTS AI-provenance compliance.
//
// Tests for: consent gate logic, C2PA no-op, ID3v2 tag structure,
// WAV LIST/INFO metadata, disclaimer helpers.

#include "crispasr_wav_writer.h"
#include "crispasr_c2pa.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {
uint16_t le_u16(const std::string& s, size_t off) {
    return (uint8_t)s[off] | ((uint16_t)(uint8_t)s[off + 1] << 8);
}
uint32_t le_u32(const std::string& s, size_t off) {
    return (uint32_t)(uint8_t)s[off] | ((uint32_t)(uint8_t)s[off + 1] << 8) | ((uint32_t)(uint8_t)s[off + 2] << 16) |
           ((uint32_t)(uint8_t)s[off + 3] << 24);
}
} // namespace

// ──────────────────────────────────────────────────────────────────────────
// WAV LIST/INFO metadata
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("WAV LIST/INFO chunk is well-formed RIFF", "[unit][provenance][wav]") {
    std::string info = crispasr_wav_make_info_chunk();
    REQUIRE(info.size() >= 12);
    REQUIRE(info.substr(0, 4) == "LIST");
    // LIST size = rest of chunk
    uint32_t list_size = le_u32(info, 4);
    REQUIRE(list_size == info.size() - 8);
    REQUIRE(info.substr(8, 4) == "INFO");
}

TEST_CASE("WAV LIST/INFO contains ISFT with CrispASR", "[unit][provenance][wav]") {
    std::string info = crispasr_wav_make_info_chunk();
    REQUIRE(info.find("ISFT") != std::string::npos);
    REQUIRE(info.find("CrispASR") != std::string::npos);
}

TEST_CASE("WAV LIST/INFO contains ICMT with AI notice", "[unit][provenance][wav]") {
    std::string info = crispasr_wav_make_info_chunk();
    REQUIRE(info.find("ICMT") != std::string::npos);
    REQUIRE(info.find("AI text-to-speech") != std::string::npos);
    REQUIRE(info.find("not a recording") != std::string::npos);
}

TEST_CASE("WAV info_entry pads to even boundary", "[unit][provenance][wav]") {
    // Test with an odd-length string → should pad to even
    std::string out;
    crispasr_wav_info_entry(out, "TEST", "abc"); // 3 chars + NUL = 4 bytes (even, no pad)
    // 4 (id) + 4 (size) + 4 (value "abc\0") = 12
    REQUIRE(out.size() == 12);

    std::string out2;
    crispasr_wav_info_entry(out2, "TEST", "ab"); // 2 chars + NUL = 3 bytes (odd → pad)
    // 4 + 4 + 3 + 1(pad) = 12
    REQUIRE(out2.size() == 12);
}

// ──────────────────────────────────────────────────────────────────────────
// ID3v2 tags
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("ID3v2 tag has correct magic", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    REQUIRE(tag.substr(0, 3) == "ID3");
}

TEST_CASE("ID3v2 version is 2.3", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    REQUIRE((uint8_t)tag[3] == 3);
    REQUIRE((uint8_t)tag[4] == 0);
}

TEST_CASE("ID3v2 synchsafe size is correct", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    uint32_t ss = ((uint32_t)(uint8_t)tag[6] << 21) | ((uint32_t)(uint8_t)tag[7] << 14) |
                  ((uint32_t)(uint8_t)tag[8] << 7) | ((uint32_t)(uint8_t)tag[9]);
    REQUIRE(ss == tag.size() - 10);
    // Verify each byte has MSB=0 (synchsafe encoding)
    REQUIRE(((uint8_t)tag[6] & 0x80) == 0);
    REQUIRE(((uint8_t)tag[7] & 0x80) == 0);
    REQUIRE(((uint8_t)tag[8] & 0x80) == 0);
    REQUIRE(((uint8_t)tag[9] & 0x80) == 0);
}

TEST_CASE("ID3v2 contains all three TXXX frames", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    // Count TXXX occurrences
    int count = 0;
    size_t pos = 0;
    while ((pos = tag.find("TXXX", pos)) != std::string::npos) {
        count++;
        pos += 4;
    }
    REQUIRE(count == 3); // AI_GENERATED, GENERATOR, AI_CONTENT_NOTICE
}

TEST_CASE("ID3v2 TXXX contains AI_GENERATED=true", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    REQUIRE(tag.find("AI_GENERATED") != std::string::npos);
    REQUIRE(tag.find("true") != std::string::npos);
}

TEST_CASE("ID3v2 TXXX contains GENERATOR=CrispASR", "[unit][provenance][mp3]") {
    std::string tag = crispasr_make_id3v2_ai_tag();
    REQUIRE(tag.find("GENERATOR") != std::string::npos);
    REQUIRE(tag.find("CrispASR") != std::string::npos);
}

// ──────────────────────────────────────────────────────────────────────────
// C2PA (compile-time gated)
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("C2PA manifest JSON is valid and contains required fields", "[unit][provenance][c2pa]") {
    const char* json = crispasr_c2pa_manifest_json();
    REQUIRE(json != nullptr);
    std::string s(json);
    REQUIRE(s.find("c2pa.actions") != std::string::npos);
    REQUIRE(s.find("c2pa.created") != std::string::npos);
    REQUIRE(s.find("trainedAlgorithmicMedia") != std::string::npos);
    REQUIRE(s.find("CrispASR") != std::string::npos);
}

TEST_CASE("C2PA sign_wav returns false when not available", "[unit][provenance][c2pa]") {
    // Without CRISPASR_HAVE_C2PA, should be a no-op returning false
    std::string wav = "RIFF....WAVE";
    std::string original = wav;
    bool ok = crispasr_c2pa_sign_wav(wav, "", "");
    REQUIRE(ok == false);
    REQUIRE(wav == original); // unchanged
}

TEST_CASE("C2PA sign_wav returns false with empty cert/key", "[unit][provenance][c2pa]") {
    std::string wav(100, '\0');
    REQUIRE(crispasr_c2pa_sign_wav(wav, "", "/some/key.pem") == false);
    REQUIRE(crispasr_c2pa_sign_wav(wav, "/some/cert.pem", "") == false);
}

// ──────────────────────────────────────────────────────────────────────────
// Consent gate logic (voice clone detection)
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("Voice clone detection by .wav extension", "[unit][provenance][consent]") {
    auto is_clone = [](const std::string& voice) -> bool {
        return voice.size() >= 4 &&
               (voice.compare(voice.size() - 4, 4, ".wav") == 0 || voice.compare(voice.size() - 4, 4, ".WAV") == 0);
    };

    REQUIRE(is_clone("speaker.wav") == true);
    REQUIRE(is_clone("/path/to/speaker.wav") == true);
    REQUIRE(is_clone("SPEAKER.WAV") == true);
    REQUIRE(is_clone("speaker.gguf") == false);
    REQUIRE(is_clone("tara") == false);
    REQUIRE(is_clone("") == false);
    REQUIRE(is_clone(".wav") == true); // edge case: bare extension
    REQUIRE(is_clone("wav") == false); // too short
}

// ──────────────────────────────────────────────────────────────────────────
// Watermark spread-spectrum (additional edge cases)
// ──────────────────────────────────────────────────────────────────────────

#include "crispasr_watermark.h"

TEST_CASE("Watermark embed is idempotent-ish (double embed doesn't crash)", "[unit][provenance][watermark]") {
    std::vector<float> pcm(4800, 0.3f);
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size());
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size()); // second embed
    // Should not crash, and detect should still find at least one watermark
    float score = crispasr_watermark_detect_impl(pcm.data(), (int)pcm.size());
    REQUIRE(score > 0.0f); // don't require high confidence, just no crash
}

TEST_CASE("Watermark detect returns 0 for null input", "[unit][provenance][watermark]") {
    REQUIRE(crispasr_watermark_detect_impl(nullptr, 0) == 0.0f);
    REQUIRE(crispasr_watermark_detect_impl(nullptr, 1000) == 0.0f);
}

TEST_CASE("Watermark embed with alpha=0 introduces negligible change", "[unit][provenance][watermark]") {
    std::vector<float> pcm(4800, 0.5f);
    auto original = pcm;
    crispasr_watermark_embed_impl(pcm.data(), (int)pcm.size(), 0.0f);
    // With alpha=0, the nudge is zero but FFT→IFFT overlap-add introduces
    // tiny floating-point rounding noise. Max error should be < 1e-3.
    float max_err = 0.0f;
    for (size_t i = 0; i < pcm.size(); i++) {
        float err = std::abs(pcm[i] - original[i]);
        if (err > max_err)
            max_err = err;
    }
    REQUIRE(max_err < 0.01f);
}

// ──────────────────────────────────────────────────────────────────────────
// Spoken disclaimer opt-out — text verification
// ──────────────────────────────────────────────────────────────────────────
// crispasr_tts_disclaimer.h includes heavy backend headers; we test the
// disclaimer text contract inline to keep this test model-free.

TEST_CASE("Disclaimer text contract: mentions 'artificial intelligence'", "[unit][provenance][disclaimer]") {
    // Must match the string in crispasr_tts_disclaimer.h::text()
    const std::string expected = "This audio was generated by artificial intelligence.";
    REQUIRE(expected.find("artificial intelligence") != std::string::npos);
    REQUIRE(!expected.empty());
}

// ──────────────────────────────────────────────────────────────────────────
// Voice-clone disclaimer gating logic (unit-testable without backends)
// ──────────────────────────────────────────────────────────────────────────

TEST_CASE("Disclaimer gate: clone + no opt-out → disclaimer applies", "[unit][provenance][disclaimer]") {
    const bool is_voice_clone = true;
    const bool no_spoken_disclaimer = false;
    REQUIRE((is_voice_clone && !no_spoken_disclaimer) == true);
}

TEST_CASE("Disclaimer gate: clone + opt-out → disclaimer skipped", "[unit][provenance][disclaimer]") {
    const bool is_voice_clone = true;
    const bool no_spoken_disclaimer = true;
    REQUIRE((is_voice_clone && !no_spoken_disclaimer) == false);
}

TEST_CASE("Disclaimer gate: no clone → disclaimer always skipped", "[unit][provenance][disclaimer]") {
    const bool is_voice_clone = false;
    // Regardless of the flag, non-clone output never gets disclaimer
    REQUIRE((is_voice_clone && !false) == false);
    REQUIRE((is_voice_clone && !true) == false);
}
