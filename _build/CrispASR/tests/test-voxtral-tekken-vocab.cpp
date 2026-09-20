// tests/test-voxtral-tekken-vocab.cpp — #338.
//
// The bug was input-dependent: only texts whose BPE merge path reached an
// inactive tail entry produced an out-of-range token id, so every smoke test
// passed. What makes it testable is that the rule is arithmetic — a token id
// the encoder can emit must be a legal row index into the embedding table —
// and that rule holds for any blob, no model required.

#include "voxtral_tekken_vocab.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace {

// Pack pieces the way the GGUF blob stores them: [u16 len][len bytes] …
std::vector<uint8_t> pack(const std::vector<std::string>& pieces) {
    std::vector<uint8_t> blob;
    for (const auto& s : pieces) {
        const uint16_t len = (uint16_t)s.size();
        blob.push_back((uint8_t)(len & 0xFF));
        blob.push_back((uint8_t)(len >> 8));
        blob.insert(blob.end(), s.begin(), s.end());
    }
    return blob;
}

std::vector<std::string> synth_pieces(int n) {
    std::vector<std::string> v;
    v.reserve(n);
    for (int i = 0; i < n; i++)
        v.push_back("p" + std::to_string(i));
    return v;
}

int max_id(const std::map<std::string, int>& m) {
    int best = -1;
    for (const auto& kv : m)
        best = kv.second > best ? kv.second : best;
    return best;
}

} // namespace

TEST_CASE("active BPE count is the ids left after the specials", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    // Voxtral-4B-TTS-2603: 131072-wide embedding, 1000 specials.
    REQUIRE(active_bpe_count(131072, 1000) == 130072);
    // Degenerate headers yield 0, never a negative count that would later be
    // compared against as if it meant "unlimited".
    REQUIRE(active_bpe_count(0, 1000) == 0);
    REQUIRE(active_bpe_count(-5, 1000) == 0);
    REQUIRE(active_bpe_count(1000, 1000) == 0);
    REQUIRE(active_bpe_count(131072, -1) == 0);
}

TEST_CASE("range predicate matches the embedding table", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    REQUIRE(token_id_in_range(0, 131072));
    REQUIRE(token_id_in_range(131071, 131072));
    REQUIRE_FALSE(token_id_in_range(131072, 131072)); // one past the last row
    REQUIRE_FALSE(token_id_in_range(-1, 131072));
}

// The regression itself. A blob longer than the embedding table is not
// malformed — Mistral ships them that way — so the decoder must keep the tail
// out of the encoder's map rather than reject the blob.
TEST_CASE("a blob longer than the embedding table cannot emit an unusable id", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    constexpr int kVocab = 1100; // stand-in for 131072
    constexpr int kSpecials = 100;
    const int limit = active_bpe_count(kVocab, kSpecials); // 1000

    // 1200 serialized pieces for 1000 usable slots: 200 inert tail entries.
    const auto pieces = synth_pieces(1200);
    const auto blob = pack(pieces);
    const std::vector<std::string> specials(kSpecials, "");

    std::vector<std::string> id_to_piece;
    std::map<std::string, int> piece_to_id;
    const auto st = decode_blob(blob, kSpecials, specials, limit, id_to_piece, piece_to_id);

    REQUIRE(st.n_active == 1000);
    REQUIRE(st.n_inactive == 200);

    // The property that matters: nothing the encoder can look up indexes past
    // the embedding table. Before the fix the map held ids up to 1299.
    REQUIRE(max_id(piece_to_id) == kVocab - 1);
    for (const auto& kv : piece_to_id)
        REQUIRE(token_id_in_range(kv.second, kVocab));

    // The tail pieces specifically must be absent — that is the merge path the
    // reporter's Italian text happened to reach.
    REQUIRE(piece_to_id.count("p999") == 1);  // last active
    REQUIRE(piece_to_id.count("p1000") == 0); // first inactive
    REQUIRE(piece_to_id.count("p1199") == 0); // last serialized

    // The whole blob is still parsed, so a debug dump can show the inert tail.
    REQUIRE((int)id_to_piece.size() == kSpecials + 1200);
    REQUIRE(id_to_piece[kSpecials + 1000] == "p1000");
}

// The same blob decoded the way the runtime used to do it — no limit at all.
// This is what shipped, and it is the arm that must produce the bad ids;
// without it the test above could pass against a decoder that simply never
// emits anything, and nobody would notice.
TEST_CASE("the unbounded decode is what produced the out-of-range ids", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    constexpr int kVocab = 1100;
    constexpr int kSpecials = 100;

    const auto blob = pack(synth_pieces(1200));
    const std::vector<std::string> specials(kSpecials, "");

    std::vector<std::string> id_to_piece;
    std::map<std::string, int> piece_to_id;
    const auto st = decode_blob(blob, kSpecials, specials, /*active_limit*/ 0, id_to_piece, piece_to_id);

    REQUIRE(st.n_inactive == 0);          // nothing held back
    REQUIRE(max_id(piece_to_id) == 1299); // 200 ids past the embedding table
    REQUIRE_FALSE(token_id_in_range(max_id(piece_to_id), kVocab));
    REQUIRE(piece_to_id.count("p1000") == 1); // the tail piece is reachable
}

TEST_CASE("a blob that fits is untouched", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    constexpr int kVocab = 1100;
    constexpr int kSpecials = 100;

    const auto pieces = synth_pieces(500);
    const auto blob = pack(pieces);
    const std::vector<std::string> specials(kSpecials, "");

    std::vector<std::string> id_to_piece;
    std::map<std::string, int> piece_to_id;
    const auto st =
        decode_blob(blob, kSpecials, specials, active_bpe_count(kVocab, kSpecials), id_to_piece, piece_to_id);

    REQUIRE(st.n_active == 500);
    REQUIRE(st.n_inactive == 0);
    // Ids stay contiguous from n_specials — the bound must not perturb the
    // common case, or every existing checkpoint would retokenize differently.
    REQUIRE(piece_to_id.at("p0") == kSpecials);
    REQUIRE(piece_to_id.at("p499") == kSpecials + 499);
}

TEST_CASE("truncated and empty blobs decode without running off the end", "[unit][voxtral]") {
    using namespace voxtral_tekken;
    std::vector<std::string> id_to_piece;
    std::map<std::string, int> piece_to_id;

    // Length header promising more bytes than remain.
    std::vector<uint8_t> truncated = {0x05, 0x00, 'a', 'b'};
    auto st = decode_blob(truncated, 0, {}, 100, id_to_piece, piece_to_id);
    REQUIRE(st.n_active == 0);
    REQUIRE(piece_to_id.empty());

    // A trailing odd byte cannot form a length header.
    std::vector<uint8_t> odd = {0x01, 0x00, 'a', 0x02};
    st = decode_blob(odd, 0, {}, 100, id_to_piece, piece_to_id);
    REQUIRE(st.n_active == 1);
    REQUIRE(piece_to_id.at("a") == 0);

    st = decode_blob({}, 0, {}, 100, id_to_piece, piece_to_id);
    REQUIRE(st.n_active == 0);
    REQUIRE(id_to_piece.empty());
}
