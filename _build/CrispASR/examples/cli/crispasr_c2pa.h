// crispasr_c2pa.h — C2PA (Content Credentials) manifest embedding for TTS output.
//
// Compile-time gated on CRISPASR_HAVE_C2PA. When enabled, signs WAV
// output with a C2PA manifest declaring AI-generated provenance.
// Uses the c2pa-c library (C bindings for the Rust c2pa crate).
//
// When CRISPASR_HAVE_C2PA is not defined, the functions are no-ops
// that return the input unchanged (the WAV LIST/INFO + watermark
// still provide provenance).
//
// Certificate: self-signed X.509 is fine for compliance marking.
// Generate with scripts/generate-c2pa-cert.sh.

#pragma once

#include <cstdio>
#include <string>

#ifdef CRISPASR_HAVE_C2PA
#include <c2pa.h>
#endif

// The C2PA manifest JSON template. digitalSourceType per IPTC vocabulary
// indicates this is trained-algorithmic media (AI-generated).
inline const char* crispasr_c2pa_manifest_json() {
    return R"({
  "claim_generator": "CrispASR",
  "claim_generator_info": [{
    "name": "CrispASR",
    "version": "0.6"
  }],
  "assertions": [{
    "label": "c2pa.actions",
    "data": {
      "actions": [{
        "action": "c2pa.created",
        "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        "softwareAgent": "CrispASR TTS"
      }]
    }
  }, {
    "label": "c2pa.training-mining",
    "data": {
      "entries": [{
        "use": "notAllowed",
        "constraint_info": "This AI-generated audio should not be used to train AI models without explicit permission."
      }]
    }
  }]
})";
}

// Sign a WAV byte string with a C2PA manifest. Modifies `wav` in-place
// (the C2PA manifest is embedded as a RIFF chunk). Returns true on
// success. When c2pa-c is not available, returns false and leaves wav
// unchanged.
//
// cert_pem: path to X.509 certificate (PEM)
// key_pem:  path to private key (PEM)
inline bool crispasr_c2pa_sign_wav(std::string& wav, const std::string& cert_pem, const std::string& key_pem) {
    if (cert_pem.empty() || key_pem.empty())
        return false;

#ifdef CRISPASR_HAVE_C2PA
    // c2pa-c API: create a builder, set manifest JSON, sign from memory.
    //
    // The c2pa-c API evolves across versions. This targets v0.5+:
    //   c2pa_builder_new()             → create builder
    //   c2pa_builder_set_manifest()    → set manifest JSON
    //   c2pa_builder_sign_file()       → sign file on disk
    //   c2pa_builder_sign()            → sign from memory buffer (v0.6+)
    //
    // For versions that only support file-based signing, we write to a
    // temp file, sign, and read back.

    C2paBuilder* builder = c2pa_builder_new();
    if (!builder) {
        fprintf(stderr, "crispasr: C2PA builder creation failed\n");
        return false;
    }

    const char* manifest = crispasr_c2pa_manifest_json();
    int rc = c2pa_builder_set_manifest_json(builder, manifest);
    if (rc != 0) {
        fprintf(stderr, "crispasr: C2PA set_manifest failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        c2pa_builder_free(builder);
        return false;
    }

    // Write WAV to temp file, sign, read back
    char tmp_in[] = "/tmp/crispasr-c2pa-XXXXXX.wav";
    char tmp_out[] = "/tmp/crispasr-c2pa-out-XXXXXX.wav";
    int fd_in = mkstemps(tmp_in, 4);
    int fd_out = mkstemps(tmp_out, 4);
    if (fd_in < 0 || fd_out < 0) {
        fprintf(stderr, "crispasr: C2PA temp file creation failed\n");
        c2pa_builder_free(builder);
        return false;
    }
    write(fd_in, wav.data(), wav.size());
    close(fd_in);
    close(fd_out);

    rc = c2pa_builder_sign_file(builder, tmp_in, tmp_out, cert_pem.c_str(), key_pem.c_str());
    c2pa_builder_free(builder);

    if (rc != 0) {
        fprintf(stderr, "crispasr: C2PA signing failed: %s\n", c2pa_error() ? c2pa_error() : "unknown");
        unlink(tmp_in);
        unlink(tmp_out);
        return false;
    }

    // Read signed WAV back
    FILE* f = fopen(tmp_out, "rb");
    if (f) {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        wav.resize((size_t)sz);
        fread(&wav[0], 1, (size_t)sz, f);
        fclose(f);
    }
    unlink(tmp_in);
    unlink(tmp_out);
    return true;
#else
    (void)wav;
    (void)cert_pem;
    (void)key_pem;
    return false;
#endif
}

// Print a one-time startup warning if C2PA is not available.
inline void crispasr_c2pa_startup_check() {
#ifndef CRISPASR_HAVE_C2PA
    static bool warned = false;
    if (!warned) {
        fprintf(stderr, "crispasr-server: C2PA signing disabled (c2pa-c library not found; "
                        "install from github.com/contentauth/c2pa-c and rebuild with "
                        "-DCMAKE_PREFIX_PATH pointing at the install prefix)\n");
        warned = true;
    }
#endif
}
